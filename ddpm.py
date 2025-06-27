import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(input, t: torch.Tensor, x: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


class BaseScheduler(nn.Module):
    """
    Variance scheduler of DDPM.
    """

    def __init__(
            self,
            num_train_timesteps: int,
            beta_1: float = 1e-4,
            beta_T: float = 0.02,
            mode: str = "linear",
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                    torch.linspace(beta_1 ** 0.5, beta_T ** 0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)


class DiffusionModule(nn.Module):
    """
    A high-level wrapper of DDPM and DDIM.
    If you want to sample data based on the DDIM's reverse process, use `ddim_p_sample()` and `ddim_p_sample_loop()`.
    """

    def __init__(self, network: nn.Module, var_scheduler: BaseScheduler):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        # For image diffusion model.
        return getattr(self.network, "image_resolution", None)

    def q_sample(self, x0, t, noise=None):
        """
        sample x_t from q(x_t | x_0) of DDPM.

        Input:
            x0 (`torch.Tensor`): clean data to be mapped to timestep t in the forward process of DDPM.
            t (`torch.Tensor`): timestep
            noise (`torch.Tensor`, optional): random Gaussian noise. if None, randomly sample Gaussian noise in the function.
        Output:
            xt (`torch.Tensor`): noisy samples
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Forward diffusion process: x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        alphas_prod_t = extract(self.var_scheduler.alphas_cumprod, t, x0)
        xt = torch.sqrt(alphas_prod_t) * x0 + torch.sqrt(1 - alphas_prod_t) * noise

        return xt

    @torch.no_grad()
    def p_sample(self, xt, t):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.

        Input:
            xt (`torch.Tensor`): samples at arbitrary timestep t.
            t (`torch.Tensor`): current timestep in a reverse process.
        Output:
            x_t_prev (`torch.Tensor`): one step denoised sample. (= x_{t-1})

        """
        if isinstance(t, int):
            t = torch.tensor([t]).to(self.device)

        # Predict noise using the network
        eps_theta = self.network(xt, t)

        # Calculate the mean of the reverse process
        eps_factor = (1 - extract(self.var_scheduler.alphas, t, xt)) / (
                1 - extract(self.var_scheduler.alphas_cumprod, t, xt)
        ).sqrt()

        mean = (xt - eps_factor * eps_theta) / extract(self.var_scheduler.alphas, t, xt).sqrt()

        # Add noise if not at the final step (t > 0)
        if t[0].item() > 0:  # Check the first element since all elements should be the same timestep
            posterior_variance = extract(self.var_scheduler.betas, t, xt)
            noise = torch.randn_like(xt)
            x_t_prev = mean + posterior_variance.sqrt() * noise
        else:
            x_t_prev = mean

        return x_t_prev

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """
        The loop of the reverse process of DDPM.

        Input:
            shape (`Tuple`): The shape of output. e.g., (num particles, 2)
        Output:
            x0_pred (`torch.Tensor`): The final denoised output through the DDPM reverse process.
        """
        # Start from pure noise
        x = torch.randn(shape).to(self.device)

        # Iteratively denoise from T to 0
        for i in reversed(range(self.var_scheduler.num_train_timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t)

        x0_pred = x
        return x0_pred

    def compute_loss(self, x0):
        """
        The simplified noise matching loss corresponding Equation 14 in DDPM paper.

        Input:
            x0 (`torch.Tensor`): clean data
        Output:
            loss: the computed loss to be backpropagated.
        """
        batch_size = x0.shape[0]
        # Sample random timesteps for each sample in the batch
        t = torch.randint(0, self.var_scheduler.num_train_timesteps, size=(batch_size,)).to(x0.device).long()

        # Sample noise
        noise = torch.randn_like(x0)

        # Apply forward diffusion to get noisy samples
        xt = self.q_sample(x0, t, noise)

        # Predict the noise
        eps_theta = self.network(xt, t)

        # Compute MSE loss between predicted and actual noise
        loss = F.mse_loss(eps_theta, noise)

        return loss

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
        }
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)