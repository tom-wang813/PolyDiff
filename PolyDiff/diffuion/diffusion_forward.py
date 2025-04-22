import torch
import math

from abc import ABC, abstractmethod


class BaseSchedule(ABC):
    """
    BaseSchedule defines the common functionality for all noise schedules,
    including total diffusion steps, device, beta, and alpha_cumprod initialization.
    """

    def __init__(self, num_steps: int, device: str = 'cpu'):
        """
        Args:
            num_steps (int): Total number of diffusion steps T.
            device (str): Computation device, e.g., 'cpu' or 'cuda'.
        """
        self.num_steps = num_steps
        self.device = device
        self.beta: torch.Tensor = None
        self.alpha_cumprod: torch.Tensor = None
        self.init_schedule()

    @abstractmethod
    def init_schedule(self) -> None:
        """
        Initialize the schedule by setting up beta and alpha_cumprod.
        Must be implemented by subclasses.
        """
        pass


class LinearSchedule(BaseSchedule):
    """
    Linear schedule: beta increases linearly from 0 to 1.
    """

    def init_schedule(self) -> None:
        # Create beta tensor with shape (num_steps+1,) linearly spaced from 0 to 1.
        self.beta = torch.linspace(0, 1, steps=self.num_steps + 1, device=self.device)
        # Compute the cumulative product of (1 - beta) starting from step 1.
        self.alpha_cumprod = torch.cumprod(1 - self.beta[1:], dim=0)


class CosineSchedule(BaseSchedule):
    """
    Cosine schedule: Uses a cosine function to generate beta values,
    based on the cosine schedule suggested in improved diffusion models.
    """

    def init_schedule(self) -> None:
        steps = self.num_steps + 1
        x = torch.linspace(0, self.num_steps, steps, device=self.device)
        s = 0.008  # small offset as suggested in the literature
        # Compute alpha_bar using a cosine function.
        alpha_bar = torch.cos(((x / self.num_steps) + s) / (1 + s) * (math.pi / 2)) ** 2
        # Normalize alpha_bar so that alpha_bar[0] equals 1.
        alpha_bar = alpha_bar / alpha_bar[0]
        # Compute beta from the ratio of successive elements in alpha_bar.
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        betas = torch.clamp(betas, 0, 0.999)
        # Set beta[0] to 0 and concatenate with the computed betas.
        self.beta = torch.cat([torch.tensor([0.], device=self.device), betas])
        # Compute the cumulative product of (1 - beta) starting from step 1.
        self.alpha_cumprod = torch.cumprod(1 - self.beta[1:], dim=0)


class SigmoidSchedule(BaseSchedule):
    """
    Sigmoid schedule: Uses a sigmoid function to generate beta values.
    The beta values follow an S-shaped curve from 0 to 1.
    """

    def init_schedule(self) -> None:
        # Generate a tensor linearly spaced between -6 and 6.
        x = torch.linspace(-6, 6, self.num_steps + 1, device=self.device)
        # Apply sigmoid so that values are between 0 and 1.
        beta = torch.sigmoid(x)
        # Normalize so that beta[0] is exactly 0 and beta[-1] is 1.
        beta = (beta - beta[0]) / (beta[-1] - beta[0])
        # Ensure beta[0] is exactly 0 and limit the upper bound to 0.999.
        beta[0] = 0.
        beta = torch.clamp(beta, max=0.999)
        self.beta = beta
        # Compute the cumulative product of (1 - beta) starting from step 1.
        self.alpha_cumprod = torch.cumprod(1 - self.beta[1:], dim=0)


class AbsorbingDiffusion:
    """
    AbsorbingDiffusion model for forward diffusion (absorbing type).
    This model injects noise into parts of the token sequence based on a provided noise schedule.
    Once a token is masked, it remains masked (absorbing behavior).
    This vectorized method computes the cumulative masking probability without iterating through each timestep.
    """
    def __init__(self,
                 num_steps: int,
                 mask_token: int,
                 schedule,  # An instance of a noise schedule with a `beta` attribute.
                 device: str = 'cpu',
                 pad_token: int = None):
        """
        Args:
            num_steps (int): Total number of diffusion steps T.
            mask_token (int): Special token representing the noisy (masked) state.
            schedule: A noise schedule instance (should have a beta attribute of shape (num_steps+1,)).
            device (str): The computation device, e.g., 'cpu' or 'cuda'.
            pad_token (int, optional): Padding token that will not be noised.
        """
        self.num_steps = num_steps
        self.mask_token = mask_token
        self.schedule = schedule
        self.device = device
        self.pad_token = pad_token

    def get_eligible_mask(self, x: torch.Tensor, condition_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute an eligibility mask for tokens that can be updated.
        Eligible tokens are those that are not padding, not conditioned,
        and not already masked.
        
        Args:
            x (torch.Tensor): Token sequence with shape (batch_size, seq_len).
            condition_mask (torch.Tensor, optional): Boolean tensor (same shape as x) where True indicates condition tokens that should not be updated.
        
        Returns:
            torch.Tensor: A boolean tensor indicating the tokens eligible for noise injection.
        """
        # Exclude pad tokens if pad_token is provided.
        if self.pad_token is not None:
            eligible = (x != self.pad_token)
        else:
            eligible = torch.ones_like(x, dtype=torch.bool)
        
        # Exclude conditioned tokens if provided.
        if condition_mask is not None:
            eligible = eligible & (~condition_mask)
        
        # Exclude tokens that have already been masked.
        eligible = eligible & (x != self.mask_token)
        return eligible

    def __call__(self, x: torch.Tensor, t, condition_mask: torch.Tensor = None, rnd: torch.Tensor = None) -> torch.Tensor:
        """
        Perform vectorized forward diffusion simulation for t steps (absorbing updates).
        This version accepts t as an int or as a tensor (of shape (batch_size,)) so that different samples
        in the batch can have different diffusion timesteps.
        
        Args:
            x (torch.Tensor): Current token sequence with shape (batch_size, seq_len).
            t (int or torch.Tensor): Number of diffusion steps. If tensor, its shape should be (batch_size,).
            condition_mask (torch.Tensor, optional): Boolean tensor (same shape as x) indicating condition tokens to not update.
            rnd (torch.Tensor, optional): A tensor of random numbers. If not provided, one will be generated with the same shape as x.
        
        Returns:
            torch.Tensor: The updated token sequence after t diffusion steps.
        """

        torch.manual_seed(0)  # For reproducibility, set a manual seed.
        # If t is an integer, convert it to a tensor with shape (batch_size,)
        if isinstance(t, int):
            t_tensor = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)
        else:
            t_tensor = t  # Expected shape: (batch_size,)

        # If all t values are 0, return original x as no diffusion steps are performed.
        if (t_tensor == 0).all():
            return x

        # Calculate the eligibility mask for each sample (shape: (batch_size, seq_len)).
        eligible_orig = self.get_eligible_mask(x, condition_mask)
        # Compute the effective ratio (proportion of eligible tokens) for each sample.
        effective_ratio = eligible_orig.float().mean(dim=1)  # shape: (batch_size,)

        # Get the maximum timestep among the batch.
        t_max = int(t_tensor.max().item())

        # Assume self.schedule.beta is a tensor of shape (num_steps+1,) where beta[0] is not used.
        beta_slice = self.schedule.beta[1:t_max+1].to(x.device)  # shape: (t_max,)

        # Create a step index matrix (shape: (1, t_max)).
        steps = torch.arange(t_max, device=x.device).unsqueeze(0)
        # Create a mask of valid steps for each sample based on t_tensor (shape: (batch_size, t_max)).
        valid_steps = steps < t_tensor.unsqueeze(1)

        # Expand beta_slice and effective_ratio to compute update factors:
        # For valid steps s, factor = (1 - beta[s] * effective_ratio).
        beta_row = beta_slice.unsqueeze(0)  # shape: (1, t_max)
        eff_ratio_col = effective_ratio.unsqueeze(1)  # shape: (batch_size, 1)
        factors = 1 - beta_row * eff_ratio_col  # shape: (batch_size, t_max)
        # For invalid steps (s >= t for a sample), set factor to 1 so it does not affect the product.
        factors = torch.where(valid_steps, factors, torch.ones_like(factors))

        # Calculate the cumulative product along the time steps for each sample.
        cumprod = torch.prod(factors, dim=1)  # shape: (batch_size,)
        cumulative_prob = 1 - cumprod  # shape: (batch_size,)

        # Generate a random tensor if one is not provided.
        if rnd is None:
            rnd = torch.rand_like(x, dtype=torch.float)
        
        # Broadcast cumulative_prob to the full token sequence shape.
        cumulative_prob_expanded = cumulative_prob.unsqueeze(1)  # shape: (batch_size, 1)
        # Determine which tokens to update: eligible tokens with a random value less than the cumulative probability.
        update_mask = self.get_eligible_mask(x, condition_mask) & (rnd < cumulative_prob_expanded)
        
        # Clone x and update tokens to the mask_token where necessary.
        x_updated = x.clone()
        x_updated[update_mask] = self.mask_token

        return x_updated


if __name__ == "__main__":

    # Define parameters for the diffusion process.
    num_steps = 10
    mask_token = -1  # For example, -1 represents a masked token.
    pad_token = 0    # For example, 0 represents a padding token.
    
    # Create an instance of the dummy schedule.
    schedule = CosineSchedule(num_steps=num_steps, device='cpu')
    
    # Instantiate the diffusion model.
    diffusion_model = AbsorbingDiffusion(num_steps, mask_token, schedule, device='cpu', pad_token=pad_token)
    
    # Create a dummy input batch of token sequences (shape: batch_size x seq_len).
    # In this example, we use two samples with tokens represented by integers.
    x = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, pad_token, 4, 5, pad_token, 7, 8, 9]
    ], dtype=torch.long)
    
    # For demonstration, assume no condition mask is provided.
    condition_mask = None
    
    # Provide different diffusion timesteps for each sample (e.g., first sample: 5 steps; second sample: 8 steps).
    t_tensor = torch.tensor([5, 8], dtype=torch.long)
    
    # Call the vectorized_q_sample function to update tokens.
    updated_x = diffusion_model(x, t_tensor, condition_mask)
    
    # Print the original and updated token sequences.
    print("Original x:")
    print(x)
    print("Updated x:")
    print(updated_x)

