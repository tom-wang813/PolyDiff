# Diffusion Module (`polydiff/diffusion`)

The `polydiff/diffusion` module contains the core logic for the discrete diffusion process, including various noise schedules and the forward diffusion mechanism. This is where the mathematical foundation of the diffusion model is implemented.

## Key Components

### `DiffusionSchedule` (Abstract Base Class)

`DiffusionSchedule` defines the abstract interface for different types of noise schedules used in diffusion models. Subclasses must implement the `_setup_schedule` method to define how the beta values (noise levels) are generated across timesteps.

-   **Purpose:** To provide a standardized way to define and manage noise schedules.
-   **Initialization (`__init__`):**
    -   `num_timesteps` (int): The total number of diffusion steps.
-   **Properties:**
    -   `betas` (torch.Tensor): The noise schedule beta values.
    -   `alphas` (torch.Tensor): The alpha values (1 - beta).
    -   `alphas_cumprod` (torch.Tensor): The cumulative product of alpha values.
-   **Methods:**
    -   `_setup_schedule()`: Abstract method to be implemented by subclasses. Sets up the noise schedule by populating `_betas`.
    -   `get_parameters(t)`: Returns diffusion parameters (alpha_t, sqrt_alpha_t, sqrt_one_minus_alpha_t) for a given timestep `t`.

### Concrete Schedule Implementations

-   **`LinearSchedule`:** Implements a linear beta schedule, where noise increases linearly over timesteps.
-   **`CosineSchedule`:** Implements a cosine beta schedule, often used for smoother noise addition and better performance.
-   **`QuadraticSchedule`:** Implements a quadratic beta schedule.
-   **`ExponentialSchedule`:** Implements an exponential beta schedule.

### `DiffusionForward`

`DiffusionForward` is a base class for the forward diffusion process, primarily designed for continuous data (though it can be adapted). It implements the `q_sample` and `q_posterior_mean_variance` methods, which are fundamental to the diffusion process.

-   **Purpose:** To define how noise is added to data in the forward pass and how posterior distributions are calculated.
-   **Initialization (`__init__`):**
    -   `schedule` (DiffusionSchedule): An instance of a diffusion schedule.
    -   `device` (str or torch.device): The device (CPU/GPU) to perform computations on.
-   **Methods:**
    -   `q_sample(x_start, t, noise=None)`: Adds noise to the input `x_start` at timestep `t` to produce `x_t`.
    -   `q_posterior_mean_variance(x_start, x_t, t)`: Computes the mean and variance of the diffusion posterior `q(x_{t-1} | x_t, x_0)`.

### `ExcitedStateDiffusion`

`ExcitedStateDiffusion` implements a Markov-style mask diffusion process, specifically designed for discrete data like SMILES tokens. It's characterized by an irreversible masking process, meaning once a token is masked, it remains masked.

-   **Purpose:** To apply a controlled masking process to discrete tokens based on the diffusion timestep.
-   **Initialization (`__init__`):**
    -   `schedule` (DiffusionSchedule): An instance of a diffusion schedule.
    -   `vocab_size` (int): The size of the vocabulary.
    -   `mask_token_id` (Optional[int]): The ID of the mask token. Defaults to `vocab_size - 1` if not provided.
    -   `pad_token_id` (Optional[int]): The ID of the padding token. Tokens with this ID are not masked.
    -   `device` (str or torch.device): The device (CPU/GPU) to perform computations on.
-   **Methods:**
    -   `forward_mask_process(x_start, t, mask_noise=None)`: Applies masking to the input `x_start` based on timestep `t`. Returns the `masked_tokens` and a boolean `mask_positions` tensor indicating which tokens were masked.
    -   `get_mask_probabilities(timesteps)`: Returns the probability of a token being masked at given timesteps.
    -   `progressive_masking(x_start, timesteps, fixed_seed=None)`: A utility method to observe the masking process at different timesteps using a fixed mask noise, useful for visualization or debugging.

## Development Guide

### Implementing New Schedules

To add a new noise schedule, create a new class that inherits from `DiffusionSchedule` and implements the `_setup_schedule` method. Ensure your `_setup_schedule` method correctly populates `self._betas` with a `torch.Tensor` of shape `(num_timesteps,)`.

### Modifying Diffusion Processes

-   **`DiffusionForward`:** If you need to change how noise is added or how posterior calculations are performed for continuous data, modify the `q_sample` or `q_posterior_mean_variance` methods.
-   **`ExcitedStateDiffusion`:** For discrete data, modifications to the `forward_mask_process` would involve changing the masking strategy (e.g., probabilistic masking, different masking patterns). Ensure that `pad_token_id` handling remains consistent if you want to avoid masking padding tokens.

### Testing

-   **Unit Tests:** Refer to `tests/diffusion/test_schedules.py` and `tests/diffusion/test_forward.py` for examples of how to test individual schedule implementations and diffusion processes. Pay close attention to edge cases (e.g., `t=0`, `t=max_timesteps`).
-   **Coverage:** When making changes, run `pytest --cov=polydiff.diffusion --cov-report=term-missing` to ensure your changes are adequately covered by tests.
