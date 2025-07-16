import torch
import pytest
from unittest.mock import Mock
from polydiff.diffusion.forward import ExcitedStateDiffusion, DiffusionForward
from polydiff.diffusion.schedules import LinearSchedule

# Mock a simple schedule for testing purposes
class MockSchedule:
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        self.alphas_cumprod = torch.linspace(0.99, 0.01, num_timesteps)
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps)

    def get_parameters(self, t):
        alpha_t = self.alphas_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        return alpha_t, sqrt_alpha_t, sqrt_one_minus_alpha_t

@pytest.fixture
def excited_state_diffusion_instance():
    schedule = MockSchedule(num_timesteps=100)
    vocab_size = 1000
    mask_token_id = 101
    pad_token_id = 0
    device = torch.device("cpu")
    return ExcitedStateDiffusion(schedule, vocab_size, mask_token_id, pad_token_id, device)

@pytest.fixture
def diffusion_forward_instance():
    schedule = MockSchedule(num_timesteps=100)
    device = torch.device("cpu")
    return DiffusionForward(schedule, device)

def test_excited_state_diffusion_initialization(excited_state_diffusion_instance):
    assert excited_state_diffusion_instance.vocab_size == 1000
    assert excited_state_diffusion_instance.mask_token_id == 101
    assert excited_state_diffusion_instance.pad_token_id == 0
    assert excited_state_diffusion_instance.device == torch.device("cpu")
    assert isinstance(excited_state_diffusion_instance.schedule, MockSchedule)

def test_forward_mask_process_no_mask(excited_state_diffusion_instance):
    input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    t = torch.tensor([0, 0]) # t=0 means no masking
    masked_tokens, mask_positions = excited_state_diffusion_instance.forward_mask_process(input_ids, t)
    assert torch.equal(masked_tokens, input_ids)
    assert torch.equal(mask_positions, torch.zeros_like(input_ids, dtype=torch.bool))

def test_forward_mask_process_full_mask(excited_state_diffusion_instance):
    input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    t = torch.tensor([excited_state_diffusion_instance.schedule.num_timesteps - 1, excited_state_diffusion_instance.schedule.num_timesteps - 1]) # t=max means full masking
    masked_tokens, mask_positions = excited_state_diffusion_instance.forward_mask_process(input_ids, t)
    expected_masked_tokens = input_ids.clone()
    expected_masked_tokens[input_ids != excited_state_diffusion_instance.pad_token_id] = excited_state_diffusion_instance.mask_token_id
    assert torch.equal(masked_tokens, expected_masked_tokens)
    assert torch.equal(mask_positions, (input_ids != excited_state_diffusion_instance.pad_token_id))

def test_forward_mask_process_partial_mask(excited_state_diffusion_instance):
    input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    t = torch.tensor([excited_state_diffusion_instance.schedule.num_timesteps // 2, excited_state_diffusion_instance.schedule.num_timesteps // 2])
    masked_tokens, mask_positions = excited_state_diffusion_instance.forward_mask_process(input_ids, t)
    # Assert that some tokens are masked and some are not
    assert not torch.equal(masked_tokens, input_ids)
    assert not torch.equal(masked_tokens, torch.full_like(input_ids, excited_state_diffusion_instance.mask_token_id))
    # Ensure pad tokens are not masked
    assert torch.all((input_ids == excited_state_diffusion_instance.pad_token_id) == (masked_tokens == excited_state_diffusion_instance.pad_token_id))
    # Ensure mask positions are correct (non-pad tokens that are masked)
    assert torch.all((mask_positions == True) == (masked_tokens == excited_state_diffusion_instance.mask_token_id))

def test_forward_mask_process_with_different_t(excited_state_diffusion_instance):
    input_ids = torch.tensor([[1, 2, 3, 0, 0]])
    t_low = torch.tensor([0])
    t_high = torch.tensor([excited_state_diffusion_instance.schedule.num_timesteps - 1])

    masked_low, _ = excited_state_diffusion_instance.forward_mask_process(input_ids, t_low)
    masked_high, _ = excited_state_diffusion_instance.forward_mask_process(input_ids, t_high)

    assert torch.equal(masked_low, input_ids) # No masking at t=0
    expected_masked_high = input_ids.clone()
    expected_masked_high[input_ids != excited_state_diffusion_instance.pad_token_id] = excited_state_diffusion_instance.mask_token_id
    assert torch.equal(masked_high, expected_masked_high) # Full masking at t=max for non-pad tokens

def test_q_sample(diffusion_forward_instance):
    x_start = torch.randn(2, 5)
    t = torch.tensor([10, 20])
    x_t, noise = diffusion_forward_instance.q_sample(x_start, t)
    assert x_t.shape == x_start.shape
    assert noise.shape == x_start.shape

    # Test with pre-generated noise
    pre_noise = torch.randn_like(x_start)
    x_t_pre, noise_pre = diffusion_forward_instance.q_sample(x_start, t, noise=pre_noise)
    assert torch.equal(noise_pre, pre_noise)

def test_q_posterior_mean_variance(diffusion_forward_instance):
    x_start = torch.randn(2, 5)
    x_t = torch.randn(2, 5)
    t = torch.tensor([10, 20])
    mean, variance = diffusion_forward_instance.q_posterior_mean_variance(x_start, x_t, t)
    assert mean.shape == x_start.shape
    assert variance.shape == x_start.shape

    # Test with single timestep
    t_single = 50
    mean_single, variance_single = diffusion_forward_instance.q_posterior_mean_variance(x_start, x_t, t_single)
    assert mean_single.shape == x_start.shape
    assert variance_single.shape == x_start.shape

def test_get_mask_probabilities(excited_state_diffusion_instance):
    timesteps = torch.tensor([0, 50, 99])
    mask_probs = excited_state_diffusion_instance.get_mask_probabilities(timesteps)
    assert mask_probs.shape == timesteps.shape
    assert torch.all(mask_probs >= 0)
    assert torch.all(mask_probs <= 1)

def test_progressive_masking(excited_state_diffusion_instance):
    x_start = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]])
    timesteps = [0, 25, 50, 75, 99]
    results = excited_state_diffusion_instance.progressive_masking(x_start, timesteps)
    assert len(results) == len(timesteps)
    for res in results:
        assert "timestep" in res
        assert "masked_tokens" in res
        assert "mask_positions" in res
        assert "mask_probability" in res
        assert "num_masked" in res
        assert res["masked_tokens"].shape == x_start.shape
        assert res["mask_positions"].shape == x_start.shape
        assert isinstance(res["mask_probability"], float)
        assert isinstance(res["num_masked"], list)
        assert len(res["num_masked"]) == x_start.shape[0]