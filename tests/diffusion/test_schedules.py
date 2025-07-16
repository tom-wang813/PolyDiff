import torch
import pytest
from polydiff.diffusion.schedules import LinearSchedule, CosineSchedule, QuadraticSchedule, ExponentialSchedule, LogarithmicSchedule

def test_linear_schedule_initialization():
    schedule = LinearSchedule(num_timesteps=1000)
    assert schedule.num_timesteps == 1000
    assert schedule.beta_start == 0.0001
    assert schedule.beta_end == 0.02

def test_linear_schedule_betas():
    schedule = LinearSchedule(num_timesteps=10)
    betas = schedule.betas
    assert isinstance(betas, torch.Tensor)
    assert betas.shape == (10,)
    assert torch.isclose(betas[0], torch.tensor(0.0001))
    assert torch.isclose(betas[-1], torch.tensor(0.02))

def test_linear_schedule_alphas():
    schedule = LinearSchedule(num_timesteps=10)
    alphas = schedule.alphas
    assert isinstance(alphas, torch.Tensor)
    assert alphas.shape == (10,)
    assert torch.all(alphas > 0)

def test_linear_schedule_alphas_cumprod():
    schedule = LinearSchedule(num_timesteps=10)
    alphas_cumprod = schedule.alphas_cumprod
    assert isinstance(alphas_cumprod, torch.Tensor)
    assert alphas_cumprod.shape == (10,)
    assert torch.all(alphas_cumprod > 0)
    assert torch.all(alphas_cumprod <= 1)

def test_linear_schedule_get_parameters():
    schedule = LinearSchedule(num_timesteps=10)
    alpha_t, sqrt_alpha_t, sqrt_one_minus_alpha_t = schedule.get_parameters(5)
    assert isinstance(alpha_t, torch.Tensor)
    assert isinstance(sqrt_alpha_t, torch.Tensor)
    assert isinstance(sqrt_one_minus_alpha_t, torch.Tensor)
    assert alpha_t.shape == ()
    assert sqrt_alpha_t.shape == ()
    assert sqrt_one_minus_alpha_t.shape == ()

    # Test with tensor input
    t_tensor = torch.tensor([0, 5, 9])
    alpha_t_batch, _, _ = schedule.get_parameters(t_tensor)
    assert alpha_t_batch.shape == (3,)

def test_cosine_schedule_betas():
    schedule = CosineSchedule(num_timesteps=10)
    betas = schedule.betas
    assert isinstance(betas, torch.Tensor)
    assert betas.shape == (10,)
    assert torch.all(betas >= 0)
    assert torch.all(betas <= 0.999)

def test_quadratic_schedule_betas():
    schedule = QuadraticSchedule(num_timesteps=10)
    betas = schedule.betas
    assert isinstance(betas, torch.Tensor)
    assert betas.shape == (10,)
    assert torch.isclose(betas[0], torch.tensor(0.0001))
    assert torch.isclose(betas[-1], torch.tensor(0.02))

def test_exponential_schedule_betas():
    schedule = ExponentialSchedule(num_timesteps=10)
    betas = schedule.betas
    assert isinstance(betas, torch.Tensor)
    assert betas.shape == (10,)
    assert torch.all(betas >= 0)
    assert torch.all(betas <= 0.999)

def test_logarithmic_schedule_betas():
    schedule = LogarithmicSchedule(num_timesteps=10)
    betas = schedule.betas
    assert isinstance(betas, torch.Tensor)
    assert betas.shape == (10,)
    assert torch.all(betas > 0)
    assert torch.all(betas <= 0.999)
    # Check if betas are increasing (logarithmically)
    assert torch.all(betas[1:] > betas[:-1])

def test_logarithmic_schedule_beta_start_zero_raises_error():
    with pytest.raises(ValueError, match="beta_start must be greater than 0"): # noqa: E501
        LogarithmicSchedule(num_timesteps=10, beta_start=0.0)
