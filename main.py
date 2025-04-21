import torch
from PolyDiff.model import DiffusionBertModel
from PolyDiff.configs import model_config, diffusion_config

def main():
    # 设置模型
    model = DiffusionBertModel()

    # 创建虚拟数据
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, model_config.VOCAB_SIZE, (batch_size, seq_len))  # 随机生成一些 token IDs
    timestep = torch.randint(0, diffusion_config.MAX_TIMESTEPS, (batch_size,))  # 随机生成时间步长
    
    # 打印输入数据
    print(f"input_ids: {input_ids}")
    print(f"timestep: {timestep}")

    # 模型推理
    logits = model(input_ids, timestep)
    
    # 打印模型输出
    print(f"Logits: {logits.shape}")
    print(logits)

if __name__ == "__main__":
    main()
