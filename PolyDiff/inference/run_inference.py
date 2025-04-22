# inference/run_inference.py
import argparse
import json
import torch
from inference.config import InferenceConfig
from inference.utils import load_model
from inference.sampler import sample_model
from PolyDiff.diffusion.schedule_factory import get_schedule

def main():
    parser = argparse.ArgumentParser(description="Run inference for PolyDiff discrete BERT diffusion")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config path (see inference/config.py)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="（選）HuggingFace tokenizer.json 的路徑，用來 decode token->SMILES")
    args = parser.parse_args()

    # 讀入設定
    cfg = InferenceConfig.from_yaml(args.config)
    # 載入模型、schedule
    model = load_model(cfg.checkpoint_path, device=cfg.device)
    schedule = get_schedule(device=cfg.device)

    # 決定序列長度
    seq_len = cfg.seq_len
    if seq_len is None:
        # 若不指定，就用 model_config.MAX_SEQ_LENGTH
        from PolyDiff.configs import model_config
        seq_len = model_config.MAX_SEQ_LENGTH

    # 取樣
    with torch.no_grad():
        samples = sample_model(
            model=model,
            schedule=schedule,
            num_samples=cfg.num_samples,
            seq_len=seq_len,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            device=cfg.device,
        )

    # decode
    if args.tokenizer:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
        smi_list = tok.batch_decode(samples.cpu().tolist(), skip_special_tokens=True)
    else:
        # 若無 tokenizer，就輸出原始 id 序列
        smi_list = [list(map(int, row)) for row in samples.cpu()]

    # 寫檔
    with open(cfg.output_path, "w", encoding="utf-8") as f:
        json.dump(smi_list, f, indent=2, ensure_ascii=False)
    print(f">>> Saved {len(smi_list)} samples to {cfg.output_path}")

if __name__ == "__main__":
    main()
