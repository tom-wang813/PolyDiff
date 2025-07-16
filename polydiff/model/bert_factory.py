# 用於快速構建中型BERT diffusion模型的工廠方法
from transformers import AutoTokenizer

from polydiff.model.bert_diffusion_model import BertDiffusionModel


def build_medium_bert_diffusion_model(tokenizer=None, diffusion_steps=100):
    """
    構建一個中型BERT diffusion模型，配置如下：
    - 層數: 6
    - hidden: 512
    - head: 8
    - mlp: 2048
    - max len: 512
    """
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        "seyonec/ChemBERTa-zinc-base-v1", use_fast=True
    )
    model = BertDiffusionModel(
        vocab_size=len(tokenizer),
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
        diffusion_steps=diffusion_steps,
    )
    return model, tokenizer
