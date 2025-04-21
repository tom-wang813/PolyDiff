import torch
import torch.nn as nn
import math

from PolyDiff.configs import model_config, diffusion_config  


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        # 使用配置中的 max_len 和 d_model，而不是硬编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))  # 不需要额外的配置
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class DiffusionBertModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 从配置中读取各项配置，保持常量为大写，但在类中使用小写
        self.vocab_size = model_config.VOCAB_SIZE
        self.d_model = model_config.D_MODEL
        self.max_seq_length = model_config.MAX_SEQ_LENGTH
        self.nhead = model_config.NHEAD
        self.dim_feedforward = model_config.DIM_FEEDFORWARD
        self.dropout = model_config.DROPOUT
        self.num_layers = model_config.NUM_LAYERS
        self.mask_token_id = model_config.MASK_TOKEN_ID
        self.pad_token_id = model_config.PAD_TOKEN_ID

        self.max_timesteps = diffusion_config.MAX_TIMESTEPS

        # 使用配置中的参数，而不是硬编码
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
        self.timestep_embedding = nn.Embedding(self.max_timesteps, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.output_head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, input_ids, timestep):
        """
        input_ids: [batch, seq_len]
        timestep: [batch] (每个 batch 的 diffusion timestep)
        """
        # 自动生成 attention mask
        attention_mask = (input_ids != self.pad_token_id).long()  # 1 为有效 token, 0 为 padding token
        
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        x = self.pos_encoding(x)
        
        # 加入时间步嵌入
        t_emb = self.timestep_embedding(timestep).unsqueeze(1)
        x = x + t_emb
        x = self.dropout_layer(x)
        
        # 处理 key_padding_mask：True 表示 padding 的位置
        key_padding_mask = (attention_mask == 0)
        
        # Transformer Encoder (batch_first=True)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        
        # 输出 logits，形状为 [batch, seq_len, vocab_size]
        logits = self.output_head(x)
        return logits
