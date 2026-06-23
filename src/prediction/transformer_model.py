# src/prediction/transformer_model.py
import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TrajectoryTransformer(nn.Module):
    """
    Transformer model for predicting future movement trajectories.
    Input:  observed positions  — shape (batch, obs_len,  2)
    Output: predicted positions — shape (batch, pred_len, 2)
    """

    def __init__(
        self,
        obs_len:            int   = 8,
        pred_len:           int   = 12,
        d_model:            int   = 64,
        nhead:              int   = 8,
        num_encoder_layers: int   = 3,
        num_decoder_layers: int   = 3,
        dim_feedforward:    int   = 256,
        dropout:            float = 0.1,
    ):
        super().__init__()
        self.obs_len  = obs_len
        self.pred_len = pred_len
        self.d_model  = d_model

        self.input_projection = nn.Linear(2, d_model)
        self.encoder_pos = PositionalEncoding(d_model, dropout, max_len=obs_len + 10)
        self.decoder_pos = PositionalEncoding(d_model, dropout, max_len=pred_len + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, norm=nn.LayerNorm(d_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers, norm=nn.LayerNorm(d_model),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )
        self.query_embed = nn.Embedding(pred_len, d_model)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, teacher_forcing: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = src.size(0)

        src_emb = self.input_projection(src)
        src_emb = self.encoder_pos(src_emb)
        memory  = self.encoder(src_emb)

        if teacher_forcing is not None:
            start_token = src[:, -1:, :]
            decoder_input = torch.cat([start_token, teacher_forcing[:, :-1, :]], dim=1)
            tgt_emb = self.input_projection(decoder_input)
        else:
            query_ids = torch.arange(self.pred_len, device=src.device)
            tgt_emb = self.query_embed(query_ids)
            tgt_emb = tgt_emb.unsqueeze(0).expand(batch_size, -1, -1)

        tgt_emb = self.decoder_pos(tgt_emb)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.pred_len, device=src.device)
        decoded = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.output_projection(decoded)

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(obs, teacher_forcing=None)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: dict) -> TrajectoryTransformer:
    cfg  = config["model"]
    traj = config["trajectory"]
    return TrajectoryTransformer(
        obs_len            = traj["obs_len"],
        pred_len           = traj["pred_len"],
        d_model            = cfg["d_model"],
        nhead              = cfg["nhead"],
        num_encoder_layers = cfg["num_encoder_layers"],
        num_decoder_layers = cfg["num_decoder_layers"],
        dim_feedforward    = cfg["dim_feedforward"],
        dropout            = cfg["dropout"],
    )


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = build_model(config)
    print(f"Parameters : {model.count_parameters():,}")
    batch = torch.randn(4, model.obs_len, 2)
    out   = model.predict(batch)
    print(f"Input  : {tuple(batch.shape)}")
    print(f"Output : {tuple(out.shape)}")
    print("✅ Model OK")