from model import *
import torch.nn.functional as F


class DualStreamTransformerGateHardPerToken(DualStreamTransformer):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_head: int = 8,
        d_hid: int = 768,
        num_encoder_layers: int = 5,
        num_decoder_layers: int = 8,
        dino_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            n_head=n_head,
            d_hid=d_hid,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dino_dim=dino_dim,
            dropout=dropout,
        )

    class DynamicGating(nn.Module):
        def __init__(self, d_model: int, dropout: float = 0.1):
            super().__init__()
            self.gate = nn.Linear(d_model * 2, 2)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(d_model)
            self.init_temperature = 1.0
            self.min_temperature = 0.1
            # use 80% of the total image caption steps for annealing
            self.anneal_steps = 307408
            self.gate_step = 0
            self.temperature = self.init_temperature

        def set_temperature(self):
            if self.gate_step >= self.anneal_steps:
                self.temperature = self.min_temperature
            else:
                self.temperature = self.init_temperature - (
                    self.init_temperature - self.min_temperature
                ) * (self.gate_step / self.anneal_steps)

        def forward(self, text_features, image_features):
            if image_features is None:
                return text_features

            if self.training:
                self.gate_step += 1
                self.set_temperature()

            combined = torch.cat([text_features, image_features], dim=-1)

            logits = self.gate(combined)

            hard = not self.training

            gate = F.gumbel_softmax(logits, tau=self.temperature, hard=hard, dim=-1)
            gate = gate[..., 0].unsqueeze(-1)

            fused = gate * text_features + (1 - gate) * image_features
            fused = self.layer_norm(self.dropout(fused))
            return fused
