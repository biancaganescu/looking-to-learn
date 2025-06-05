from model import *


class DualStreamTransformerFiLMOnText(DualStreamTransformer):
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

    class FiLM(nn.Module):
        def __init__(self, d_model: int, condition_dim: int):
            super().__init__()
            self.gamma = nn.Linear(condition_dim, d_model)
            self.beta = nn.Linear(condition_dim, d_model)

        def forward(self, x, condition):
            gamma = self.gamma(condition)
            beta = self.beta(condition)
            return x * gamma + beta

    class MultimodalDecoderLayer(nn.Module):
        def __init__(self, d_model: int, n_head: int, d_hid: int, dropout: float = 0.1):
            super().__init__()
            # Self Attention
            self.self_attn = nn.MultiheadAttention(
                d_model, n_head, dropout=dropout, batch_first=True
            )
            # Cross Attention with Image
            self.cross_attn_txt_image = nn.MultiheadAttention(
                d_model, n_head, dropout=dropout, batch_first=True
            )

            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)

            self.dropout = nn.Dropout(dropout)

            self.film = self.FiLM(d_model, d_model)

            # Gating
            self.gate = self.DynamicGating(d_model, dropout)

            self.ff = nn.Sequential(
                nn.Linear(d_model, d_hid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hid, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, tgt, image_memory, tgt_mask=None, tgt_key_padding_mask=None):
            tgt_norm = self.norm1(tgt)
            self_attn_output, _ = self.self_attn(
                tgt_norm,
                tgt_norm,
                tgt_norm,
                key_padding_mask=tgt_key_padding_mask,
                attn_mask=tgt_mask,
                is_causal=True,
            )

            tgt = tgt + self.dropout(self_attn_output)

            if image_memory is not None:
                tgt_norm = self.norm2(tgt)
                cross_attn_output, _ = self.cross_attn_txt_image(
                    tgt_norm, image_memory, image_memory
                )
                cross_attn_output = self.dropout(cross_attn_output)

                tgt_modulated = self.film(tgt_norm, image_memory)

                fused = self.gate(tgt_modulated, cross_attn_output)
                tgt = tgt + fused

            tgt_norm = self.norm3(tgt)
            ff_output = self.ff(tgt_norm)
            tgt = tgt + self.dropout(ff_output)

            return tgt
