from model import *
from torchvision.ops import SqueezeExcitation


class DualStreamTransformerChannelAttention(DualStreamTransformer):
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

    class Encoder(nn.Module):
        def __init__(
            self,
            d_model: int,
            n_head: int,
            d_hid: int,
            n_layers: int,
            dropout: float = 0.1,
        ):
            super().__init__()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, n_head, d_hid, dropout, activation="gelu", batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
            self.se = SqueezeExcitation(d_model, d_model // 16)

        def forward(self, src, src_mask=None, src_key_padding_mask=None):
            output = self.encoder(src, src_mask, src_key_padding_mask)
            output = output.permute(0, 2, 1).unsqueeze(2)
            output = self.se(output)
            output = output.squeeze(2).permute(0, 2, 1)
            return output
