from model import *


class DualStreamTransformerMLPEncoder(DualStreamTransformer):
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

    def forward(
        self, input_ids, dino_embedding=None, padding_mask=None, use_image: bool = False
    ):
        embedded = self.text_embedding(input_ids)

        if (
            use_image
            and dino_embedding is not None
            and not torch.all(dino_embedding == 0)
        ):
            image_embedded = self.image_embedding(dino_embedding)
        else:
            image_embedded = None

        seq_len = embedded.size(1)

        tgt_mask = self.decoder.generate_square_subsequent_mask(seq_len).to(
            embedded.device
        )

        decoder_output = self.decoder(
            tgt=embedded,
            image_memory=image_embedded,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask,
        )

        output = self.output_layer(decoder_output)

        return output

    class DinoImageEmbedding(nn.Module):
        def __init__(
            self,
            dino_dim: int,
            d_model: int,
            hidden_dim: int = None,
            dropout: float = 0.1,
        ):
            super().__init__()
            hidden_dim = hidden_dim or d_model
            self.proj = nn.Sequential(
                nn.Linear(dino_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
            )
            self.layer_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out = self.proj(x)
            out = self.dropout(out)
            out = self.layer_norm(out)
            return out.unsqueeze(1)
