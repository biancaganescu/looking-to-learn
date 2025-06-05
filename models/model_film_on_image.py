from model import *


class DualStreamTransformerFiLMOnImage(DualStreamTransformer):
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
            image_encoded = self.image_encoder(
                image_embedded, text_memory=embedded, text_key_padding_mask=padding_mask
            )
        else:
            image_encoded = None

        seq_len = embedded.size(1)

        tgt_mask = self.decoder.generate_square_subsequent_mask(seq_len).to(
            embedded.device
        )

        decoder_output = self.decoder(
            tgt=embedded,
            image_memory=image_encoded,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask,
        )

        output = self.output_layer(decoder_output)

        return output

    class FiLM(nn.Module):
        def __init__(self, d_model: int, condition_dim: int):
            super().__init__()
            self.gamma = nn.Linear(condition_dim, d_model)
            self.beta = nn.Linear(condition_dim, d_model)

        def forward(self, x, condition):
            gamma = self.gamma(condition)
            beta = self.beta(condition)
            if gamma.dim() == 2:
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            output = x * gamma + beta
            return output

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
            self.film = self.FiLM(d_model, d_model)
            self.norm = nn.LayerNorm(d_model)

        def forward(
            self,
            src,
            src_mask=None,
            src_key_padding_mask=None,
            text_memory=None,
            text_key_padding_mask=None,
        ):
            if text_memory is not None:
                if text_key_padding_mask is not None:
                    masked_text = text_memory.masked_fill(
                        text_key_padding_mask.unsqueeze(-1), 0
                    )
                    text_pooled = masked_text.sum(dim=1) / (~text_key_padding_mask).sum(
                        dim=1, keepdim=True
                    )
                else:
                    text_pooled = text_memory.mean(dim=1)

                src = self.film(src, text_pooled)

            return self.encoder(src, src_mask, src_key_padding_mask)
