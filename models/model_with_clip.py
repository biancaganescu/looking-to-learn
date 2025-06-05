from model import *
import torch.nn.functional as F


class DualStreamTransformerWithCLIP(DualStreamTransformer):
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
        # For contrastive loss
        self.image_projection = nn.Linear(d_model, d_model)
        self.text_projection = nn.Linear(d_model, d_model)

        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.07)))

    def forward(
        self,
        input_ids,
        dino_embedding=None,
        padding_mask=None,
        use_image: bool = False,
        return_pooled: bool = False,
    ):
        embedded = self.text_embedding(input_ids)

        if (
            use_image
            and dino_embedding is not None
            and not torch.all(dino_embedding == 0)
        ):
            image_embedded = self.image_embedding(dino_embedding)
            image_encoded = self.image_encoder(image_embedded)
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

        if return_pooled:
            text_unimodal = embedded.mean(dim=1)
            text_proj = self.text_projection(text_unimodal)

            image_proj = self.image_projection(image_encoded.squeeze(1))

            return output, text_proj, image_proj

        return output

    def compute_contrastive_loss(self, text_features, image_features, batch_size):
        tau = torch.exp(self.log_tau)
        text_features = F.normalize(text_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)

        logits = torch.matmul(text_features, image_features.t()) / tau
        labels = torch.arange(batch_size, device=logits.device)

        loss_image_to_text = F.cross_entropy(logits, labels)
        loss_text_to_image = F.cross_entropy(logits.t(), labels)

        return (loss_image_to_text + loss_text_to_image) / 2
