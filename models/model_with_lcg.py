from model import *
import torch.nn.functional as F


class DualStreamTransformerWithLCG(DualStreamTransformer):
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

        self.output_layer.weight = self.text_embedding.token_embedding.weight

        self.lcg_loss = self.LCGLoss(d_model)

    def forward(
        self,
        input_ids,
        dino_embedding=None,
        padding_mask=None,
        use_image: bool = False,
        return_embeddings: bool = False,
    ):
        embedded = self.text_embedding(input_ids)

        if (
            use_image
            and dino_embedding is not None
            and not torch.all(dino_embedding == 0)
        ):
            image_embedded = self.image_embedding(dino_embedding)
            image_encoded = self.image_encoder(image_embedded)
            image_features = image_encoded.squeeze(1)
        else:
            image_encoded = None

        seq_len = embedded.size(1)

        tgt_mask = self.decoder.generate_square_subsequent_mask(seq_len).to(
            embedded.device
        )

        if return_embeddings:
            decoder_output, first_layer_output = self.decoder(
                tgt=embedded,
                image_memory=image_encoded,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=padding_mask,
                return_first_layer=return_embeddings,
            )
        else:
            decoder_output = self.decoder(
                tgt=embedded,
                image_memory=image_encoded,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=padding_mask,
                return_first_layer=return_embeddings,
            )

        output = self.output_layer(decoder_output)

        if return_embeddings:
            return output, first_layer_output, image_features

        return output

    class LCGLoss(nn.Module):
        def __init__(self, d_model):
            super().__init__()

            self.d_model = d_model
            self.tau = nn.Parameter(torch.ones(1))

            self.MV = nn.Linear(d_model, d_model, bias=False)
            self.ML = nn.Linear(d_model, d_model, bias=False)

        # def forward(self, text_features, image_features, padding_mask=None):
        #     batch_size, seq_len, _ = text_features.size()

        #     text_projected = self.ML(text_features)
        #     image_projected = self.MV(image_features)

        #     valid_mask = (~padding_mask) if padding_mask is not None else torch.ones(batch_size, seq_len, dtype=torch.bool, device=text_features.device)

        #     loss = 0.0

        #     for i in range(batch_size):
        #         image_i = image_projected[i]

        #         for j in range(seq_len):
        #             if not valid_mask[i, j]:
        #                 continue

        #             token_ij = text_projected[i, j]

        #             #formula for s(i, j, i) similarity for token j in caption i and image i
        #             s_iji = torch.dot(image_i, token_ij) / self.tau

        #             first_term_denom = 0.0

        #             for k in range(batch_size):
        #                 if not valid_mask[k, j]:
        #                     continue
        #                 s_kji = torch.dot(image_projected[k], token_ij) / self.tau
        #                 first_term_denom += torch.exp(s_kji)

        #             #formula for neg(i, j) = exp(s(i,j,i)) + sum_k sum_o (1-delta_i(k))*exp(s(i,o,k))
        #             neg_term = torch.exp(s_iji)

        #             for k in range(batch_size):
        #                 if k == i:
        #                     continue
        #                 for o in range(seq_len):
        #                     if not valid_mask[k, o]:
        #                         continue

        #                     s_iok = torch.dot(image_i, text_projected[k, o]) /self.tau
        #                     neg_term += torch.exp(s_iok)

        #             first_term_loss = -torch.log(torch.exp(s_iji) / first_term_denom)
        #             second_term_loss = -torch.log(torch.exp(s_iji) / neg_term)

        #             loss += 0.5 * (first_term_loss + second_term_loss)

        #     num_valid_tokens = valid_mask.sum().item()
        #     if num_valid_tokens > 0:
        #         loss = loss/num_valid_tokens

        #     return loss

        def forward(self, text_features, image_features, padding_mask=None):
            batch_size, seq_len, _ = text_features.size()
            device = text_features.device

            # clamp temperature because of traininf instability
            tau = torch.clamp(self.tau, min=0.05, max=2.0)

            # from the paper, the ML and MV projection matrices
            text_proj = self.ML(text_features)
            image_proj = self.MV(image_features)

            # create valid tokens mask aka attention mask
            if padding_mask is None:
                valid = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
            else:
                valid = ~padding_mask

            # matching score s_iji between image i and token j in caption i, computes for all i,j
            s_iji = torch.einsum("bd,bld->bl", image_proj, text_proj) / tau

            # matching score s_kji between between image k and token j in caption i, computes for all i,j,k
            s_kji = torch.einsum("ild,kd->ilk", text_proj, image_proj) / tau

            # attention mask for valid tokens in the other captions
            mask_kj = valid.t()[None].expand(batch_size, seq_len, batch_size)
            s_kji = s_kji.masked_fill(~mask_kj, float("-inf"))
            denom_log = torch.logsumexp(s_kji, dim=2)

            # calculating neg
            s_iok = torch.einsum("id,kld->ikl", image_proj, text_proj) / tau
            mask_ko = valid[None].expand(batch_size, batch_size, seq_len)
            # mask_neq is 0 on diagonal and 1 elsewhere
            mask_neq = (
                ~torch.eye(batch_size, device=device, dtype=torch.bool)
            ).unsqueeze(-1)
            # exclude k==i and padded tokens
            mask_x = mask_ko & mask_neq
            s_iok_f = s_iok.masked_fill(~mask_x, float("-inf")).view(batch_size, -1)
            cross_log = torch.logsumexp(s_iok_f, dim=1)
            neg_log = torch.logaddexp(s_iji, cross_log.unsqueeze(1))

            # clamp -inf to zero in masked positions just in case
            denom_log = torch.where(valid, denom_log, torch.zeros_like(denom_log))
            neg_log = torch.where(valid, neg_log, torch.zeros_like(neg_log))

            # first and seconde term of eq, in log calculations for stability
            f1 = denom_log - s_iji
            f2 = neg_log - s_iji
            loss_mat = 0.5 * (f1 + f2)

            # final formula
            loss = loss_mat[valid].sum() / valid.sum()
            return loss

    class MultimodalDecoderLayer(nn.Module):
        def __init__(
            self,
            d_model: int,
            n_head: int,
            d_hid: int,
            dropout: float = 0.1,
            narrow_attention: bool = False,
        ):
            super().__init__()
            self.narrow_attention = narrow_attention
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

            # Gating
            self.gate = self.DynamicGating(d_model, dropout)

            self.ff = nn.Sequential(
                nn.Linear(d_model, d_hid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hid, d_model),
                nn.Dropout(dropout),
            )

        def forward(
            self,
            tgt,
            image_memory,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            return_first_layer=False,
        ):
            # seq_len = tgt.size(1)
            # def make_local_attention_mask(seq_len, window):
            #     # idx[i] = i
            #     idx = torch.arange(seq_len, device=device)

            #     i = idx.view(-1, 1)   # [[0],[1],…,[L-1]]
            #     j = idx.view(1, -1)   # [[0,1,…,L-1]]

            #     # mask for j is in the future (j > i) or in the past before window (i-j > window)
            #     mask = (j > i) | ((i - j) > window)

            #     return mask

            # if self.narrow_attention:
            #     narrow_mask = make_local_attention_mask(seq_len, window=3, device=tgt.device)
            #     tgt_norm = self.norm1(tgt)

            #     self_attn_output, _ = self.self_attn(tgt_norm, tgt_norm, tgt_norm, key_padding_mask=tgt_key_padding_mask, attn_mask=narrow_mask, is_causal=False)

            #     # replace with 0s in the self attention where quey is padding, otherwise you get nans and training instability
            #     if tgt_key_padding_mask is not None:
            #         pad_q = tgt_key_padding_mask.unsqueeze(-1)
            #         self_attn_output = self_attn_output.masked_fill(pad_q, 0.0)

            #     tgt = tgt + self.dropout(self_attn_output)
            #     first_layer_output = tgt.clone()
            # else:
            #     first_layer_output = None

            tgt_norm = self.norm1(tgt)

            self_attn_output, _ = self.self_attn(
                tgt_norm,
                tgt_norm,
                tgt_norm,
                key_padding_mask=tgt_key_padding_mask,
                attn_mask=tgt_mask,
                is_causal=True,
            )
            first_layer_output = tgt.clone()
            tgt = tgt + self.dropout(self_attn_output)

            if not (self.narrow_attention) and image_memory is not None:
                tgt_norm = self.norm2(tgt)
                cross_attn_output, _ = self.cross_attn_txt_image(
                    tgt_norm, image_memory, image_memory
                )
                cross_attn_output = self.dropout(cross_attn_output)

                fused = self.gate(tgt_norm, cross_attn_output)
                tgt = tgt + fused

            tgt_norm = self.norm3(tgt)
            ff_output = self.ff(tgt_norm)
            tgt = tgt + self.dropout(ff_output)

            return (tgt, first_layer_output) if return_first_layer else tgt

    class MultimodalDecoder(nn.Module):
        def __init__(
            self,
            d_model: int,
            n_head: int,
            d_hid: int,
            n_layers: int,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    self.MultimodalDecoderLayer(
                        d_model, n_head, d_hid, dropout, narrow_attention=(i == 0)
                    )
                    for i in range(n_layers)
                ]
            )

        def generate_square_subsequent_mask(self, size):
            mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
            return mask

        def forward(
            self,
            tgt,
            image_memory,
            tgt_mask,
            tgt_key_padding_mask=None,
            return_first_layer=False,
        ):
            output = tgt
            first_layer_output = None
            for i, layer in enumerate(self.layers):
                if i == 0 and return_first_layer:
                    output, first_layer_output = layer(
                        output,
                        image_memory,
                        tgt_mask,
                        tgt_key_padding_mask,
                        (return_first_layer and i == 0),
                    )
                else:
                    output = layer(
                        output,
                        image_memory,
                        tgt_mask,
                        tgt_key_padding_mask,
                        (return_first_layer and i == 0),
                    )
            if return_first_layer:
                return output, first_layer_output
            return output
