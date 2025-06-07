import torch
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional, Dict, Any
from lm_eval.api.model import TemplateLM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, clear_torch_cache
from lm_eval import utils
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
import os
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)

from models.model_film_on_image import DualStreamTransformer
from datasets import load_dataset


@register_model("dualstream_film_on_image")
class DualStreamLM(TemplateLM):
    _DEFAULT_MAX_LENGTH = 128

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = None,
        max_length: int = 128,
        image_src: Optional[str] = None,
        image_src_split: Optional[str] = None,
        image_key: str = "image",
    ):
        super().__init__()

        self._device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.tokenizer.add_special_tokens(
            {"pad_token": "[PAD]", "eos_token": "[EOS]", "bos_token": "[BOS]"}
        )
        self.tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=self.tokenizer.bos_token + " $A " + self.tokenizer.eos_token,
            special_tokens=[
                (self.tokenizer.eos_token, self.tokenizer.eos_token_id),
                (self.tokenizer.bos_token, self.tokenizer.bos_token_id),
            ],
        )

        vocab_size = len(self.tokenizer)
        self.vocab_size = vocab_size

        checkpoint = torch.load(model_path, map_location=self._device)

        model_args = checkpoint.get("model_args", {})
        if not model_args:
            raise ValueError("Checkpoint does not contain model_args")
        self._model = DualStreamTransformer(**model_args)
        print("Model args are ", model_args)

        print("Global step is ", checkpoint["global_step"])
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model = self._model.to(device=device)
        self._model.eval()

        self.dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        self.dino_processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
        self.dino_model.eval()

        self._max_length = max_length
        self.truncation_max_length = max_length
        self.batch_size_per_gpu = (
            int(batch_size) if isinstance(batch_size, str) else batch_size
        )
        self.image_key = image_key
        self._rank = 0
        self._world_size = 1
        if image_src:
            if image_src_split:
                self.image_src_split = image_src_split
            else:
                self.image_src_split = "test"
            self.image_src = load_dataset(image_src)[self.image_src_split]
        else:
            self.image_src = None

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = self.tokenizer.pad_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 128

    @property
    def prefix_token_id(self):
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        if self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        return self.tokenizer.pad_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(
        self,
        string: str,
        left_truncate_len=None,
        add_special_tokens=False,  # set to false because the ctx has BOS
    ) -> List[int]:
        special_tokens_kwargs = {}
        if add_special_tokens is None:
            special_tokens_kwargs = {"add_special_tokens": True}
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(
            string,
            truncation=True,
            max_length=self.truncation_max_length,
            **special_tokens_kwargs,
        )

        if left_truncate_len is not None:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_decode(self, tokens, skip_special_tokens=False):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(
        self,
        inputs: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dino_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        padding_mask = attn_mask.eq(0) if attn_mask is not None else None
        use_image = dino_embedding is not None
        # print("input_ids", self.tok_decode(inputs[0]))
        with torch.no_grad():
            logits = self._model(
                input_ids=inputs,
                dino_embedding=dino_embedding,
                padding_mask=padding_mask,
                use_image=use_image,
            )

        return logits

    def _select_cont_toks(
        self, logits: torch.Tensor, contlen: int, inplen: int
    ) -> torch.Tensor:
        logits = logits[inplen - contlen : inplen]
        return logits

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Optional[str], List[int], List[int], Optional[Any]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            return list(req[-len(req) + 1]) + req[-len(req) + 2][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_fn=_lookup_one_token_cont,
        )

        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )

        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []
            dino_embeddings = []

            padding_len_inp = None

            for item in chunk:
                if len(item) == 3:
                    _, context_enc, continuation_enc = item
                    image_id = None
                elif len(item) == 5:
                    _, context_enc, continuation_enc, image_id, image_key = item
                else:
                    raise ValueError(f"Invalid request format: {item}")

                if len(context_enc) == 0:
                    context_enc = [self.prefix_token_id]

                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self._device,
                )
                (inplen,) = inp.shape

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                # Handle image
                dino_embedding = None
                if image_id is not None and self.image_src is not None:
                    try:
                        image = self.image_src[image_id][image_key]
                        dino_embedding = self._load_dino_embedding(image)
                    except Exception as e:
                        print(f"Error loading image {image_id}: {e}")
                        dino_embedding = None

                inps.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)
                dino_embeddings.append(dino_embedding)

            batched_inps = pad_and_concat(padding_len_inp, inps, padding_side="right")
            batched_attn_mask = pad_and_concat(
                padding_len_inp,
                [torch.ones_like(inp) for inp in inps],
                padding_side="right",
            )

            dino_emb = None
            if any(de is not None for de in dino_embeddings):
                dino_emb = torch.stack([de for de in dino_embeddings])

            multi_logits = F.log_softmax(
                self._model_call(
                    batched_inps, attn_mask=batched_attn_mask, dino_embedding=dino_emb
                ),
                dim=-1,
            )

            for item, logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                if len(item) == 3:
                    request_str, ctx_tokens, _ = item
                    image_id = None
                elif len(item) == 5:
                    request_str, ctx_tokens, _, image_id, image_key = item
                contlen = len(cont_toks)

                ctx_len = inplen + (logits.shape[0] - padding_len_inp)
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)

                greedy_tokens = logits.argmax(dim=-1)

                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(0)
                    max_equal = (greedy_tokens == cont_toks).all()

                    # print(self.tok_decode(greedy_tokens[0]))

                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )

                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def _load_dino_embedding(self, image):
        if isinstance(image, str):
            image_path = image
            if self.image_src and not os.path.isabs(image_path):
                if self.image_src_split:
                    image_path = os.path.join(
                        self.image_src, self.image_src_split, image_path
                    )
                else:
                    image_path = os.path.join(self.image_src, image_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found at {image_path}")
                img = Image.open(image_path).convert("RGB")
        else:
            img = image.convert("RGB")

        inputs = self.dino_processor(images=img, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            cls_token = outputs.last_hidden_state[:, 0, :]

        return cls_token.squeeze(0)  # [768]

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        pass

    def _model_generate(
        self,
        context,
        max_length,
        stop,
        dino_embedding: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ):
        pass

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        pass
