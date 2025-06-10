import os
import time
import math
from datetime import datetime
import torch.nn.functional as F
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
import sys


class Trainer:
    def __init__(
        self,
        model,
        text_train_loader=None,
        image_caption_train_loader=None,
        text_val_loader=None,
        image_caption_val_loader=None,
        text_test_loader=None,
        image_caption_test_loader=None,
        unified=False,
        unified_train_loader=None,
        unified_val_loader=None,
        unified_test_loader=None,
        lr=1e-5,
        warmup_steps=1000,
        weight_decay=0.01,
        total_steps=100000,
        text_only_epochs=5,
        image_caption_epochs=5,
        checkpoint_dir="checkpoints",
        device="cuda",
        clip_grad_norm=None,
        eval_steps=5000,
        checkpoint_steps=5000,
        early_stopping_patience=5,
        wandb_project="dual-stream-model",
        wandb_run_name=None,
        wandb_config=None,
        lambda_contrastive_loss=1.0,
    ):
        self.model = model
        self.text_train_loader = text_train_loader
        self.image_caption_train_loader = image_caption_train_loader

        self.text_val_loader = text_val_loader
        self.image_caption_val_loader = image_caption_val_loader

        self.text_test_loader = text_test_loader
        self.image_caption_test_loader = image_caption_test_loader

        self.unified = unified
        if unified:
            self.unified_train_loader = unified_train_loader
            self.unified_val_loader = unified_val_loader
            self.unified_test_loader = unified_test_loader

        self.device = device
        self.model.to(device)

        self.text_only_epochs = text_only_epochs
        self.image_caption_epochs = image_caption_epochs
        self.max_epochs = text_only_epochs + image_caption_epochs

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        self.warmup_steps = warmup_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: (step + 1) / self.warmup_steps
            if step < self.warmup_steps
            else max(
                0.0,
                0.5
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * (step - self.warmup_steps)
                        / max(1, total_steps - self.warmup_steps)
                    )
                ),
            ),
        )

        self.scaler = GradScaler()

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.total_steps = total_steps
        self.clip_grad_norm = clip_grad_norm
        self.eval_steps = eval_steps
        self.checkpoint_steps = checkpoint_steps
        self.early_stopping_patience = early_stopping_patience

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        self._setup_wandb(wandb_project, wandb_run_name, wandb_config)

        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # contrastive loss
        self.lambda_contrastive_loss = lambda_contrastive_loss

    def _setup_wandb(self, project, run_name, config):
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if config is None:
            config = {
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
                "total_steps": self.total_steps,
                "max_epochs": self.max_epochs,
                "model": self.model.__class__.__name__,
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
            }
        wandb.init(project=project, name=run_name, config=config)

    def _process_batch(self, batch):
        input_ids = batch["text_input"].to(self.device)
        attention_mask = batch.get("text_mask", None)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        dino_embedding = batch.get("dino_embedding", None)

        if dino_embedding is not None:
            dino_embedding = dino_embedding.to(self.device)

        return input_ids, attention_mask, dino_embedding

    def _compute_loss(
        self,
        logits,
        targets,
        attention_mask=None,
        text_pooled=None,
        image_pooled=None,
        use_image=False,
    ):
        logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
        targets = targets[:, 1:].contiguous().view(-1)

        if attention_mask is not None:
            mask = attention_mask[:, 1:].contiguous().view(-1).bool()
            logits = logits[mask]
            targets = targets[mask]

        loss_ntp = self.criterion(logits, targets)

        # contrastive loss
        contrastive_loss = 0.0
        if use_image and text_pooled is not None and image_pooled is not None:
            batch_size = text_pooled.size(0)
            contrastive_loss = self.model.compute_contrastive_loss(
                text_pooled, image_pooled, batch_size
            )

        loss = loss_ntp + self.lambda_contrastive_loss * contrastive_loss
        return loss, loss_ntp, contrastive_loss

    def train_step(self, batch, use_image):
        input_ids, attention_mask, dino_embedding = self._process_batch(batch)

        self.optimizer.zero_grad()

        with autocast():
            outputs = self.model(
                input_ids=input_ids,
                padding_mask=(attention_mask == 0),
                dino_embedding=dino_embedding,
                use_image=use_image,
                return_pooled=use_image,
            )
            if use_image:
                outputs, text_pooled, image_pooled = outputs
            else:
                outputs = outputs
                text_pooled = None
                image_pooled = None

            loss, loss_ntp, contrastive_loss = self._compute_loss(
                outputs, input_ids, attention_mask, text_pooled, image_pooled, use_image
            )

        self.scaler.scale(loss).backward()

        if self.clip_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        with torch.no_grad():
            self.model.log_tau.clamp_(min=math.log(0.05), max=math.log(1.0))

        return (
            loss.item(),
            loss_ntp.item(),
            contrastive_loss.item() if use_image else None,
        )

    def train(self):
        start_time = time.time()
        if not self.unified:
            for epoch in range(self.start_epoch, self.max_epochs):
                if epoch % 2 == 0:
                    loader = self.text_train_loader
                    train_use_image = False
                else:
                    loader = self.image_caption_train_loader
                    train_use_image = True

                epoch_loss = 0.0
                epoch_loss_ntp = 0.0
                epoch_contrastive_loss = 0.0
                batch_count = 0
                contrastive_loss_batch_count = 0
                progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.max_epochs}")
                for batch in progress_bar:
                    loss, loss_ntp, contrastive_loss = self.train_step(
                        batch, use_image=train_use_image
                    )
                    epoch_loss += loss
                    epoch_loss_ntp += loss_ntp
                    if train_use_image and contrastive_loss is not None:
                        epoch_contrastive_loss += contrastive_loss
                        contrastive_loss_batch_count += 1
                    batch_count += 1
                    self.global_step += 1

                    log_dict = {
                        "train/total_loss": loss,
                        "train/next_token_loss": loss_ntp,
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "global_step": self.global_step,
                    }

                    if train_use_image and contrastive_loss is not None:
                        log_dict["train/contrastive_loss"] = contrastive_loss
                    wandb.log(log_dict, step=self.global_step)
                    progress_bar.set_postfix(
                        total_loss=loss,
                        next_token_loss=loss_ntp,
                        contrastive_loss=contrastive_loss if train_use_image else "N/A",
                        lr=self.optimizer.param_groups[0]["lr"],
                    )

                    if self.global_step % self.eval_steps == 0:
                        if not train_use_image and self.text_val_loader is not None:
                            val_loss, val_metrics = self.evaluate_loader(
                                self.text_val_loader, use_image=False, prefix="val/text"
                            )
                        elif (
                            train_use_image
                            and self.image_caption_val_loader is not None
                        ):
                            val_loss, val_metrics = self.evaluate_loader(
                                self.image_caption_val_loader,
                                use_image=True,
                                prefix="val/image_caption",
                            )
                        else:
                            val_loss = None

                        if val_loss is not None:
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                self.patience_counter = 0
                                self.save_checkpoint(epoch, is_best=True)
                            else:
                                self.patience_counter += 1

                            # if (self.early_stopping_patience > 0 and
                            #     self.patience_counter >= self.early_stopping_patience):
                            #     print(f"Early stopping triggered at step {self.global_step}")
                            #     break

                    if self.global_step % self.checkpoint_steps == 0:
                        self.save_checkpoint(epoch)
                    # if self.global_step >= self.total_steps:
                    #     break

                epoch_loss /= batch_count
                epoch_loss_ntp /= batch_count
                epoch_contrastive_loss = (
                    epoch_contrastive_loss / max(contrastive_loss_batch_count, 1)
                    if train_use_image
                    else None
                )

                log_dict = {
                    "train/epoch_total_loss": epoch_loss,
                    "train/epoch_next_token_loss": epoch_loss_ntp,
                    "epoch": epoch,
                }
                if train_use_image and epoch_contrastive_loss is not None:
                    log_dict["train/epoch_contrastive_loss"] = epoch_contrastive_loss
                wandb.log(log_dict, step=self.global_step)

                # if (self.global_step >= self.total_steps or
                #     (self.early_stopping_patience > 0 and self.patience_counter >= self.early_stopping_patience)):
                #     break
                torch.cuda.empty_cache()

            if not train_use_image and self.text_test_loader is not None:
                self.evaluate_loader(
                    self.text_test_loader, use_image=False, prefix="test/text"
                )
            elif train_use_image and self.image_caption_test_loader is not None:
                self.evaluate_loader(
                    self.image_caption_test_loader,
                    use_image=True,
                    prefix="test/image_caption",
                )
            self.save_checkpoint(epoch=self.max_epochs - 1, is_best=False)
            total_time = time.time() - start_time
            print(f"Training completed in {total_time:.2f}s")
            wandb.finish()

    def evaluate_loader(self, loader, use_image, prefix):
        self.model.eval()
        total_loss = 0.0
        total_loss_ntp = 0.0
        total_contrastive_loss = 0.0
        batch_count = 0
        contrastive_loss_batch_count = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {prefix}"):
                input_ids, attention_mask, dino_embedding = self._process_batch(batch)

                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        padding_mask=(attention_mask == 0),
                        dino_embedding=dino_embedding,
                        use_image=use_image,
                        return_pooled=use_image,
                    )
                    if use_image:
                        outputs, text_pooled, image_pooled = outputs
                    else:
                        outputs = outputs
                        text_pooled = None
                        image_pooled = None

                    loss, loss_ntp, contrastive_loss = self._compute_loss(
                        outputs,
                        input_ids,
                        attention_mask,
                        text_pooled,
                        image_pooled,
                        use_image,
                    )

                total_loss += loss.item()
                total_loss_ntp += loss_ntp.item()
                if use_image:
                    total_contrastive_loss += contrastive_loss.item()
                    contrastive_loss_batch_count += 1
                batch_count += 1

        avg_loss = total_loss / max(batch_count, 1)
        avg_loss_ntp = total_loss_ntp / max(batch_count, 1)
        avg_contrastive_loss = (
            total_contrastive_loss / max(contrastive_loss_batch_count, 1)
            if use_image
            else None
        )

        perplexity = torch.exp(torch.tensor(avg_loss_ntp)).item()
        self.model.train()

        metrics = {
            f"{prefix}/total_loss": avg_loss,
            f"{prefix}/next_token_loss": avg_loss_ntp,
            f"{prefix}/perplexity": perplexity,
        }
        if use_image and avg_contrastive_loss is not None:
            metrics[f"{prefix}/contrastive_loss"] = avg_contrastive_loss
        wandb.log(metrics, step=self.global_step)
        return avg_loss, metrics

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "model_args": {
                "vocab_size": self.model.vocab_size,
                "d_model": self.model.d_model,
                "n_head": self.model.n_head,
                "d_hid": self.model.d_hid,
                "num_encoder_layers": self.model.num_encoder_layers,
                "num_decoder_layers": self.model.num_decoder_layers,
                "dino_dim": self.model.dino_dim,
                "dropout": self.model.dropout,
            },
        }
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_{self.global_step}.pt"
        )
        torch.save(checkpoint, ckpt_path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
            torch.save(checkpoint, best_path)
        return ckpt_path

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.start_epoch = checkpoint.get("epoch", 0)
        return checkpoint.get("epoch", 0)
