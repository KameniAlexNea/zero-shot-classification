"""Custom Trainer with discriminative learning rates.

The backbone (pretrained transformer) uses a lower learning rate while the
task-specific head (aggregator, scoring, label enrichment) uses a higher one.
This follows ULMFiT (Howard & Ruder, 2018) and GLiClass's practice of
separating encoder vs classifier learning rates.
"""

from transformers import Trainer

HEAD_LR_MULTIPLIER = 10.0
"""Head parameters get `lr * HEAD_LR_MULTIPLIER`."""


class GliZNetTrainer(Trainer):
    """Trainer with discriminative learning rates for backbone vs head."""

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        lr = self.args.learning_rate
        wd = self.args.weight_decay

        # Split parameters: backbone vs everything else (aggregator, loss_fn)
        decay_backbone = []
        no_decay_backbone = []
        decay_head = []
        no_decay_head = []

        no_decay_names = {
            "bias",
            "LayerNorm.weight",
            "layernorm.weight",
            "layer_norm.weight",
        }

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            is_backbone = name.startswith("backbone.")
            is_no_decay = any(nd in name for nd in no_decay_names)

            if is_backbone:
                if is_no_decay:
                    no_decay_backbone.append(param)
                else:
                    decay_backbone.append(param)
            else:
                if is_no_decay:
                    no_decay_head.append(param)
                else:
                    decay_head.append(param)

        head_lr = lr * HEAD_LR_MULTIPLIER

        optimizer_grouped_parameters = [
            {"params": decay_backbone, "lr": lr, "weight_decay": wd},
            {"params": no_decay_backbone, "lr": lr, "weight_decay": 0.0},
            {"params": decay_head, "lr": head_lr, "weight_decay": wd},
            {"params": no_decay_head, "lr": head_lr, "weight_decay": 0.0},
        ]

        from torch.optim import AdamW

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer
