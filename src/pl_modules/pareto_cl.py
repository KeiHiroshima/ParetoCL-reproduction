import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.data.continual_data import ReplayBuffer
from src.models.pareto_model import PreferenceConditionedModel


class ParetoCL(pl.LightningModule):
    """Lightning module implementing Algorithm 1 (training) and Algorithm 2 (inference)."""

    # K preference samples per training iteration (Algorithm 1)
    TRAIN_K = 5
    # K preference samples at inference time (Algorithm 2)
    INFER_K = 20

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet18",
        learning_rate: float = 0.05,
        buffer: ReplayBuffer = None,
        preference_dim: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["buffer"])

        self.model = PreferenceConditionedModel(
            num_classes,
            backbone_name,
            preference_dim=preference_dim,
        )
        self.buffer = buffer
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

        # Epoch offset: accumulated epochs from previous tasks (for continuous WandB logging)
        self.epoch_offset = 0
        # Accumulators for epoch-level metrics (reset each epoch)
        self._train_loss_sum = 0.0
        self._train_loss_incoming = 0.0
        self._train_loss_buffer = 0.0
        self._train_correct = 0.0
        self._train_total = 0

    def forward(self, x, preference):
        return self.model(x, preference)

    def expand_head(self, new_num_classes: int) -> None:
        self.model.expand_head(new_num_classes)

    def _sample_dirichlet_preference(self, K: int) -> torch.Tensor:
        """Sample K preference vectors from a 2D Dirichlet distribution.

        Dirichlet(1, 1) is equivalent to Uniform over the 2-simplex, giving
        (α₁, α₂) pairs where α₁ + α₂ = 1.  This is the prior p(α) used in
        the paper.

        Returns:
            preferences: (K, self.model.preference_dim) tensor on self.device
        """
        concentration = torch.ones(K, self.model.preference_dim, device=self.device)
        dist = torch.distributions.Dirichlet(concentration)
        return dist.sample()  # (K, self.model.preference_dim)

    # ------------------------------------------------------------------
    # Training lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        self._train_loss_sum = 0.0
        self._train_loss_incoming = 0.0
        self._train_loss_buffer = 0.0
        self._train_correct = 0.0
        self._train_total = 0

    # ------------------------------------------------------------------
    # Training step — Algorithm 1
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x_new, y_new = batch

        has_replay = self.buffer is not None and self.buffer.current_size > 0

        if has_replay:
            x_old, y_old, _ = self.buffer.sample(x_new.size(0))
            x_old = x_old.to(self.device)
            y_old = y_old.to(self.device)
        else:
            x_old = torch.empty(0, device=self.device)
            y_old = torch.empty(0, device=self.device)

        # Sample K preference vectors from Dirichlet(1,1), one per inner iteration
        preferences = self._sample_dirichlet_preference(self.TRAIN_K)

        total_loss = torch.tensor(0.0, device=self.device)
        loss_incoming = torch.tensor(0.0, device=self.device)
        loss_buffer = torch.tensor(0.0, device=self.device)
        total_correct = torch.tensor(0.0, device=self.device)

        for k in range(self.TRAIN_K):
            alpha = preferences[k].unsqueeze(0)  # (1, 2)
            alpha_stab = preferences[k, 0]  # α₁ (stability)
            alpha_plast = preferences[k, 1]  # α₂ (plasticity)

            # Plasticity loss — new task data
            logits_new = self(x_new, alpha)
            loss_new_k = self.criterion(logits_new, y_new)
            total_correct += (logits_new.argmax(dim=1) == y_new).float().sum()

            # Stability loss — replay buffer
            if has_replay:
                logits_old = self(x_old, alpha)
                loss_old_k = self.criterion(logits_old, y_old)
                total_correct += (logits_old.argmax(dim=1) == y_old).float().sum()
                loss_buffer_k = alpha_stab * loss_old_k
            else:
                loss_buffer_k = torch.tensor(0.0, device=self.device)

            loss_incoming += alpha_plast * loss_new_k
            loss_buffer += loss_buffer_k
            total_loss += alpha_plast * loss_new_k + loss_buffer_k

        # Average over K samples (Eq. 4: expectation over p(α))
        loss = total_loss / self.TRAIN_K

        # Accumulate for epoch-level logging (logged in on_train_epoch_end)
        self._train_loss_sum += loss.item()
        self._train_loss_incoming += (loss_incoming / self.TRAIN_K).item()
        self._train_loss_buffer += (loss_buffer / self.TRAIN_K).item()
        self._train_correct += total_correct.item()
        self._train_total += self.TRAIN_K * (x_new.size(0) + x_old.size(0))

        return loss

    # ------------------------------------------------------------------
    # Epoch-end logging
    # ------------------------------------------------------------------

    def on_train_epoch_end(self):
        if self._train_total == 0 or self.logger is None:
            return

        num_batches = self.trainer.num_training_batches

        self.log(
            "train_loss_epoch",
            self._train_loss_sum / num_batches,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_loss_incoming_epoch",
            self._train_loss_incoming / num_batches,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_loss_buffer_epoch",
            self._train_loss_buffer / num_batches,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_acc_epoch",
            self._train_correct / self._train_total,
            on_step=False,
            on_epoch=True,
        )

    # ------------------------------------------------------------------
    # Inference — Algorithm 2
    # ------------------------------------------------------------------

    def _infer_with_preference_adaptation(self, x: torch.Tensor) -> torch.Tensor:
        """Algorithm 2: inference-time preference adaptation.

        Samples INFER_K preference vectors, runs a forward pass for each, then
        selects — per sample — the logits from the preference vector that yields
        the lowest output entropy (highest confidence).

        Args:
            x: (B, C, H, W) input images

        Returns:
            logits: (B, num_classes) best logits for each sample
        """
        B = x.size(0)
        preferences = self._sample_dirichlet_preference(self.INFER_K)  # (K, 2)

        all_logits = []
        for k in range(self.INFER_K):
            alpha = preferences[k].unsqueeze(0)  # (1, 2)
            logits_k = self(x, alpha)  # (B, num_classes)
            all_logits.append(logits_k)

        all_logits = torch.stack(all_logits, dim=0)  # (K, B, num_classes)

        # Compute entropy for each (k, b) pair: H = -sum(p * log(p))
        probs = torch.softmax(all_logits, dim=-1)  # (K, B, num_classes)
        entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)  # (K, B)

        # For each sample select the k with minimum entropy (Eq. 6)
        best_k = entropy.argmin(dim=0)  # (B,)

        best_k_expanded = best_k.view(1, B, 1).expand(1, B, all_logits.size(-1))
        best_logits = all_logits.gather(0, best_k_expanded).squeeze(0)  # (B, num_classes)

        return best_logits

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self._infer_with_preference_adaptation(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # dataloader_idx corresponds to the task index when multiple dataloaders are returned
        self.log(f"val_loss_task_{dataloader_idx}", loss, add_dataloader_idx=False)
        self.log(f"val_acc_task_{dataloader_idx}", acc, add_dataloader_idx=False)

        return loss

    def configure_optimizers(self):
        # NOTE: the original codebase's README documents momentum=0.9, but the
        # actual implementation that produced the logged results used plain
        # SGD (no momentum). Kept faithful to the latter for reproducibility.
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
