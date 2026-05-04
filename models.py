"""
models.py
=========
All classifier implementations used in the benchmark.

Model catalogue
---------------
CNN_Softmax          : two-layer MLP on top of CNN frame features.
                       Used for 'baseline' and 'temporal' modes.
ELM_Model            : Extreme Learning Machine with ridge regularisation.
                       hidden_dim=512, reg via stable ridge solve.
MLP_Model            : sklearn MLPClassifier (256 hidden units).
                       Used for 'skeleton' mode.
DeepLearningSearch   : Hyper-parameter search over SequenceCNN, SequenceLSTM,
                       TemporalConvNet on video-level sequences.
                       Incorporates early stopping (patience=5), best-val
                       checkpoint restoration, and padding-mask support.

All PyTorch models use a unified _fit_sequence / _predict_sequence interface
so training curves are always populated. Validation data is required for all
deep learning models (enables meaningful early stopping and reproducible
best-checkpoint selection).

Reproducibility notes
---------------------
- All random seeds are set at the experiment level in pipeline.py.
- ELM weights are seeded deterministically via np.random.default_rng(seed).
- Dropout is disabled during inference via model.eval().
"""

import copy
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_tensor(array, dtype, device):
    return torch.tensor(array, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Baseline / temporal: MLP classifier on aggregated CNN features
# ---------------------------------------------------------------------------

class CNN_Softmax:
    """Two-layer MLP trained end-to-end on aggregated CNN frame features.

    Architecture: Linear(D→512) → ReLU → Dropout(0.3) → Linear(512→C)

    Training uses Adam with a cosine LR schedule and restores the checkpoint
    with the best validation accuracy.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        lr: float = 1e-3,
        epochs: int = 10,
        dropout: float = 0.3,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr * 0.01
        )
        self.history_ = {
            "train_loss": [], "train_accuracy": [],
            "val_loss":   [], "val_accuracy":   [],
        }

    def fit(self, X, y, X_val=None, y_val=None):
        X_t = _to_tensor(X, torch.float32, self.device)
        y_t = _to_tensor(y, torch.long,    self.device)

        has_val = (X_val is not None) and (y_val is not None)
        if has_val:
            X_v = _to_tensor(X_val, torch.float32, self.device)
            y_v = _to_tensor(y_val, torch.long,    self.device)

        best_state  = copy.deepcopy(self.model.state_dict())
        best_val_acc = -np.inf

        for epoch in range(self.epochs):
            self.model.train()
            logits = self.model(X_t)
            loss = self.criterion(logits, y_t)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_acc = (torch.argmax(logits, dim=1) == y_t).float().mean().item()
            self.history_["train_loss"].append(float(loss.item()))
            self.history_["train_accuracy"].append(float(train_acc))

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_v)
                    val_loss   = self.criterion(val_logits, y_v).item()
                    val_acc    = (torch.argmax(val_logits, dim=1) == y_v).float().mean().item()

                self.history_["val_loss"].append(float(val_loss))
                self.history_["val_accuracy"].append(float(val_acc))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state   = copy.deepcopy(self.model.state_dict())

            print(
                f"[CNN_Softmax] Epoch {epoch + 1}/{self.epochs} | "
                f"train_loss={loss.item():.4f}  train_acc={train_acc:.4f}"
                + (f"  val_acc={val_acc:.4f}" if has_val else "")
            )

        self.model.load_state_dict(best_state)

    def predict(self, X):
        self.model.eval()
        X_t = _to_tensor(X, torch.float32, self.device)
        with torch.no_grad():
            return torch.argmax(self.model(X_t), dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# ELM
# ---------------------------------------------------------------------------

class ELM_Model:
    """Extreme Learning Machine (single hidden layer, random projection).

    The hidden weights W and bias b are drawn once from a Gaussian and frozen.
    The output weights beta are solved analytically via ridge regression:

        beta = (H^T H + reg * I)^{-1} H^T Y

    where H = ReLU(X @ W + b) and Y is the one-hot label matrix.

    Ridge regularisation (reg=1e-2) prevents overfitting on small datasets and
    yields a stable solution even when H^T H is near-singular (preferred over
    pinv which silently ignores small singular values).

    hidden_dim=512 matches the MLP baseline hidden layer size for a fair
    parameter-count comparison.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        reg: float = 1e-2,
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.reg = reg
        rng = np.random.default_rng(seed)

        # He-style initialisation: scale by 1/sqrt(input_dim)
        self.W = rng.standard_normal((input_dim, hidden_dim)).astype(np.float32) \
                 / np.sqrt(max(input_dim, 1))
        self.b = rng.standard_normal(hidden_dim).astype(np.float32) * 0.1
        self.beta = None
        self.history_ = {}

    def _hidden(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, X @ self.W + self.b)

    def fit(self, X, y, X_val=None, y_val=None):
        H = self._hidden(X)
        num_classes = int(np.max(y)) + 1
        Y = np.eye(num_classes, dtype=np.float32)[y]
        I = np.eye(H.shape[1], dtype=np.float32)
        # Ridge: beta = (H^T H + reg * I)^{-1} H^T Y
        self.beta = np.linalg.solve(H.T @ H + self.reg * I, H.T @ Y)

    def predict(self, X):
        return np.argmax(self._hidden(X) @ self.beta, axis=1)


# ---------------------------------------------------------------------------
# MLP (sklearn)
# ---------------------------------------------------------------------------

class MLP_Model:
    """Scikit-learn MLP — used for the skeleton-only mode."""

    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(256,),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self.history_ = {}

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


# ---------------------------------------------------------------------------
# Sequence model architectures
# ---------------------------------------------------------------------------

class TemporalConvNet(nn.Module):
    """Two-layer dilated TCN followed by global average pooling.

    Layer 1: kernel=3, dilation=1, padding=1  (receptive field: 3)
    Layer 2: kernel=3, dilation=2, padding=2  (receptive field: 7)
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask=None):
        # x: (B, T, D) → transpose to (B, D, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.net(x).squeeze(-1)
        return self.head(x)


class SequenceCNN(nn.Module):
    """1-D CNN with two convolutional stages and global average pooling."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        x = self.net(x).squeeze(-1)
        return self.head(x)


class SequenceLSTM(nn.Module):
    """Single-layer bidirectional LSTM.

    Using the last *real* timestep (not the final padded position)
    as the sequence representation, guided by the padding mask.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # bidirectional → 2 × hidden_dim
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, mask=None):
        output, _ = self.lstm(x)
        if mask is not None:
            # Gather the last real (non-padded) timestep for each sample
            lengths = mask.sum(dim=1).long().clamp(min=1) - 1  # (B,)
            idx = lengths.view(-1, 1, 1).expand(-1, 1, output.size(-1))
            last = output.gather(1, idx).squeeze(1)
        else:
            last = output[:, -1, :]
        return self.head(self.dropout(last))


# ---------------------------------------------------------------------------
# Deep learning hyper-parameter search
# ---------------------------------------------------------------------------

class DeepLearningSearchModel:
    """Grid search over sequence architectures and hyper-parameters.

    Trains every (architecture, config) combination independently,
    selects the configuration with the highest validation accuracy,
    and exposes a sklearn-compatible predict() interface on the winner.

    Early stopping patience is set to 5 epochs to avoid premature termination
    on small datasets where validation accuracy oscillates.
    """

    # Search grid — architectures × configurations
    SEARCH_CONFIGS = [
        ("cnn",  {"hidden_dim":  64, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 16, "epochs": 15}),
        ("cnn",  {"hidden_dim": 128, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 16, "epochs": 15}),
        ("cnn",  {"hidden_dim": 256, "dropout": 0.3, "lr": 5e-4, "weight_decay": 1e-4, "batch_size": 16, "epochs": 15}),
        ("lstm", {"hidden_dim":  64, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 16, "epochs": 15}),
        ("lstm", {"hidden_dim": 128, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 16, "epochs": 15}),
        ("lstm", {"hidden_dim": 256, "dropout": 0.3, "lr": 5e-4, "weight_decay": 1e-4, "batch_size": 16, "epochs": 15}),
        ("tcn",  {"hidden_dim":  64, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 16, "epochs": 15}),
        ("tcn",  {"hidden_dim": 128, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 16, "epochs": 15}),
        ("tcn",  {"hidden_dim": 256, "dropout": 0.3, "lr": 5e-4, "weight_decay": 1e-4, "batch_size": 16, "epochs": 15}),
    ]

    EARLY_STOPPING_PATIENCE = 5

    def __init__(self, input_dim: int, num_classes: int):
        self.input_dim   = input_dim
        self.num_classes = num_classes
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.best_model_      = None
        self.best_model_name_ = None
        self.best_config_     = None
        self.history_         = {}
        self.search_results_  = []

    def _build(self, name: str, hidden_dim: int, dropout: float) -> nn.Module:
        kwargs = dict(
            input_dim  = self.input_dim,
            hidden_dim = hidden_dim,
            num_classes= self.num_classes,
            dropout    = dropout,
        )
        if name == "cnn":  return SequenceCNN(**kwargs)
        if name == "lstm": return SequenceLSTM(**kwargs)
        if name == "tcn":  return TemporalConvNet(**kwargs)
        raise ValueError(f"Unknown sequence model: {name}")

    def _train_one(self, name, cfg, X, y, masks, X_val, y_val, masks_val):
        model = self._build(name, cfg["hidden_dim"], cfg["dropout"]).to(self.device)
        opt   = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)
        crit  = nn.CrossEntropyLoss()

        X_t   = _to_tensor(X,     torch.float32, self.device)
        y_t   = _to_tensor(y,     torch.long,    self.device)
        m_t   = _to_tensor(masks, torch.bool,    self.device) if masks is not None else None

        X_v   = _to_tensor(X_val,     torch.float32, self.device)
        y_v   = _to_tensor(y_val,     torch.long,    self.device)
        m_v   = _to_tensor(masks_val, torch.bool,    self.device) if masks_val is not None else None

        ds     = TensorDataset(X_t, y_t) if m_t is None else TensorDataset(X_t, y_t, m_t)
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

        history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
        best_state   = copy.deepcopy(model.state_dict())
        best_val_acc = -np.inf
        patience_left = self.EARLY_STOPPING_PATIENCE

        for epoch in range(cfg["epochs"]):
            model.train()
            ep_losses, ep_preds, ep_true = [], [], []

            for batch in loader:
                if m_t is not None:
                    bX, by, bm = batch
                    logits = model(bX, bm)
                else:
                    bX, by = batch
                    logits = model(bX)

                loss = crit(logits, by)
                opt.zero_grad()
                loss.backward()
                opt.step()

                ep_losses.append(loss.item())
                ep_preds.append(torch.argmax(logits, 1).detach().cpu().numpy())
                ep_true.append(by.detach().cpu().numpy())

            sched.step()

            train_loss = float(np.mean(ep_losses))
            train_acc  = float(accuracy_score(
                np.concatenate(ep_true), np.concatenate(ep_preds)
            ))

            model.eval()
            with torch.no_grad():
                val_logits = model(X_v, m_v)
                val_loss   = float(crit(val_logits, y_v).item())
                val_acc    = float(accuracy_score(
                    y_val, torch.argmax(val_logits, 1).cpu().numpy()
                ))

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            print(
                f"  [{name} h={cfg['hidden_dim']} d={cfg['dropout']}] "
                f"ep {epoch + 1:2d}/{cfg['epochs']} | "
                f"train {train_loss:.4f}/{train_acc:.4f}  "
                f"val {val_loss:.4f}/{val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc  = val_acc
                best_state    = copy.deepcopy(model.state_dict())
                patience_left = self.EARLY_STOPPING_PATIENCE
            else:
                patience_left -= 1
                if patience_left == 0:
                    print(f"  Early stopping at epoch {epoch + 1}.")
                    break

        model.load_state_dict(best_state)
        return model, history, best_val_acc

    def fit(self, X, y, X_val=None, y_val=None, masks=None, masks_val=None):
        if X_val is None or y_val is None:
            raise ValueError(
                "DeepLearningSearchModel requires validation data. "
                "Pass X_val and y_val to fit()."
            )

        best_score = -np.inf

        for name, cfg in self.SEARCH_CONFIGS:
            print(f"\n--- Search: {name} hidden={cfg['hidden_dim']} dropout={cfg['dropout']} ---")
            model, history, score = self._train_one(
                name, cfg, X, y, masks, X_val, y_val, masks_val
            )
            self.search_results_.append({
                "model_name":       name,
                "config":           cfg,
                "best_val_accuracy": score,
            })

            if score > best_score:
                best_score            = score
                self.best_model_      = model
                self.best_model_name_ = name
                self.best_config_     = cfg
                self.history_         = history

        print(
            f"\nBest configuration: {self.best_model_name_} "
            f"(val_acc={best_score:.4f})"
        )

    def predict(self, X, masks=None):
        self.best_model_.eval()
        X_t = _to_tensor(X, torch.float32, self.device)
        m_t = _to_tensor(masks, torch.bool, self.device) if masks is not None else None

        with torch.no_grad():
            logits = self.best_model_(X_t, m_t)
            return torch.argmax(logits, dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(mode: str, input_dim=None, num_classes=None, params=None):
    """Construct the appropriate model for the given pipeline mode."""
    if mode in ("baseline", "temporal"):
        return CNN_Softmax(input_dim, num_classes)

    if mode == "hybrid_svm":
        C = params.get("C", 1) if params else 1
        return SVC(kernel="linear", C=C, max_iter=5000)

    if mode == "hybrid_elm":
        return ELM_Model(input_dim)

    if mode == "skeleton":
        return MLP_Model()

    if mode == "deep_learning":
        return DeepLearningSearchModel(input_dim, num_classes)

    raise ValueError(
        f"Unknown mode: '{mode}'. Valid modes: baseline, temporal, "
        "hybrid_svm, hybrid_elm, skeleton, deep_learning."
    )
