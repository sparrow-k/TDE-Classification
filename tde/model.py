"""Unidirectional GRU that outputs a prediction after every observation (RAPID-style)."""
import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """GRU encoder + per-step head.

    n_outputs=1  -> binary TDE logit per step (MALLORN).
    n_outputs=C  -> C class logits per step (PLAsTiCC pre-training); `tde_class` is the TDE column.
    """

    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.1, n_outputs=1, tde_class=None):
        super().__init__()
        # Temporal encoder. Unidirectional on purpose: the output at step k only sees steps <= k.
        self.encoder = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        # Classification head applied independently at every time step.
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_outputs),
        )
        self.n_outputs = n_outputs
        self.tde_class = tde_class
        self.encoder_frozen = False

    def forward(self, x):
        """x: (batch, time, n_features) -> logits (batch, time) or (batch, time, n_outputs)."""
        hidden, _ = self.encoder(x)
        out = self.head(hidden)
        return out.squeeze(-1) if self.n_outputs == 1 else out

    def tde_probability(self, logits):
        """Per-step P(TDE) from forward() output: sigmoid (binary head) or softmax column (multi-class head)."""
        if self.n_outputs == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)[..., self.tde_class]

    def freeze_encoder(self):
        """Fix the encoder weights (proposal: fine-tune only the classification layers)."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder_frozen = True
        self.encoder.eval()

    def train(self, mode=True):
        # A frozen encoder is a fixed feature extractor, so its dropout stays off during training too.
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self


def load_encoder_weights(model, state_dict):
    """Copy only the encoder weights from a (pre-trained) model state_dict into `model`."""
    encoder_state = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(encoder_state)
