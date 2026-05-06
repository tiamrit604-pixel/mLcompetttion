"""
models.py
=========
Deep learning model definitions.  Three architectures used in the
notebook:

    BPNN  -- 5-layer FC, operates on PCA-reduced flat features
    CNN   -- 3-channel 2-D ConvNet, operates on (3, 32, 64) MFCC images
    BiLSTM-- 2-layer bidirectional LSTM on (64, 32) MFCC time-series

These class definitions must match the notebook bit-for-bit so saved
state_dicts load without architecture mismatches.
"""
import torch
import torch.nn as nn

from .constants import (
    NUM_CLASSES, FEATURE_DIM, CNN_CHANNELS, N_MFCC,
    FIXED_FRAMES, SEQ_LEN, INPUT_SIZE,
)


class BPNN(nn.Module):
    """
    Backpropagation Neural Network -- 5 fully-connected layers.

    Operates on PCA-reduced flat features (typically 83-d after PCA on the
    221-d raw feature vector).  BatchNorm + dropout at every layer.
    """
    def __init__(self, input_dim=FEATURE_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,  64),       nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):
    """
    3-channel 2-D CNN for the MFCC image representation.

    Input  : (B, 3, 32, 64)  -- MFCC | delta-MFCC | log-Mel
    Output : (B, num_classes)
    """
    def __init__(self, in_channels=CNN_CHANNELS, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),          nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),          nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM on a genuine MFCC time-series.

    Input  : (B, 64, 32)  -- 64 time frames of 32 MFCC coefficients
    Output : (B, num_classes)

    LayerNorm on the final time step prevents the long-range gradient
    decay that otherwise cripples LSTMs on ~64-step sequences.
    """
    def __init__(self, input_size=INPUT_SIZE, hidden_size=32,
                 num_layers=2, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        out, _    = self.lstm(x)
        last_step = out[:, -1, :]
        return self.classifier(self.layer_norm(last_step))


# =============================================================================
# Training & evaluation utilities (Section 13 of the notebook)
# =============================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    """One training pass with gradient clipping (max_norm=1.0)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += X_batch.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """No-grad evaluation. Returns (acc, loss, preds, true)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_true = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        preds  = logits.argmax(1)
        correct    += (preds == y_batch).sum().item()
        total      += X_batch.size(0)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_true.extend(y_batch.cpu().numpy().tolist())
    return correct / total, total_loss / total, all_preds, all_true


def train_model(model, tr_loader, ts_loader, optimizer, criterion,
                n_epochs, patience, device, scheduler=None,
                progress_callback=None, model_name="model"):
    """
    Full training loop with early stopping on test accuracy.

    progress_callback(epoch, n_epochs, train_acc, test_acc) -- optional,
    used by the Streamlit UI to update the progress bar.
    """
    best_acc      = 0.0
    best_state    = None
    epochs_no_imp = 0
    history       = {'tr_loss': [], 'ts_loss': [],
                     'tr_acc' : [], 'ts_acc' : []}

    for epoch in range(n_epochs):
        tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, criterion, device)
        ts_acc, ts_loss, _, _ = evaluate(model, ts_loader, criterion, device)

        history['tr_loss'].append(tr_loss)
        history['ts_loss'].append(ts_loss)
        history['tr_acc'].append(tr_acc)
        history['ts_acc'].append(ts_acc)

        if scheduler is not None:
            scheduler.step()

        if progress_callback is not None:
            progress_callback(epoch + 1, n_epochs, tr_acc, ts_acc)

        if ts_acc > best_acc:
            best_acc      = ts_acc
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
