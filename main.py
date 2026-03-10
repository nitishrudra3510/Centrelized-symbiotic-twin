"""
Centralized Training Script
--------------------------------
- Loads processed CSV
- Trains simple MLP
- Uses energy estimation (same logic as federated)
- Saves metrics.csv for dashboard
"""

import time
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ======================================================
# 🔋 ENERGY MODEL (same logic as federated)
# ======================================================

BASELINE_POWER_W = 2.5  # Simulated IoT baseline power


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_energy(computation_time_s, model=None, power_w=BASELINE_POWER_W):
    if model is not None:
        param_count = count_parameters(model)
        scale_factor = 1.0 + (param_count / 100_000) * 0.5
        power_w = power_w * scale_factor

    energy_j = power_w * computation_time_s
    return round(energy_j, 6)


class EnergyMonitor:
    def __init__(self, model=None, power_w=BASELINE_POWER_W):
        self.model = model
        self.power_w = power_w
        self.energy_j = 0.0
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        self.energy_j = estimate_energy(elapsed, self.model, self.power_w)


# ======================================================
# 📄 LOAD CONFIG
# ======================================================

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

input_size = config["model"]["input_size"]
hidden_size = config["model"]["hidden_size"]
num_classes = config["model"]["num_classes"]
dropout = config["model"]["dropout"]
test_split = config["data"]["test_split"]


# ======================================================
# 📊 LOAD PROCESSED DATA
# ======================================================

df = pd.read_csv("data/processed/processed.csv")

FEATURE_COLS = ["co", "humidity", "light", "lpg", "motion", "smoke", "temp"]

X = df[FEATURE_COLS].values.astype("float32")
y = df["label"].values.astype("int64")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_split, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)


# ======================================================
# 🧠 MODEL
# ======================================================

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)


model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ======================================================
# 🚀 TRAINING LOOP
# ======================================================

epochs = 30
metrics = []

print("Starting Centralized Training...\n")

for epoch in range(1, epochs + 1):

    with EnergyMonitor(model=model) as monitor:

        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    energy_j = monitor.energy_j
    latency_ms = (energy_j / BASELINE_POWER_W) * 1000  # approximate ms

    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        acc = accuracy_score(y_test, preds.numpy())

    metrics.append([epoch, acc, latency_ms, energy_j])

    print(
        f"Epoch {epoch:02d} | "
        f"Accuracy: {acc:.4f} | "
        f"Latency: {latency_ms:.2f} ms | "
        f"Energy: {energy_j:.6f} J"
    )


# ======================================================
# 💾 SAVE METRICS
# ======================================================

metrics_df = pd.DataFrame(
    metrics, columns=["round", "accuracy", "latency", "energy"]
)

metrics_df.to_csv("metrics.csv", index=False)

print("\nTraining completed.")
print("Metrics saved to metrics.csv")