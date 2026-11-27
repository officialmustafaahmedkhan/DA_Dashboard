# ============================================================
# PHASE 12: DEEP LEARNING MODELS (NEURAL NETWORK & TRANSFORMER)
# Dataset: sindh-education-stats_2.csv (Uses P2, P4 & P5 Outputs)
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("=" * 65)
print("PHASE 12: DEEP LEARNING MODELING (PYTORCH)")
print("=" * 65)

OUTPUT_DIR = Path("Phase12_DL_Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ============================================================
# PART A: DEEP NEURAL NETWORK (MLP) ON PCA FEATURES (P5 OUTPUT)
# ============================================================

print("\n--- PART A: DEEP MLP NEURAL NETWORK (ON PCA FEATURES) ---")

X_train_pca = np.load("P5_X_train_pca_95.npy")
X_test_pca = np.load("P5_X_test_pca_95.npy")
y_train = np.load("P4_y_train.npy")
y_test = np.load("P4_y_test.npy")

train_ds = TensorDataset(torch.tensor(X_train_pca, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_ds = TensorDataset(torch.tensor(X_test_pca, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

class DeepMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

mlp_model = DeepMLPClassifier(input_dim=X_train_pca.shape[1], num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.002)

epochs = 40
train_losses, val_accuracies = [], []

for epoch in range(epochs):
    mlp_model.train()
    running_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = mlp_model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * bx.size(0)

    train_losses.append(running_loss / len(train_ds))

    mlp_model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            out = mlp_model(bx)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            actuals.extend(by.numpy())

    val_acc = accuracy_score(actuals, preds)
    val_accuracies.append(val_acc)

print(f"Final Deep MLP Test Accuracy (on PCA Features): {val_accuracies[-1] * 100:.2f}%")
print("\nDeep MLP Classification Report:")
print(classification_report(actuals, preds, target_names=["ENROLMENT", "INSTITUTIONS", "TEACHING STAFF"]))

# Plot Training Curve
plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, color="crimson")
plt.title("Deep MLP Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")

plt.subplot(1, 2, 2)
plt.plot([a * 100 for a in val_accuracies], color="navy")
plt.title("Deep MLP Validation Accuracy (%)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "P12_deep_mlp_training_curve.png", dpi=150)
plt.show()

# ============================================================
# PART B: TABULAR TRANSFORMER WITH ENTITY EMBEDDINGS
# ============================================================

print("\n--- PART B: TABULAR TRANSFORMER (ENTITY EMBEDDINGS + ATTENTION) ---")

df = pd.read_csv("P2_cleaned_data.csv")
df["Log_Number"] = np.log1p(df["Number"])

cat_cols = ["District", "Location", "School Type", "Composition"]
num_cols = ["Number", "Population", "No/100000", "Log_Number"]
target_col = "Category"

encoders = {}
for col in cat_cols + [target_col]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

X_cat = df[cat_cols].values
X_num = df[num_cols].values
y_all = df[target_col].values

Xc_tr, Xc_te, Xn_tr, Xn_te, y_tr, y_te = train_test_split(
    X_cat, X_num, y_all, test_size=0.20, random_state=42, stratify=y_all
)

tr_tab_ds = TensorDataset(
    torch.tensor(Xc_tr, dtype=torch.long),
    torch.tensor(Xn_tr, dtype=torch.float32),
    torch.tensor(y_tr, dtype=torch.long)
)
te_tab_ds = TensorDataset(
    torch.tensor(Xc_te, dtype=torch.long),
    torch.tensor(Xn_te, dtype=torch.float32),
    torch.tensor(y_te, dtype=torch.long)
)

tr_tab_loader = DataLoader(tr_tab_ds, batch_size=64, shuffle=True)
te_tab_loader = DataLoader(te_tab_ds, batch_size=64, shuffle=False)

class TabularTransformerClassifier(nn.Module):
    def __init__(self, cardinalities, emb_dim, num_features, d_model, n_heads, n_layers, num_classes):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cardinalities])
        in_dim = len(cardinalities) * emb_dim + num_features
        self.proj = nn.Linear(in_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128, dropout=0.15, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x_cat, x_num):
        emb = torch.cat([layer(x_cat[:, i]) for i, layer in enumerate(self.embeddings)], dim=1)
        x = torch.cat([emb, x_num], dim=1)
        x = self.proj(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.classifier(x)

cardinalities = [len(encoders[c].classes_) for c in cat_cols]
transformer_model = TabularTransformerClassifier(
    cardinalities=cardinalities,
    emb_dim=16,
    num_features=len(num_cols),
    d_model=64,
    n_heads=4,
    n_layers=2,
    num_classes=3
).to(device)

optimizer_tf = optim.Adam(transformer_model.parameters(), lr=0.001)

for epoch in range(25):
    transformer_model.train()
    for bc, bn, by in tr_tab_loader:
        bc, bn, by = bc.to(device), bn.to(device), by.to(device)
        optimizer_tf.zero_grad()
        out = transformer_model(bc, bn)
        loss = criterion(out, by)
        loss.backward()
        optimizer_tf.step()

transformer_model.eval()
tf_preds, tf_actuals = [], []
with torch.no_grad():
    for bc, bn, by in te_tab_loader:
        bc, bn = bc.to(device), bn.to(device)
        out = transformer_model(bc, bn)
        tf_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
        tf_actuals.extend(by.numpy())

tf_acc = accuracy_score(tf_actuals, tf_preds)
print(f"Tabular Transformer Test Accuracy (with District Embeddings): {tf_acc * 100:.2f}%")

# Save trained PyTorch weights
torch.save(mlp_model.state_dict(), OUTPUT_DIR / "P12_deep_mlp_weights.pth")
torch.save(transformer_model.state_dict(), OUTPUT_DIR / "P12_tabular_transformer_weights.pth")

print("\nPHASE 12 COMPLETED SUCCESSFULLY! Models saved in:", OUTPUT_DIR.resolve())