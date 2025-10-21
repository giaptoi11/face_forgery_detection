import pandas as pd
import matplotlib.pyplot as plt

# Đọc file CSV
df = pd.read_csv("/home/toi/research/face_forgery_detection/baseline/MoE-FFD/models/train/log.csv")

# --- Biểu đồ 1: Train Loss ---
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_loss"], marker="o")
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.grid(True)
plt.show()
plt.savefig("TrainLoss.png")

# --- Biểu đồ 2: Train Accuracy ---
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_acc"], marker="o", label="Train Acc")
plt.plot(df["epoch"], df["V_Acc"], marker="s", label="Val Acc")
# plt.plot(df["epoch"], df["F_Acc"], marker="^", label="Fake Acc")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("Accuracy.png")

# --- Biểu đồ 3: AUC ---
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["V_Auc"], marker="o", label="V_Auc")
plt.plot(df["epoch"], df["F_Auc"], marker="s", label="F_Auc")
plt.title("AUC per Epoch")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("AUC.png")

# --- Biểu đồ 4: EER ---
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["V_EER"], marker="o", label="V_EER")
plt.plot(df["epoch"], df["F_EER"], marker="s", label="F_EER")
plt.title("Equal Error Rate per Epoch")
plt.xlabel("Epoch")
plt.ylabel("EER (%)")
plt.legend()
plt.grid(True)
plt.show()


plt.savefig("EER.png")