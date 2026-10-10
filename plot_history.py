# 训练结果可视化
import matplotlib.pyplot as plt
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载训练历史
with open("./train_history.json", "r") as f:
    history = json.load(f)

epochs = range(1, len(history["train_loss"]) + 1)

# 绘制损失曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, history["train_loss"], "b-", label="训练损失")
plt.plot(epochs, history["val_loss"], "r-", label="验证损失")
plt.title("训练和验证损失曲线")
plt.xlabel("轮次 (Epoch)")
plt.ylabel("损失 (Loss)")
plt.legend()
plt.grid(True, alpha=0.3)

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, history["train_acc"], "b-", label="训练准确率")
plt.plot(epochs, history["val_acc"], "r-", label="验证准确率")
plt.title("训练和验证准确率曲线")
plt.xlabel("轮次 (Epoch)")
plt.ylabel("准确率 (Accuracy)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./training_curves.png", dpi=300, bbox_inches="tight")
plt.show()

print("训练曲线已保存为 training_curves.png")
