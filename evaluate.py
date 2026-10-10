# 模型评估脚本
import os
os.environ['TORCH_HOME'] = './torch_cache'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from flower_dataset import FlowerDataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载测试集
print("加载测试集...")
test_dataset = FlowerDataset(
    root="./data/flowers-102",
    split="test",
    transform=test_transform
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
print(f"测试集: {len(test_dataset)} 张")

# 加载模型
print("\n加载模型...")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 102)

try:
    model.load_state_dict(torch.load("./best_model.pth", map_location=device))
    print("成功加载 best_model.pth")
except:
    print("未找到训练好的模型，使用预训练模型（仅作演示）")
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(num_ftrs, 102)

model = model.to(device)
model.eval()

# 在测试集上预测
print("\n在测试集上预测...")
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# 计算指标
accuracy = np.mean(all_preds == all_labels)
print(f"\n测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 分类报告
print("\n分类报告（前10类）：")
report = classification_report(
    all_labels, all_preds,
    digits=4,
    output_dict=True,
    zero_division=0
)

print(f"{'类别':<6} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
print("-" * 40)
for i in range(10):
    class_key = str(i)
    if class_key in report:
        r = report[class_key]
        print(f"{i:<6} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1-score']:<10.4f}")

print(f"\n整体指标：")
print(f"  宏平均精确率: {report['macro avg']['precision']:.4f}")
print(f"  宏平均召回率: {report['macro avg']['recall']:.4f}")
print(f"  宏平均F1分数: {report['macro avg']['f1-score']:.4f}")
print(f"  加权平均F1分数: {report['weighted avg']['f1-score']:.4f}")

# Top-5准确率
print("\n计算Top-5准确率...")
all_top5_preds = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, top5_preds = torch.topk(outputs, 5, dim=1)
        all_top5_preds.extend(top5_preds.cpu().numpy())

top5_correct = sum(1 for i, label in enumerate(all_labels) if label in all_top5_preds[i])
top5_accuracy = top5_correct / len(all_labels)
print(f"Top-5准确率: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")

# 绘制混淆矩阵（前20类，避免太密集）
print("\n绘制混淆矩阵（前20类）...")
cm = confusion_matrix(all_labels, all_preds)
cm_subset = cm[:20, :20]

plt.figure(figsize=(12, 10))
sns.heatmap(cm_subset, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(20), yticklabels=range(20))
plt.title("混淆矩阵（前20类）")
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.tight_layout()
plt.savefig("./confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

print("混淆矩阵已保存为 confusion_matrix.png")
print("\n评估完成！")
