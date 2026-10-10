# 花朵分类模型训练脚本
# 基于ResNet18迁移学习，Flower-102数据集（102类花卉）

import os
os.environ['TORCH_HOME'] = './torch_cache'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from flower_dataset import FlowerDataset
import time
import copy
import json
import os

# 设置随机种子，保证结果可复现
torch.manual_seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 1. 数据预处理 ====================
print("\n1. 数据预处理...")

# 训练集数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 验证集预处理（不做增强）
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==================== 2. 加载数据集 ====================
print("\n2. 加载数据集...")

try:
    data_root = "./data/flowers-102"
    train_dataset = FlowerDataset(
        root=data_root,
        split="train",
        transform=train_transform
    )

    val_dataset = FlowerDataset(
        root=data_root,
        split="val",
        transform=val_transform
    )

    print(f"训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")
    print(f"类别数: 102")

except Exception as e:
    print(f"数据集加载失败: {e}")
    print("请先运行 python download_data.py 下载数据集。")
    exit()

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"批次大小: {batch_size}")
print(f"训练批次数: {len(train_loader)}")

# ==================== 3. 构建模型 ====================
print("\n3. 构建模型...")

# 加载预训练的ResNet18
model = models.resnet18(weights='IMAGENET1K_V1')

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 替换最后一层全连接层
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 102)

model = model.to(device)

print(f"模型: ResNet18（迁移学习，只训练最后一层）")
print(f"输入特征数: {num_ftrs}")
print(f"输出类别数: 102")

# 统计参数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

# ==================== 4. 定义损失函数和优化器 ====================
print("\n4. 定义损失函数和优化器...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

print("损失函数: CrossEntropyLoss")
print("优化器: Adam")
print("初始学习率: 0.001")

# ==================== 5. 训练模型 ====================
print("\n5. 开始训练...")

num_epochs = 10
since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

# 记录训练历史
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

print(f"训练轮数: {num_epochs}")
print("开始训练...（每轮约2-5分钟）\n")

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-" * 30)

    # 训练阶段
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc.item())

    print(f"训练 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # 验证阶段
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_dataset)
    epoch_acc = running_corrects.double() / len(val_dataset)

    val_loss_history.append(epoch_loss)
    val_acc_history.append(epoch_acc.item())

    print(f"验证 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # 保存最好的模型
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        print("  ★ 新的最佳模型！")

    scheduler.step()
    print()

time_elapsed = time.time() - since
print(f"训练完成！")
print(f"用时: {time_elapsed//60:.0f}分 {time_elapsed%60:.0f}秒")
print(f"最佳验证准确率: {best_acc:.4f}")

# 加载最佳模型
model.load_state_dict(best_model_wts)

# 保存模型
torch.save(model.state_dict(), "./best_model.pth")
print("\n模型已保存为 best_model.pth")

# 保存训练历史
history = {
    "train_loss": train_loss_history,
    "train_acc": train_acc_history,
    "val_loss": val_loss_history,
    "val_acc": val_acc_history
}
with open("./train_history.json", "w") as f:
    json.dump(history, f)
print("训练历史已保存为 train_history.json")
