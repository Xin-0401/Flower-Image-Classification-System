# 花朵图像分类系统

基于PyTorch和ResNet18迁移学习的102类花卉图像分类系统，使用Streamlit构建前端界面。

## 功能特点

- 基于ResNet18预训练模型，采用迁移学习策略
- 支持102种花卉类别的分类识别
- 提供完整的训练、评估、可视化流程
- Streamlit Web前端，支持图片上传和实时预测
- 展示Top-5预测结果和概率分布图

## 项目结构

```
花朵分类系统/
├── train.py              # 模型训练脚本（ResNet18迁移学习）
├── plot_history.py       # 训练曲线可视化
├── evaluate.py           # 模型评估脚本（准确率、混淆矩阵、Top-5）
├── app.py                # Streamlit前端应用
├── requirements.txt      # 项目依赖
├── data/                 # 数据集目录（Flower-102）
│   └── flowers-102/      # 花卉图片
├── best_model.pth        # 训练好的模型权重
├── train_history.json    # 训练历史记录
├── training_curves.png   # 训练曲线图
└── confusion_matrix.png  # 混淆矩阵图
```

## 环境配置

### 1. 创建Anaconda虚拟环境

```bash
conda create -n pytorch_env python=3.10 -y
conda activate pytorch_env
```

### 2. 安装PyTorch

```bash
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 安装其他依赖

```bash
pip install numpy matplotlib pillow scikit-learn seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install streamlit==1.35.0 tornado==6.3.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或直接使用：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练10轮，自动保存最佳模型为 `best_model.pth`，训练历史保存为 `train_history.json`。

### 2. 可视化训练曲线

```bash
python plot_history.py
```

生成训练/验证的损失和准确率曲线图 `training_curves.png`。

### 3. 评估模型

```bash
python evaluate.py
```

在测试集上计算准确率、精确率、召回率、F1分数、Top-5准确率，并生成混淆矩阵热力图。

### 4. 启动Web应用

```bash
streamlit run app.py
```

浏览器自动打开应用，上传花朵图片即可进行分类识别。

## 技术实现

- 深度学习框架：PyTorch
- 基础模型：ResNet18（ImageNet预训练）
- 迁移学习：冻结卷积层，只训练最后一层全连接层
- 数据集：Flower-102（102类，8189张图片）
- 前端框架：Streamlit
- 评估指标：Accuracy、Precision、Recall、F1-Score、Top-5 Accuracy
