# 花朵分类系统 - Streamlit前端
import os
os.environ['TORCH_HOME'] = './torch_cache'
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Oxford Flower-102 标准类别名称（按类别编号1-102排列，索引0=类别1）
FLOWER_NAMES = [
    "pink primrose (粉月见草)",          # 1
    "hard-leaved pocket orchid (硬叶兜兰)",  # 2
    "canterbury bells (风铃草)",         # 3
    "sweet pea (香豌豆)",                # 4
    "english marigold (金盏花)",         # 5
    "tiger lily (卷丹百合)",              # 6
    "moon orchid (蝴蝶兰)",              # 7
    "bird of paradise (天堂鸟)",         # 8
    "monkshood (乌头)",                  # 9
    "globe thistle (球蓟)",              # 10
    "snapdragon (金鱼草)",               # 11
    "colt's foot (款冬)",                # 12
    "king protea (帝王花)",              # 13
    "spear thistle (蓟)",                # 14
    "yellow iris (黄鸢尾)",              # 15
    "globe-flower (金莲花)",             # 16
    "purple coneflower (紫锥菊)",         # 17
    "peruvian lily (六出花)",            # 18
    "balloon flower (桔梗)",             # 19
    "giant white arum lily (白马蹄莲)",   # 20
    "fire lily (火百合)",                # 21
    "pincushion flower (松虫草)",         # 22
    "fritillary (贝母)",                # 23
    "red ginger (红姜花)",               # 24
    "grape hyacinth (葡萄风信子)",        # 25
    "corn poppy (虞美人)",               # 26
    "prince of wales feathers (鸡冠花)",  # 27
    "stemless gentian (无茎龙胆)",        # 28
    "artichoke (洋蓟)",                  # 29
    "sweet william (美洲石竹)",           # 30
    "carnation (康乃馨)",                # 31
    "garden phlox (天蓝绣球)",            # 32
    "love in the mist (黑种草)",          # 33
    "mexican aster (墨西哥紫菀)",         # 34
    "alpine sea holly (高山刺芹)",        # 35
    "ruby-lipped cattleya (卡特兰)",      # 36
    "cape flower (帝王花)",               # 37
    "great masterwort (大星芹)",          # 38
    "siam tulip (暹罗郁金香)",            # 39
    "lenten rose (铁筷子)",              # 40
    "barbeton daisy (非洲菊)",            # 41
    "daffodil (水仙)",                   # 42
    "sword lily (剑兰)",                 # 43
    "poinsettia (一品红)",               # 44
    "bolero deep blue (深蓝飞燕草)",      # 45
    "wallflower (桂竹香)",               # 46
    "marigold (万寿菊)",                 # 47
    "buttercup (毛茛)",                  # 48
    "oxeye daisy (牛眼菊)",               # 49
    "common dandelion (蒲公英)",           # 50
    "petunia (矮牵牛)",                  # 51
    "wild pansy (三色堇)",               # 52
    "primula (报春花)",                  # 53
    "sunflower (向日葵)",                # 54
    "pelargonium (天竺葵)",               # 55
    "bishop of llandaff (大丽花)",        # 56
    "gaura (山桃草)",                    # 57
    "geranium (老鹳草)",                 # 58
    "orange dahlia (橙大丽花)",           # 59
    "pink-yellow dahlia (粉黄大丽花)",     # 60
    "cautleya spicata (距药姜)",          # 61
    "japanese anemone (秋牡丹)",          # 62
    "black-eyed susan (金光菊)",          # 63
    "silverbush (银叶花)",               # 64
    "californian poppy (花菱草)",         # 65
    "osteospermum (蓝目菊)",              # 66
    "spring crocus (番红花)",             # 67
    "bearded iris (鸢尾)",               # 68
    "windflower (银莲花)",               # 69
    "tree poppy (木罂粟)",               # 70
    "gazania (勋章菊)",                  # 71
    "azalea (杜鹃花)",                   # 72
    "water lily (睡莲)",                 # 73
    "rose (玫瑰)",                       # 74
    "thorn apple (曼陀罗)",              # 75
    "morning glory (牵牛花)",            # 76
    "passion flower (西番莲)",            # 77
    "lotus lotus (荷花)",               # 78
    "toad lily (油点百合)",               # 79
    "anthurium (红掌)",                  # 80
    "frangipani (鸡蛋花)",               # 81
    "clematis (铁线莲)",                 # 82
    "hibiscus (木槿)",                   # 83
    "columbine (耧斗菜)",                # 84
    "desert-rose (沙漠玫瑰)",            # 85
    "tree mallow (树锦葵)",              # 86
    "magnolia (玉兰)",                   # 87
    "cyclamen (仙客来)",                 # 88
    "watercress (水田芥)",               # 89
    "canna lily (美人蕉)",               # 90
    "hippeastrum (朱顶红)",              # 91
    "bee balm (蜂薄荷)",                 # 92
    "ball moss (球苔)",                  # 93
    "foxglove (毛地黄)",                 # 94
    "bougainvillea (三角梅)",            # 95
    "camellia (山茶花)",                 # 96
    "mallow (锦葵)",                    # 97
    "mexican petunia (翠芦莉)",           # 98
    "bromelia (凤梨花)",                 # 99
    "blanket flower (天人菊)",            # 100
    "trumpet creeper (凌霄花)",          # 101
    "blackberry lily (射干)"             # 102
]

# 页面配置
st.set_page_config(
    page_title="花朵图像分类系统",
    page_icon="🌸",
    layout="wide"
)

# 加载模型（缓存，只加载一次）
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 102)

    try:
        model.load_state_dict(torch.load("./best_model.pth", map_location="cpu"))
        st.sidebar.success("✅ 模型加载成功")
    except:
        st.sidebar.warning("⚠️ 未找到模型文件，使用随机初始化模型")

    model.eval()
    return model

# 图像预处理
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# 预测函数
def predict(image, model):
    img_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probs = probabilities[0].numpy()

    # 获取Top-5
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5_probs = probs[top5_idx]
    top5_names = [FLOWER_NAMES[i] if i < len(FLOWER_NAMES) else f"类别{i}" for i in top5_idx]

    return top5_idx, top5_probs, top5_names

# ==================== 主页面 ====================
st.title("🌸 花朵图像分类系统")
st.markdown("基于ResNet18迁移学习的102类花卉识别系统")
st.markdown("---")

# 侧边栏
st.sidebar.header("📋 系统信息")
model = load_model()
st.sidebar.info("模型：ResNet18（迁移学习）")
st.sidebar.info("类别数：102种花卉")
st.sidebar.info("输入尺寸：224×224")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 设置")
show_plot = st.sidebar.checkbox("显示概率分布图", value=True)

# 主内容区
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 上传图片")
    uploaded_file = st.file_uploader("选择一张花朵图片...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="上传的图片", use_column_width=True)

        # 预测按钮
        if st.button("🔍 开始识别", type="primary"):
            with st.spinner("正在识别中..."):
                top5_idx, top5_probs, top5_names = predict(image, model)
            st.session_state['prediction'] = (top5_idx, top5_probs, top5_names)
            st.success("识别完成！")

with col2:
    if uploaded_file is not None and 'prediction' in st.session_state:
        top5_idx, top5_probs, top5_names = st.session_state['prediction']

        st.subheader("🎯 识别结果")

        # 显示Top-1结果
        st.markdown(f"### 最可能：**{top5_names[0]}**")
        st.markdown(f"### 置信度：**{top5_probs[0]*100:.2f}%**")
        st.markdown("---")

        # 显示Top-5结果
        st.markdown("#### Top-5 预测结果：")
        for i, (name, prob) in enumerate(zip(top5_names, top5_probs)):
            st.markdown(f"{i+1}. **{name}** — {prob*100:.2f}%")
            st.progress(float(prob))

        # 绘制概率分布图
        if show_plot:
            st.markdown("---")
            st.markdown("#### 概率分布图：")
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.barh(range(5), top5_probs[::-1]*100, color="skyblue")
            ax.set_yticks(range(5))
            ax.set_yticklabels(top5_names[::-1])
            ax.set_xlabel("概率 (%)")
            ax.set_title("Top-5 预测概率分布")
            ax.set_xlim(0, 100)

            for bar, prob in zip(bars, top5_probs[::-1]*100):
                ax.text(prob + 1, bar.get_y() + bar.get_height()/2,
                        f"{prob:.1f}%", va="center")

            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.subheader("👋 欢迎使用")
        st.info("请在左侧上传一张花朵图片，点击\"开始识别\"按钮查看结果。")
        st.markdown("### 支持的花卉类别：")
        st.markdown("- 共 **102** 种花卉")
        st.markdown("- 包括玫瑰、郁金香、向日葵、百合等")
        st.markdown("- 模型基于迁移学习训练")

# 页脚
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>花朵图像分类系统 | 专业实习项目</div>",
            unsafe_allow_html=True)
