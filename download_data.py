# 通过HuggingFace中国镜像下载Flower-102数据集（修复标签映射）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
import numpy as np
from PIL import Image
import io
import scipy.io as sio
import json

print("正在从HuggingFace镜像下载Flower-102数据集...")

ds = load_dataset("nelorth/oxford-flowers")
print(f"数据集下载完成！")
print(f"训练集: {len(ds['train'])} 张")
print(f"测试集: {len(ds['test'])} 张")

# 获取正确的标签映射
# HuggingFace的label是按字符串排序的(1,10,100,...,99)，需要转回数字顺序(1,2,3,...,102)
label_names = ds['train'].features['label'].names  # ['1','10','100','101','102','11',...]
print(f"\nHuggingFace标签顺序(前10个): {label_names[:10]}")

# 构建映射：HF label index -> 原始类别编号(1-102)
hf_to_orig = {}
for hf_idx, name in enumerate(label_names):
    hf_to_orig[hf_idx] = int(name)  # '1'->1, '10'->10, '100'->100, ...

print(f"HF label 0 -> 原始类别 {hf_to_orig[0]}")
print(f"HF label 1 -> 原始类别 {hf_to_orig[1]}")
print(f"HF label 2 -> 原始类别 {hf_to_orig[2]}")

# Oxford Flower-102 类别名称映射 (1-102)
CAT_TO_NAME = {
    "1": "pink primrose", "2": "hard-leaved pocket orchid", "3": "canterbury bells",
    "4": "sweet pea", "5": "english marigold", "6": "tiger lily", "7": "moon orchid",
    "8": "bird of paradise", "9": "monkshood", "10": "globe thistle",
    "11": "snapdragon", "12": "colt's foot", "13": "king protea", "14": "spear thistle",
    "15": "yellow iris", "16": "globe-flower", "17": "purple coneflower",
    "18": "peruvian lily", "19": "balloon flower", "20": "giant white arum lily",
    "21": "fire lily", "22": "pincushion flower", "23": "fritillary",
    "24": "red ginger", "25": "grape hyacinth", "26": "corn poppy",
    "27": "prince of wales feathers", "28": "stemless gentian", "29": "artichoke",
    "30": "sweet william", "31": "carnation", "32": "garden phlox",
    "33": "love in the mist", "34": "mexican aster", "35": "alpine sea holly",
    "36": "ruby-lipped cattleya", "37": "cape flower", "38": "great masterwort",
    "39": "siam tulip", "40": "lenten rose", "41": "barbeton daisy",
    "42": "daffodil", "43": "sword lily", "44": "poinsettia",
    "45": "bolero deep blue", "46": "wallflower", "47": "marigold",
    "48": "buttercup", "49": "oxeye daisy", "50": "common dandelion",
    "51": "petunia", "52": "wild pansy", "53": "primula", "54": "sunflower",
    "55": "pelargonium", "56": "bishop of llandaff", "57": "gaura",
    "58": "geranium", "59": "orange dahlia", "60": "pink-yellow dahlia",
    "61": "cautleya spicata", "62": "japanese anemone", "63": "black-eyed susan",
    "64": "silverbush", "65": "californian poppy", "66": "osteospermum",
    "67": "spring crocus", "68": "bearded iris", "69": "windflower",
    "70": "tree poppy", "71": "gazania", "72": "azalea", "73": "water lily",
    "74": "rose", "75": "thorn apple", "76": "morning glory",
    "77": "passion flower", "78": "lotus lotus", "79": "toad lily",
    "80": "anthurium", "81": "frangipani", "82": "clematis", "83": "hibiscus",
    "84": "columbine", "85": "desert-rose", "86": "tree mallow",
    "87": "magnolia", "88": "cyclamen", "89": "watercress", "90": "canna lily",
    "91": "hippeastrum", "92": "bee balm", "93": "ball moss", "94": "foxglove",
    "95": "bougainvillea", "96": "camellia", "97": "mallow",
    "98": "mexican petunia", "99": "bromelia", "100": "blanket flower",
    "101": "trumpet creeper", "102": "blackberry lily"
}

# 保存类别名称映射
with open("./flower_names.json", "w", encoding="utf-8") as f:
    json.dump(CAT_TO_NAME, f, ensure_ascii=False, indent=2)
print("flower_names.json 已保存")

# 创建torchvision期望的目录结构
base_dir = "./data/flowers-102"
jpg_dir = os.path.join(base_dir, "jpg")
os.makedirs(jpg_dir, exist_ok=True)

print("\n将数据转换为标准格式（修复标签映射）...")

all_labels = []
img_idx = 1

# 处理train split
for item in ds['train']:
    img = item['image']
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img))
    img = img.convert('RGB')
    img.save(os.path.join(jpg_dir, f"image_{img_idx:05d}.jpg"))
    # 使用正确的标签映射
    orig_label = hf_to_orig[item['label']]
    all_labels.append(orig_label)
    img_idx += 1

# 处理test split
for item in ds['test']:
    img = item['image']
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img))
    img = img.convert('RGB')
    img.save(os.path.join(jpg_dir, f"image_{img_idx:05d}.jpg"))
    orig_label = hf_to_orig[item['label']]
    all_labels.append(orig_label)
    img_idx += 1

total = img_idx - 1
n_train_hf = len(ds['train'])

print(f"共保存 {total} 张图片")

# 随机划分train为train/val (80%/20%)
np.random.seed(42)
all_train_ids = list(range(1, n_train_hf + 1))
np.random.shuffle(all_train_ids)

n_train = int(n_train_hf * 0.8)
n_val = n_train_hf - n_train

train_ids = sorted(all_train_ids[:n_train])
val_ids = sorted(all_train_ids[n_train:])
test_ids = list(range(n_train_hf + 1, total + 1))

# 验证类别分布
train_labels_arr = np.array(all_labels)
train_set_labels = train_labels_arr[np.array(train_ids) - 1]
val_set_labels = train_labels_arr[np.array(val_ids) - 1]
test_set_labels = train_labels_arr[np.array(test_ids) - 1]
print(f"训练集: {len(train_ids)} 张, {len(np.unique(train_set_labels))} 类")
print(f"验证集: {len(val_ids)} 张, {len(np.unique(val_set_labels))} 类")
print(f"测试集: {len(test_ids)} 张, {len(np.unique(test_set_labels))} 类")

# 保存mat文件
sio.savemat(os.path.join(base_dir, "imagelabels.mat"),
            {"labels": np.array(all_labels, dtype=np.uint8)})

sio.savemat(os.path.join(base_dir, "setid.mat"),
            {"trnid": np.array(train_ids, dtype=np.uint16),
             "valid": np.array(val_ids, dtype=np.uint16),
             "tstid": np.array(test_ids, dtype=np.uint16)})

print(f"\nimagelabels.mat 和 setid.mat 已保存")
print("数据集准备完成！标签映射已修复。")
