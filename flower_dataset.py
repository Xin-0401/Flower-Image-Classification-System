# 自定义Flower-102数据集类
import os
from PIL import Image
from torch.utils.data import Dataset
import scipy.io as sio

class FlowerDataset(Dataset):
    """自定义花卉数据集，直接从jpg文件夹和.mat标签文件加载"""

    def __init__(self, root, split='train', transform=None):
        """
        Args:
            root: 数据根目录 (e.g. "./data/flowers-102")
            split: 'train', 'val', 或 'test'
            transform: 图像变换
        """
        self.root = root
        self.split = split
        self.transform = transform

        # 加载标签
        labels_mat = sio.loadmat(os.path.join(root, 'imagelabels.mat'))
        self.labels = labels_mat['labels'].flatten()  # shape: (8189,)

        # 加载split
        setid_mat = sio.loadmat(os.path.join(root, 'setid.mat'))
        if split == 'train':
            self.ids = setid_mat['trnid'].flatten()
        elif split == 'val':
            self.ids = setid_mat['valid'].flatten()
        elif split == 'test':
            self.ids = setid_mat['tstid'].flatten()
        else:
            raise ValueError(f"Unknown split: {split}")

        self.jpg_dir = os.path.join(root, 'jpg')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.jpg_dir, f'image_{img_id:05d}.jpg')
        image = Image.open(img_path).convert('RGB')
        label = int(self.labels[img_id - 1] - 1)  # 转为0-indexed

        if self.transform:
            image = self.transform(image)

        return image, label
