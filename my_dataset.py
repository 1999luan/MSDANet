import os

import PIL
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class LungDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(LungDataset, self).__init__()
        data_root = os.path.join(root, "CVC", "training" if train else "test")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.flag = "training" if train else "test"
        self.transforms = transforms

        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        mask_names = [i for i in os.listdir(os.path.join(data_root, "masks")) if i.endswith(".tif")]
        self.mask_list = [os.path.join(data_root, "masks", i) for i in mask_names]

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        #img = Image.open(self.img_list[idx]).convert('RGB')
        #img = img.resize((256, 256))
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        #img = np.array(img)
        #masks = Image.open(self.mask_list[idx]).convert('L')
        mask = cv2.imread(self.mask_list[idx])
        #masks = masks.resize((256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
        #masks = np.array(masks) / 255 #要想用transform中的RandomRotation(90),RandomZoom((0.9, 1.1))，需要将/255操作在ToTenSor中的target处理前实现。这里不需要/255
        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        #mask = Image.fromarray(masks)
        mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) / 255)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #img = Image.fromarray(img)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

