
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import torch
class MyData(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        self.train_labels = Label

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        target = self.train_labels[index]

        return image_ms, image_pan, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)

class MyData1(Dataset):
    def __init__(self, MS4, Pan, xy, cut_size, noise_std=0.02):
        self.train_data1 = MS4
        self.train_data2 = Pan

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

        self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        # transforms.RandomErasing(p=0.5)
                    ])
        
        self.noise_std = noise_std

    def add_gaussian_noise(self, image):
        """img: Tensor, shape [C, H, W]"""
        return image + torch.randn_like(image) * self.noise_std

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        image_ms_view1 = self.add_gaussian_noise(self.transform(image_ms))
        image_ms_view2 = self.transform(image_ms)

        image_pan_view1 = self.transform(image_pan)
        image_pan_view2 = self.add_gaussian_noise(self.transform(image_pan))

        return image_ms_view1, image_ms_view2, image_pan_view1, image_pan_view2, index, locate_xy

    def __len__(self):
        return len(self.gt_xy)