import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch


class MSTEPatchDataset(Dataset):
    def __init__(self, root_dir, subjects=None, image_size=64, subject_to_mst=None, mst_to_vec=None, augment=True):
        self.image_paths = []
        self.targets = []

        self.subject_to_mst = subject_to_mst
        self.mst_to_vec = mst_to_vec

        self.include_targets = self.subject_to_mst is not None and self.mst_to_vec is not None

        for subject in os.listdir(root_dir):
            if subjects and subject not in subjects:
                continue
            subject_path = os.path.join(root_dir, subject)
            if not os.path.isdir(subject_path):
                continue
            mst = self.subject_to_mst.get(
                subject) if self.include_targets else None
            color_vec = self.mst_to_vec.get(mst) if mst is not None else None

            for file in os.listdir(subject_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(subject_path, file))
                    if self.include_targets:
                        self.targets.append(color_vec)

        # Define safe augmentation pipeline
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((72, 72)),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),  # values in [0, 1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        x = self.transform(image)
        if self.include_targets:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, y
        return x
