from torch import nn
from typing import Optional
from abc import abstractmethod
from torch.utils.data import Dataset
import torch
import os
import wandb
import pandas as pd

import cv2
import albumentations as A

from typing import Optional
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
# from .base import ImageRetrievalDataset
# from .flickr8k import Flickr8kDataset
# from .flickr30k import Flickr30kDataset

DATASET_LOOKUP = {
    "flickr8k": Flickr8kDataset,
    # 'capture-24', Capture24Dataset
    # "flickr30k": Flickr30kDataset
}

class ImageRetrievalDataModule(LightningDataModule):
    def __init__(
        self,
        artifact_id: str,
        dataset_name: str,
        val_split: float = 0.2,
        tokenizer_alias: Optional[str] = None,
        target_size: int = 224,
        max_length: int = 100,
        lazy_loading: bool = False,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        num_workers: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.artifact_id = artifact_id
        self.dataset_name = dataset_name
        self.val_split = val_split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_alias)
        self.target_size = target_size
        self.max_length = max_length
        self.lazy_loading = lazy_loading
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
    
    @staticmethod
    def split_data(dataset: ImageRetrievalDataset, val_split: float):
        train_length = int((1 - val_split) * len(dataset))
        val_length = len(dataset) - train_length
        train_dataset, val_dataset = random_split(dataset, lengths=[train_length, val_length])
        return train_dataset, val_dataset

    def setup(
        self,
        stage: Optional[str] = None,
    ) -> None:
        dataset = DATASET_LOOKUP[self.dataset_name](
            artifact_id=self.artifact_id,
            tokenizer=self.tokenizer,
            target_size=self.target_size,
            max_length=self.max_length,
            lazy_loading=self.lazy_loading,
        )
        self.train_dataset, self.val_dataset = self.split_data(dataset, val_split=self.val_split)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
        )
class ImageRetrievalDataset(Dataset):
    def __init__(
        self,
        artifact_id: str,
        tokenizer=None,
        target_size: Optional[int] = None,
        max_length: int = 200,
        lazy_loading: bool = False,
    ) -> None:
        super().__init__()
        self.artifact_id = artifact_id
        self.target_size = target_size
        self.image_files, self.captions = self.fetch_dataset()
        self.lazy_loading = lazy_loading
        self.images = self.image_files


        assert tokenizer is not None


        self.tokenizer = tokenizer


        self.tokenized_captions = tokenizer(
            list(self.captions), 
            padding=True, 
            truncation=True, 
            max_length=max_length,
            return_tensors='pt'
        )
        self.transforms = A.Compose(
            [
                A.Resize(target_size, target_size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )


    @abstractmethod
    def fetch_dataset(self):
        pass


    def __len__(self):
        return len(self.captions)


    def __getitem__(self, index):
        item = {
            key: values[index]
            for key, values in self.tokenized_captions.items()
        }
        image = cv2.imread(self.image_files[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        item["caption"] = self.captions[index]
        return item

class Flickr8kDataset(ImageRetrievalDataset):
    def __init__(
        self,
        artifact_id: str,
        tokenizer=None,
        target_size: Optional[int] = None,
        max_length: int = 100,
        lazy_loading: bool = False,
    ) -> None:
        super().__init__(artifact_id, tokenizer, target_size, max_length, lazy_loading)


    def fetch_dataset(self):
        if wandb.run is None:
            api = wandb.Api()
            artifact = api.artifact(self.artifact_id, type="dataset")
        else:
            artifact = wandb.use_artifact(self.artifact_id, type="dataset")
        artifact_dir = artifact.download()
        annotations = pd.read_csv(os.path.join(artifact_dir, "captions.txt"))
        image_files = [
            os.path.join(artifact_dir, "Images", image_file)
            for image_file in annotations["image"].to_list()
        ]
        for image_file in image_files:
            assert os.path.isfile(image_file)
        captions = annotations["caption"].to_list()
        return image_files, captions