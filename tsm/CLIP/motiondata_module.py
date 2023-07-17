from torch import nn
from typing import Optional
from abc import abstractmethod
from torch.utils.data import Dataset
import torch
import os
import wandb
import pandas as pd
from datasets import load_dataset
import numpy as np
import cv2
import scipy.signal as signal
import albumentations as A

from typing import Optional
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
class MotionRetrievalDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer=None,
        max_length: int = 300,
        downsample: bool = False,
        original_sampling_rate: int = 100,
        desired_sampling_rate: int = 30,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.downsample = downsample
        self.original_sampling_rate = original_sampling_rate
        self.desired_sampling_rate = desired_sampling_rate
        # self.x_values, self.captions = self.fetch_dataset()
        self.tokenized_captions = self.tokenizer(
            dataset['y'], 
            padding=True, 
            truncation=True, 
            max_length=self.max_length,
            return_tensors='pt'
        )
    # def fetch_dataset(self):
    #     x_values = []
    #     y_values = []
    
    # def process_single_example(self, example):
    #     x_value = torch.tensor(example['x']).clone().detach()
    #     if self.downsample:
    #         x_value = self.downsample_data(x_value)
    #     if x_value.shape[0] > self.max_length:
    #         x_value = x_value[:self.max_length]
    #     elif x_value.shape[0] < self.max_length:
    #         padding = torch.zeros((self.max_length - x_value.shape[0], 3))
    #         x_value = torch.cat([x_value, padding], dim=0)
        # for example in self.dataset:
        #     x_value = torch.tensor(example['x']).clone().detach()
        #     if self.downsample:
        #         x_value = self.downsample_data(x_value)
        #     if x_value.shape[0] > self.max_length:
        #         x_value = x_value[:self.max_length]
        #     elif x_value.shape[0] < self.max_length:
        #         padding = torch.zeros((self.max_length - x_value.shape[0], 3))
        #         x_value = torch.cat([x_value, padding], dim=0)
        #     x_values.append(x_value)
        #     y_values.append(example['y'])

        # x_values = torch.stack(x_values)

        # return x_values, y_values
    
    def downsample_data(self, data):
        # Define the low-pass filter
        nyquist_frequency = 0.5 * self.original_sampling_rate
        cutoff_frequency = 0.5 * self.desired_sampling_rate
        filter_order = 4
        b, a = signal.butter(filter_order, cutoff_frequency / nyquist_frequency, 'low')

        # Apply the low-pass filter
        filtered_data = signal.filtfilt(b, a, data, axis=0)

        # Decimate the filtered data
        decimation_factor = int(self.original_sampling_rate / self.desired_sampling_rate)
        downsampled_data = filtered_data[::decimation_factor]

        return downsampled_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = {
            key: values[index]
            for key, values in self.tokenized_captions.items()
        }
        item['motion'] = self.dataset[index]['x']

        item["caption"] = self.dataset[index]['y']
        return item
    

def test_motion_retrieval_dataset():
    # Load the dataset
    dataset = load_dataset("alexshengzhili/Accel2ActivityCrawl", split='capture24_100hz_w10_o0_rawlabel')

    # Create a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Create a dataset instance
    motion_dataset = MotionRetrievalDataset(dataset, 
        tokenizer=tokenizer, 
        max_length=300, 
        original_sampling_rate=100,
        desired_sampling_rate=30)

    # Test __len__ method
    assert len(motion_dataset) == len(dataset)

    # Test __getitem__ method
    item = motion_dataset[0]
    assert isinstance(item, dict)
    print(item['motion'].shape)
    # assert item['x'].shape == torch.Size([item['x'].shape[0], 3])
    print(item.keys())
    #assert item['y'] == dataset[0]['y']

    # Create a data loader
    batch_size = 4
    motion_dataloader = DataLoader(motion_dataset, batch_size=batch_size, shuffle=True)

    # # Test data loader
    batch = next(iter(motion_dataloader))
    # assert isinstance(batch, dict)
    assert set(batch.keys()) == {'input_ids', 'attention_mask', 'motion', 'caption'}
    print(batch['motion'].shape)
    assert len(batch['caption']) == batch_size

class MotionRetrievalDataModule(LightningDataModule):
    def __init__(
        self,
        dataset,
        tokenizer_alias: Optional[str] = None,
        val_split: float = 0.2,
        max_length: int = 100,
        batch_size: int = 32,
        num_workers: int = 4,
        downsample: bool = False,
        origional_sampling_rate: int = 100,
        desired_sampling_rate: int = 30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.val_split = val_split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_alias)
        self.max_length = max_length
        self.train_batch_size = batch_size
        self.val_batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.downsample = downsample
        self.origional_sampling_rate = origional_sampling_rate
        self.desired_sampling_rate = desired_sampling_rate

    @staticmethod
    def split_data(dataset: MotionRetrievalDataset, val_split: float):
        train_length = int((1 - val_split) * len(dataset))
        val_length = len(dataset) - train_length
        print(train_length, val_length)
        train_dataset, val_dataset = random_split(dataset, lengths=[train_length, val_length])
        return train_dataset, val_dataset

    def setup(
        self,
        stage: Optional[str] = None,
    ) -> None:
        dataset = MotionRetrievalDataset(
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            downsample=self.downsample,
            original_sampling_rate=self.origional_sampling_rate,
            desired_sampling_rate=self.desired_sampling_rate
        )
        self.train_dataset, self.val_dataset = self.split_data(dataset, val_split=self.val_split)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,

        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
def test_module():
    dataset = load_dataset("alexshengzhili/Accel2ActivityCrawl", split='realworld')

    text_encoder_alias = "distilbert-base-uncased"
    data_module = MotionRetrievalDataModule(
        dataset=dataset,  # Replace with your motion dataset
        tokenizer_alias=text_encoder_alias,
        max_length=300,
        train_batch_size=16,
        val_batch_size=16,
        num_workers=4,
    )
    data_module.setup()
class Accel2ActivityCrawlDataset(MotionRetrievalDataset):
    def __init__(
        self,
        dataset,
        tokenizer=None,
        max_length: int = 100,
    ) -> None:
        super().__init__(dataset, tokenizer, max_length)

    def fetch_dataset(self):
        x_values = self.dataset['x']
        y_values = self.dataset['y']
        return x_values, y_values
    
if __name__ == '__main__':
    test_motion_retrieval_dataset()
