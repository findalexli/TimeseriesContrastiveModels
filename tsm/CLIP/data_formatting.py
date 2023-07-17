import torch
from datasets import load_dataset
dataset = load_dataset('alexshengzhili/Accel2ActivityCrawl', split='capture24_30hz_w10_o0_unfileterd_rawlabel').with_format("torch")
max_length = 300

def process_single_example(self, example):
    x_value = torch.tensor(example['x'])
    if self.downsample:
        x_value = self.downsample_data(x_value)
    if x_value.shape[0] > max_length:
        x_value = x_value[:max_length]
    elif x_value.shape[0] < max_length:
        padding = torch.zeros((max_length - x_value.shape[0], 3))
        x_value = torch.cat([x_value, padding], dim=0)
    example['x'] = x_value
    return example

formated_dataset = dataset.map(process_single_example)
formated_dataset