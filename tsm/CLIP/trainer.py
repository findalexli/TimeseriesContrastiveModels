from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.tuner import Tuner

from motionmodel import CLIPDualEncoderModel
from motiondata_module import MotionRetrievalDataModule
from motionmodel import MotionCNNEncoder, MotionTFEncoder
import torch
from datasets import load_dataset, load_from_disk
torch.set_float32_matmul_precision('high')
text_encoder_alias = "distilbert-base-uncased"
# iterable_dataset = dataset.to_iterable_dataset(num_shards=128)
# shuffled_iterable_dataset = iterable_dataset.shuffle(seed=42, buffer_size=1000)
# dataset = load_from_disk('/home/ubuntu/ssl-wearables/CLIP/capture24_30hz_w10_o0_unfileterd_converted_y_non_empty_non_sleep').with_format("torch")
split_name = 'wisdm'
dataset = load_dataset("alexshengzhili/Accel2ActivityCrawl", split=split_name)

dataset.shuffle(seed=42)
max_length = 300


# Create a tokenizer
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Create a dataset instance
# motion_dataset = MotionRetrievalDataset(dataset, tokenizer=tokenizer, max_length=300)



# Instantiate the motion data module
data_module = MotionRetrievalDataModule(
    dataset=dataset,  # Replace with your motion dataset
    tokenizer_alias=text_encoder_alias,
    max_length=300,
    batch_size=320*4,
    num_workers=8,
    downsample=False,
)
data_module.setup()

# Instantiate the motion encoder
#motion_encoder = MotionEncoder(d_model=256, nhead=8, num_layers=5, dim_feedforward=1024)
motion_encoder = MotionCNNEncoder()
# Instantiate the CLIP dual encoder model with the motion encoder
model = CLIPDualEncoderModel(
    text_encoder_alias, 
    motion_encoder,
    projection_dims=512,
    text_encoder_trainable = True)
# compiled_model = torch.compile(model)

# Instantiate the logger
logger = WandbLogger(log_model="all", name=f"Preatrained-CNN-Encoder-512-projection-dim-on-{split_name}-dataset")

# Define callbacks
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint()

# Instantiate the Trainer
trainer = Trainer(
    accelerator="gpu", 
    devices=[0], 
    # strategy="auto",
    precision=16,
    logger=logger,
    callbacks=[lr_monitor, checkpoint_callback],
    max_epochs=10,
    log_every_n_steps=1,
)
tuner = Tuner(trainer)
# tuner.scale_batch_size(model, data_module)
lr_finder = tuner.lr_find(model, data_module.train_dataloader(), data_module.val_dataloader(), attr_name='motion_encoder_lr')
print(lr_finder.results)
new_lr = lr_finder.suggestion()
print(new_lr)
model.motion_encoder_lr = new_lr
# Train the model
trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())