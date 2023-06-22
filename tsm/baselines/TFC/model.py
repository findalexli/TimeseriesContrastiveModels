from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
class TFC_resnet(nn.Module): 
    def __init__(self, configs):
        super().__init__()
        self.conv_block1_t = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2_t = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3_t = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.projector_t = nn.Sequential(
            nn.Linear(configs.CNNoutput_channel * configs.final_out_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.conv_block1_f = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2_f = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3_f = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.projector_f = nn.Sequential(
            nn.Linear(configs.CNNoutput_channel * configs.final_out_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )


    def forward(self, x_in_t, x_in_f):

        """Time-based Contrastive Encoder"""
        x = self.conv_block1_t(x_in_t)
        x = self.conv_block2_t(x)
        x = self.conv_block3_t(x)
        h_time = x.reshape(x.shape[0], -1)
        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.conv_block1_f(x_in_f)
        f = self.conv_block2_f(f)
        f = self.conv_block3_f(f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


"""Two contrastive encoders"""
class TFC(nn.Module):
    def __init__(self, configs):
        super(TFC, self).__init__()

        encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned, nhead=2, )
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        encoder_layers_f = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned,nhead=2,)
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )


    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.transformer_encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


"""Downstream classifier only used in finetuning"""
class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred
