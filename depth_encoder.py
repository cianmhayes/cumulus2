from cumulus2_vae_model import Cumulus2Vae
from typing import Dict, Sequence, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from model_trainer import LossCalculator, ModuleFactory, ModuleOptimizer, ModuleSnapshotSaver, Progress
import math
import torch.optim as optim

class DepthEncoder(torch.nn.Module):
    def __init__(self, encoded_dimensions:int) -> None:
        super().__init__()
        self.encoded_dimensions = encoded_dimensions

        self.encode_outer = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4, padding=3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))

        self.encode_middle = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))

        self.encode_inner = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))

        self.encode_pseudo_dense = nn.Sequential(
            nn.Conv2d(128, encoded_dimensions, 1),
            nn.LeakyReLU())

        self.encode_mu = nn.Conv2d(encoded_dimensions, encoded_dimensions, 1)
        self.encode_log_var = nn.Conv2d(encoded_dimensions, encoded_dimensions, 1)

    def forward(self, source_image:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_encoding = self.encode_outer(source_image)
        conv_encoding = self.encode_middle(conv_encoding)
        conv_encoding = self.encode_inner(conv_encoding)
        conv_encoding = self.encode_pseudo_dense(conv_encoding)
        return self.encode_mu(conv_encoding), self.encode_log_var(conv_encoding)

class DepthEncoderFactory(ModuleFactory):
    def __init__(self, encoded_dimensions:int) -> None:
        self.encoded_dimensions = encoded_dimensions

    def create_instance(self) -> DepthEncoder:
        return DepthEncoder(self.encoded_dimensions)

    def get_construction_parameters(self) -> Dict:
        return {
            "encoded_dimensions": self.encoded_dimensions}

class DepthEncoderLoss(LossCalculator):
    def __init__(self, vae_model:Cumulus2Vae) -> None:
        super().__init__()
        self.vae_model = vae_model

    def set_device(self, device:torch.device) -> None:
        super().set_device(device)
        self.vae_model.to(device)

    def get_loss(
            self,
            sample:Sequence[torch.Tensor],
            module:torch.nn.Module,
            snapshot_savers:Sequence[ModuleSnapshotSaver] = None,
            progress:Progress = None) -> torch.Tensor:
        input_sample = sample[0].to(self.device)
        target_mu = sample[1].to(self.device)
        target_log_var = sample[2].to(self.device)
        mu, log_var = module(input_sample)
        if snapshot_savers and progress:
            for snapshot_saver in snapshot_savers:
                z = self.vae_model.reparameterize(mu, log_var)
                transcoded_image = self.vae_model.decode(z)
                snapshot_saver.save_sample(transcoded_image, progress)
        return F.mse_loss(mu, target_mu, reduction="sum") + F.mse_loss(log_var, target_log_var, reduction="sum")

if __name__ == "__main__":
    model = DepthEncoder(16)
    input_tensor = torch.zeros((1,1, 496, 480))
    mu, log_var = model(input_tensor)
    print(mu.size())
    print(log_var.size())