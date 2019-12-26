from typing import Dict, Sequence, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from model_trainer import LossCalculator, ModuleFactory, ModuleOptimizer, ModuleSnapshotSaver, Progress
import math
import torch.optim as optim

class Cumulus2Vae(torch.nn.Module):
    def __init__(self, channels, encoded_dimensions) -> None:
        super().__init__()
        self.encoded_dimensions = encoded_dimensions

        self.encode_outer = nn.Sequential(
            nn.Conv2d(channels, 32, 8, stride=4, padding=3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))

        self.encode_middle = nn.Sequential(
            nn.Conv2d(32, 64, 8, stride=4, padding=3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))

        self.encode_inner = nn.Sequential(
            nn.Conv2d(64, 128, 8, stride=4, padding=3, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        
        self.encode_pseudo_dense = nn.Sequential(
            nn.Conv2d(128, encoded_dimensions, 1),
            nn.LeakyReLU())

        self.encode_mu = nn.Conv2d(encoded_dimensions, encoded_dimensions, 1)
        self.encode_log_var = nn.Conv2d(encoded_dimensions, encoded_dimensions, 1)

        self.decode_pseudo_dense = nn.Sequential(
            nn.Conv2d(encoded_dimensions, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.decode_inner = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 8, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.decode_middle = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 8, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.decode_outer = nn.Sequential(
            nn.ConvTranspose2d(32, channels, 8, stride=4, padding=2, bias=False),
            nn.Sigmoid())

    def reparameterize(self, mu:torch.Tensor, log_var:torch.Tensor) -> torch.Tensor:
        std_dev = torch.exp(0.5*log_var)
        epsilon = torch.randn_like(std_dev)
        return mu + std_dev*epsilon

    def encode(self, source_image:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_encoding = self.encode_outer(source_image)
        conv_encoding = self.encode_middle(conv_encoding)
        conv_encoding = self.encode_inner(conv_encoding)
        conv_encoding = self.encode_pseudo_dense(conv_encoding)
        return self.encode_mu(conv_encoding), self.encode_log_var(conv_encoding)

    def decode(self, z:torch.Tensor) -> torch.Tensor:
        decode = self.decode_pseudo_dense(z)
        decode = self.decode_inner(decode)
        decode = self.decode_middle(decode)
        decode = self.decode_outer(decode)
        return decode

    def forward(self, source_image:torch.Tensor) -> Sequence[torch.Tensor]:
        mu, log_variance = self.encode(source_image)
        z = self.reparameterize(mu, log_variance)
        return self.decode(z), mu, log_variance, z


class Cloud2VaeFactory(ModuleFactory):
    def __init__(self, image_channels:int, encoded_dimensions:int) -> None:
        self.image_channels = image_channels
        self.encoded_dimensions = encoded_dimensions

    def create_instance(self) -> Cumulus2Vae:
        return Cumulus2Vae(self.image_channels, self.encoded_dimensions)

    def get_construction_parameters(self) -> Dict:
        return {
            "encoded_dimensions": self.encoded_dimensions,
            "image_channels": self.image_channels}


class Cloud2VaeLoss(LossCalculator):
    def __init__(self) -> None:
        super().__init__()

    def get_loss(
            self,
            sample:Sequence[torch.Tensor],
            module:torch.nn.Module,
            snapshot_savers:Sequence[ModuleSnapshotSaver] = None,
            progress:Progress = None) -> torch.Tensor:
        input_sample = sample[0].to(self.device)
        transcoded_image, mu, log_variance, _ = module(input_sample)
        if snapshot_savers and progress:
            for snapshot_saver in snapshot_savers:
                snapshot_saver.save_sample(transcoded_image, progress)
        return self.loss_function(transcoded_image, input_sample, mu, log_variance)

    def loss_function(
            self,
            decoded_values:torch.Tensor,
            values:torch.Tensor,
            mu:torch.Tensor,
            log_variance:torch.Tensor):
        cross_entropy = F.binary_cross_entropy(decoded_values, values, reduction="sum")
        kl_divergence = -0.5 * torch.sum(1+ log_variance - mu.pow(2) - log_variance.exp())
        return cross_entropy + kl_divergence


class Cloud2VaeOptimizer(ModuleOptimizer):

    def __init__(
            self,
            starting_learning_rate= 0.001,
            lr_decay_steps=20,
            lr_decay_gamma=0.5):
        self._optimizer = None
        self._lr_scheduler = None
        self._starting_learning_rate = starting_learning_rate
        self._lr_decay_steps = lr_decay_steps
        self._lr_decay_gamma = lr_decay_gamma

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer

    @property
    def lr_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        return self._lr_scheduler

    def configure(self, module:torch.nn.Module) -> None:
        self._optimizer = optim.Adam(module.parameters(), lr=self._starting_learning_rate)
        self._lr_scheduler = optim.lr_scheduler.StepLR(self._optimizer, self._lr_decay_steps, gamma=self._lr_decay_gamma)