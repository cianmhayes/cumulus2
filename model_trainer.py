from typing import Dict, Generic, Sequence, Iterable, TypeVar
from abc import ABC, abstractmethod

import torch
import torch.nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

class ModuleFactory(ABC):

    @abstractmethod
    def create_instance(self) -> torch.nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def get_construction_parameters(self) -> Dict:
        raise NotImplementedError()


class Progress(object):
    def __init__(self) -> None:
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def start_epoch(self) -> None:
        self._epoch += 1


class ModuleSnapshotSaver(ABC):

    @abstractmethod
    def should_save(self, progress: Progress) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def save(self, module: torch.nn.Module, module_parameters: Dict, progress: Progress) -> None:
        raise NotImplementedError()

    @abstractmethod
    def save_sample(self, sample:torch.Tensor, progress: Progress) -> None:
        raise NotImplementedError()


class ProgressLogger(object):
    def __init__(self, output_path) -> None:
        self.current_epoch_training_loss = 0.0
        self.current_epoch_test_loss = 0.0
        self.current_epoch_training_count = 0
        self.current_epoch_testsing_count = 0
        self.current_epoch = 0
        self.current_epoch_start_time = time.clock()
        self.tb_writer = SummaryWriter(output_path)

    def start_epoch(self) -> None:
        self.current_epoch += 1
        print("==================================================")
        print("Starting epoch ", self.current_epoch)
        print("==================================================")
        self.current_epoch_training_loss = 0.0
        self.current_epoch_test_loss = 0.0
        self.current_epoch_training_count = 0
        self.current_epoch_test_count = 0
        self.current_epoch_start_time = time.clock()

    def log_training_loss(self, loss:float, batch_size:int) -> None:
        self.current_epoch_training_loss += loss
        self.current_epoch_training_count += batch_size

    def log_test_loss(self, loss:float, batch_size:int) -> None:
        self.current_epoch_test_loss += loss
        self.current_epoch_test_count += batch_size

    def end_epoch(self) -> None:
        duration = time.clock() - self.current_epoch_start_time
        training_loss = self.current_epoch_training_loss / self.current_epoch_training_count
        test_loss = self.current_epoch_test_loss / self.current_epoch_test_count
        print('Duration: {:.2f} minutes'.format(duration / 60))
        print('Train loss: {:.4f}'.format(training_loss))
        print('Test loss: {:.4f}'.format(test_loss))
        self.tb_writer.add_scalar("train_loss", training_loss, self.current_epoch)
        self.tb_writer.add_scalar("test_loss", test_loss, self.current_epoch)
        

class ModuleOptimizer(ABC):

    @property
    @abstractmethod
    def optimizer(self) -> Optimizer:
        raise NotImplementedError()

    @property
    @abstractmethod
    def lr_scheduler(self) -> _LRScheduler:
        raise NotImplementedError()

    @abstractmethod
    def configure(self, module:torch.nn.Module) -> None:
        raise NotImplementedError()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()

    def step_lr_schedule(self, epoch:int) -> None:
        self.lr_scheduler.step(epoch)


class LossCalculator(ABC):

    @abstractmethod
    def get_loss(
            self,
            sample:Sequence[torch.Tensor],
            module:torch.nn.Module,
            snapshot_savers:Sequence[ModuleSnapshotSaver] = None,
            progress:Progress = None) -> torch.Tensor:
        raise NotImplementedError()


class ModelTrainer(object):
    def __init__(
            self,
            factory:ModuleFactory,
            optimizer:ModuleOptimizer,
            loss_function:LossCalculator,
            dataset:torch.utils.data.Dataset,
            snapshot_savers:Sequence[ModuleSnapshotSaver],
            progress_logger:ProgressLogger,
            test_split=0.2,
            batch_size=1,
            gradient_clip=None,
            force_cpu=False) -> None:
        self.module_factory = factory
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.dataset = dataset
        self.snapshot_savers = snapshot_savers
        self.gradient_clip = gradient_clip
        self.batch_size = batch_size
        self.test_split = test_split

        self._configure_data_loaders(test_split,batch_size)

        device_name = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        self.device = torch.device(device_name)

        self.module = self.module_factory.create_instance()
        self.module.to(self.device)
        self.optimizer.configure(self.module)
        self.progress = Progress()
        self.progress_logger = progress_logger

    def start(self, epochs:int) -> None:
        for _ in range(epochs):
            self.progress.start_epoch()
            self.progress_logger.start_epoch()
            self._train()
            self._test()
            self._save()
            self.optimizer.step_lr_schedule(self.progress.epoch)
            self.progress_logger.end_epoch()

    def _configure_data_loaders(self, test_split:float, batch_size:int) -> None:
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(test_split * dataset_size)
        np.random.shuffle(indices)
        train_indices = indices[split:]
        test_indices = indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        self.training_data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=train_sampler)
        self.test_data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=test_sampler)

    def _train(self) -> None:
        self.module.train()
        for _, training_sample in enumerate(self.training_data_loader):
            self.optimizer.zero_grad()
            loss = self.loss_function.get_loss(training_sample, self.module)
            loss.backward()
            if self.gradient_clip is not None:
                # https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191/16
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.gradient_clip)
            self.progress_logger.log_training_loss(float(loss), self.batch_size)
            self.optimizer.step()

    def _test(self) -> None:
        self.module.eval()
        with torch.no_grad():
            for _, test_sample in enumerate(self.test_data_loader):
                self.optimizer.zero_grad()
                loss = self.loss_function.get_loss(test_sample, self.module, self.snapshot_savers, self.progress)
                self.progress_logger.log_test_loss(float(loss), self.batch_size)

    def _save(self) -> None:
        paramaters = self.module_factory.get_construction_parameters()
        for saver in self.snapshot_savers:
            if saver.should_save(self.progress):
                saver.save(self.module, paramaters, self.progress)