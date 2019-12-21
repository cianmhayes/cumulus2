from typing import Dict, Generic, Sequence, Iterable, TypeVar
from abc import ABC, abstractmethod

import torch
import torch.nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


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
    def get_loss(self, sample:Sequence[torch.Tensor], module:torch.nn.Module) -> torch.Tensor:
        raise NotImplementedError()


class ModelTrainer(object):
    def __init__(
            self,
            factory:ModuleFactory,
            optimizer:ModuleOptimizer,
            loss_function:LossCalculator,
            dataset:torch.utils.data.Dataset,
            snapshot_savers:Sequence[ModuleSnapshotSaver],
            gradient_clip=None,
            force_cpu=False) -> None:
        self.module_factory = factory
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.dataset = dataset
        self.snapshot_savers = snapshot_savers
        self.gradient_clip = gradient_clip

        device_name = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        self.device = torch.device(device_name)

        self.dataset.set_device(self.device)
        self.module = self.module_factory.create_instance()
        self.module.to(self.device)
        self.optimizer.configure(self.module)
        self.progress = Progress()

    def start(self, epochs:int) -> None:
        for _ in range(epochs):
            self.progress.start_epoch()
            print("==================================================")
            print("Start Epoch {}".format(self.progress.epoch))
            print("==================================================")
            self._train()
            self._test()
            self._save()
            self.optimizer.step_lr_schedule(self.progress.epoch)

    def _train(self) -> None:
        self.module.train()
        training_loss = 0.0
        count = 0
        for n, training_sample in enumerate(self.dataset.train_set):
            self.optimizer.zero_grad()
            loss = self.loss_function.get_loss(training_sample, self.module)
            loss.backward()
            if self.gradient_clip is not None:
                # https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191/16
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.gradient_clip)
            training_loss += float(loss)
            count += n
            self.optimizer.step()
        training_loss /= count
        print('Train set loss: {:.4f}'.format(training_loss))

    def _test(self) -> None:
        self.module.eval()
        with torch.no_grad():
            test_loss = 0.0
            count = 0
            for n, test_sample in enumerate(self.dataset.test_set):
                self.optimizer.zero_grad()
                loss = self.loss_function.get_loss(test_sample, self.module)
                test_loss += float(loss)
                count += n
            test_loss /= count
            print('Test set loss: {:.4f}'.format(test_loss))

    def _save(self) -> None:
        paramaters = self.module_factory.get_construction_parameters()
        for saver in self.snapshot_savers:
            if saver.should_save(self.progress):
                saver.save(self.module, paramaters, self.progress)