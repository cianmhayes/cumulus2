import os
import random
from cumulus2_vae_model import *
from model_trainer import *
from PIL import Image
from torchvision.transforms.functional import to_tensor
from local_image_dataset import LocalImageDataset
from uuid import uuid4
from torchvision.transforms import ToPILImage

class LocalSnapshotSaver(ModuleSnapshotSaver):
    def __init__(self, output_root, model_name) -> None:
        self.output_root = output_root
        self.model_name = model_name

    def should_save(self, progress: Progress) -> bool:
        if progress.epoch <= 10:
            return True
        elif progress.epoch <= 100 and progress.epoch % 10 == 0:
            return True
        elif progress.epoch % 50 == 0:
            return True
        else:
            return False

    def save(self, module: torch.nn.Module, module_parameters: Dict, progress: Progress) -> None:
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)
        model_path = os.path.join(self.output_root, "{}_e{}.pt".format(self.model_name, progress.epoch))
        torch.save(
            {
                "parameters" : module_parameters,
                "module_state": module.state_dict()
            },
            model_path)

    def save_sample(self, sample:torch.Tensor, progress: Progress) -> None:
        epoch_folder = os.path.join(            self.output_root,
            str(progress.epoch))
        if not os.path.exists(epoch_folder):
            os.makedirs(epoch_folder)
        output_path = os.path.join(
            epoch_folder,
            "{}.png".format(str(uuid4())))
        single_image = sample[0]
        im = ToPILImage()(single_image.to("cpu"))
        im.save(output_path)
