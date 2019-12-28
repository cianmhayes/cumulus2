import os
import random
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
from depth_encoder import *
from cumulus2_vae_model import Cloud2VaeOptimizer, Cumulus2Vae
from model_trainer import *
from PIL import Image
from torchvision.transforms.functional import to_tensor
from local_image_dataset import LocalImageDataset
from uuid import uuid4
from torchvision.transforms import ToPILImage
from local_snapshot_saver import LocalSnapshotSaver
from typing import Dict, Iterable, NewType, Sequence, Tuple, Optional
from torchvision.transforms import ToTensor
from train_vae import standardize_image
from datetime import datetime

class DepthEncoderDataset(Dataset):
    def __init__(
            self,
            root_path:str,
            cloud_model:Cumulus2Vae,
            taret_device:torch.device,
            image_limit:Optional[int]=None) -> None:
        self.root_path = root_path
        self.colour_root = os.path.join(self.root_path, "colour")
        self.desat_root = os.path.join(self.root_path, "desat")
        self.source_files = []
        self.cloud_model = cloud_model
        self.device = taret_device
        for dir_path, _, file_names in os.walk(self.desat_root):
            for file_name in file_names:
                full_desat_path = os.path.join(dir_path, file_name)
                full_colour_path = os.path.join(self.colour_root, file_name)
                if not os.path.exists(full_desat_path):
                    raise Exception("Missing file".format(full_desat_path))
                if not os.path.exists(full_colour_path):
                    raise Exception("Missing file".format(full_colour_path))
                self.source_files.append({"colour": full_colour_path, "desat": full_desat_path})
        if image_limit:
            self.source_files = self.source_files[:image_limit]

    def __len__(self) -> int:
        return len(self.source_files)

    def __getitem__(self, idx:int) -> Iterable[torch.Tensor]:
        desat = Image.open(self.source_files[idx]["desat"])
        desat_tensor = ToTensor()(desat).to(self.device)
        im = Image.open(self.source_files[idx]["colour"])
        im_tensor = ToTensor()(im).to(self.device)
        mu, log_var = self.cloud_model.encode(torch.stack([im_tensor]))
        return [desat_tensor, mu[0], log_var[0]]

def prepare_dataset(source_root:str, output_root:str) -> Tuple[str, str]:
    colour_output_root = os.path.join(output_root, "colour")
    if not os.path.exists(colour_output_root):
        os.makedirs(colour_output_root)
    desat_output_root = os.path.join(output_root, "desat")
    if not os.path.exists(desat_output_root):
        os.makedirs(desat_output_root)
    for dir_path, _, file_names in os.walk(source_root):
        for file_name in file_names:
            full_path = os.path.join(dir_path, file_name)
            colour_output_path = os.path.join(colour_output_root, file_name)
            desat_output_path = os.path.join(desat_output_root, file_name)
            im = standardize_image(full_path)
            im.save(colour_output_path)
            desat = im.convert("L")
            new_size = (int(im.size[0]/4), int(im.size[1]/4))
            desat = desat.resize(new_size, Image.BICUBIC)
            desat.save(desat_output_path)
    return colour_output_root, desat_output_root


def load_cloud_model(path:str):
    print("Loading model from:", path)
    checkpoint = torch.load(path, map_location="cpu")
    model = Cumulus2Vae(3, 16)
    model.load_state_dict(checkpoint["module_state"])
    model.eval()
    print("Model loaded")
    return model

def main():
    output_root = os.path.join(os.path.dirname(__file__), "output")
    output_root = os.path.join(
        output_root,
        "depth_encoder",
        "{:%Y%m%d_%H%M%S}".format(datetime.now()))

    data_root = os.path.join(os.path.dirname(__file__), "data", "clouds_depth")
    cloud_model_path = os.path.join(os.path.dirname(__file__), "cumulus2_vae_e650.pt")

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(device_name)
    cloud_model = load_cloud_model(cloud_model_path)
    cloud_model = cloud_model.to(target_device)

    trainer = ModelTrainer(
        DepthEncoderFactory(16),
        Cloud2VaeOptimizer(),
        DepthEncoderLoss(cloud_model),
        DepthEncoderDataset(data_root, cloud_model, target_device),
        [LocalSnapshotSaver(output_root, "depth_encoder")],
        ProgressLogger(output_root),
        test_split=0.1)
    trainer.start(1000)

if __name__ == "__main__":
    main()
    #source = os.path.join(os.path.dirname(__file__), "data", "clouds")
    #dest = os.path.join(os.path.dirname(__file__), "data", "clouds_depth")
    #prepare_dataset(source, dest)s