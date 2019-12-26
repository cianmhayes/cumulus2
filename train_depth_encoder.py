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

class DepthEncoderDataset(Dataset):
    def __init__(
            self,
            root_path:str,
            cloud_model:Cumulus2Vae,
            image_limit:Optional[int]=None) -> None:
        self.root_path = root_path
        self.source_files = []
        for dir_path, _, file_names in os.walk(self.root_path):
            for file_name in file_names:
                full_path = os.path.join(dir_path, file_name)
                self.source_files.append(full_path)
        if image_limit:
            self.source_files = self.source_files[:image_limit]
        self.desaturated_images = []
        self.target_mu = []
        self.target_log_var = []
        for path in self.source_files:
            im = Image.open(path)
            desat = im.convert("L")
            im_tensor = ToTensor()(im)
            mu, log_var = cloud_model.encode(im_tensor)
            self.desaturated_images.append(desat)
            self.target_mu.append(mu)
            self.target_log_var.append(log_var)

    def __len__(self) -> int:
        return len(self.desaturated_images)

    def __getitem__(self, idx:int) -> Iterable[torch.Tensor]:
        return [
            self.desaturated_images[idx],
            self.target_mu[idx],
            self.target_log_var[idx]]


def load_cloud_model(path:str):
    checkpoint = torch.load(path, map_location="cpu")
    model = Cumulus2Vae(3, 16)
    model.load_state_dict(checkpoint["module_state"])
    model.eval()
    return model

def main():
    output_root = os.path.join(os.path.dirname(__file__), "output")
    data_root = os.path.join(os.path.dirname(__file__), "clouds_standard")
    cloud_model_path = os.path.join(os.path.dirname(__file__), "cumulus2_vae_e650.pt")

    cloud_model = load_cloud_model(cloud_model_path)

    trainer = ModelTrainer(
        DepthEncoderFactory(16),
        Cloud2VaeOptimizer(),
        DepthEncoderLoss(cloud_model),
        DepthEncoderDataset(data_root, cloud_model),
        [LocalSnapshotSaver(output_root, "depth_encoder")],
        ProgressLogger(output_root),
        test_split=0.1)
    trainer.start(1000)

if __name__ == "__main__":
    main()