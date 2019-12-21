import os
import json
import random
from cumulus2_vae_model import *
from model_trainer import *
from PIL import Image
from torchvision.transforms.functional import to_tensor
from local_image_dataset import LocalImageDataset

class LocalSnapshotSaver(ModuleSnapshotSaver):
    def __init__(self, output_root, model_name) -> None:
        self.output_root = output_root
        self.model_name = model_name

    def should_save(self, progress: Progress) -> bool:
        if progress.epoch <= 10:
            return True
        elif progress.epoch <= 100 and progress.epoch % 10 == 0:
            return True
        elif progress.epoch % 50:
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



def standardize_image(
        file_path:str,
        min_short_side_length:int = 720,
        max_short_side_length:int = 1200,
        dimension_factor:int = 60
    ) -> Image:
    im = Image.open(file_path)
    im_size = im.size
    shortest_side = min(im_size)
    longest_side = max(im_size)
    aspect_ratio = longest_side / shortest_side
    target_short_side_length = 0
    target_long_side_length = 0
    if shortest_side < min_short_side_length:
        target_short_side_length = min_short_side_length
        target_long_side_length = math.floor(target_short_side_length * aspect_ratio)
        target_long_side_length = target_long_side_length - (target_long_side_length % dimension_factor)
    elif shortest_side > max_short_side_length:
        target_short_side_length = max_short_side_length
        target_long_side_length = math.floor(target_short_side_length * aspect_ratio)
        target_long_side_length = target_long_side_length - (target_long_side_length % dimension_factor)
    else:
        target_short_side_length = shortest_side - (shortest_side % dimension_factor)
        target_long_side_length = longest_side - (longest_side % dimension_factor)

    new_size = (target_short_side_length, target_long_side_length)
    if im_size[1] <= im_size[0]:
        new_size = (target_long_side_length, target_short_side_length)
    return im.resize(new_size, Image.BICUBIC)

def standardize_dataset(source_path:str, output_path:str) -> None:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for dir_path, _, file_names in os.walk(source_path):
        for file_name in file_names:
            full_path = os.path.join(dir_path, file_name)
            output_file = os.path.join(output_path, file_name)
            if not os.path.exists(output_file):
                im = standardize_image(full_path)
                im.save(output_file)


def main():
    output_root = os.path.join(os.path.dirname(__file__), "output")
    #print("Preparing dataset")
    #standardize_dataset("C:\\data\\clouds", "C:\\data\\clouds_standard")
    #print("Preparing dataset complete")
    trainer = ModelTrainer(
        Cloud2VaeFactory(3, 8),
        Cloud2VaeOptimizer(),
        Cloud2VaeLoss(),
        LocalImageDataset("C:\\data\\clouds_standard"),
        [LocalSnapshotSaver(output_root, "cumulus2_vae")])
    trainer.start(100)

if __name__ == "__main__":
    main()
    # TODO move datset to cloud storage
    # TODO better loss logging
    # TODO also render a sample from the test set