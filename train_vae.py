import os
import random
from cumulus2_vae_model import *
from model_trainer import *
from PIL import Image
from torchvision.transforms.functional import to_tensor
from local_image_dataset import LocalImageDataset
from uuid import uuid4
from torchvision.transforms import ToPILImage
from local_snapshot_saver import LocalSnapshotSaver

random.seed(20190629)


def standardize_image(
        file_path:str,
        min_short_side_length:int = 960,
        max_short_side_length:int = 1920,
        dimension_factor:int = 64 ) -> Image:
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
    return im.resize(new_size, Image.BICUBIC).convert("RGB")

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

def prep_dataset():
    print("Preparing dataset")
    standardize_dataset("C:\\data\\clouds", "C:\\data\\clouds_standard")
    print("Preparing dataset complete")

def main():
    output_root = os.path.join(os.path.dirname(__file__), "output")
    data_root = os.path.join(os.path.dirname(__file__), "clouds_standard")
    trainer = ModelTrainer(
        Cloud2VaeFactory(3, 16),
        Cloud2VaeOptimizer(),
        Cloud2VaeLoss(),
        LocalImageDataset(data_root),
        [LocalSnapshotSaver(output_root, "cumulus2_vae")],
        ProgressLogger(output_root),
        test_split=0.1)
    trainer.start(1000)

if __name__ == "__main__":
    #prep_dataset()
    main()