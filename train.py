import os
import copy
import torch
import warnings
from detectron2.engine import DefaultTrainer, default_argument_parser, launch
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import transforms as T
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper
from detectron2 import model_zoo

# Set Environment
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
num_gpus = 3
batch_size = 2 * num_gpus

# Initialize args and data
args = default_argument_parser().parse_args()
setup_logger()
train_path = "train_images/"
json_file = "train_images/pascal_train.json"
register_coco_instances("VOC_dataset", {}, json_file, train_path)


class Trainer(DefaultTrainer):
    ''' custom Trainer '''
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=mapper)


def mapper(dataset_dict):
    ''' Build a Mapper of Data Augmentation '''
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    # Transformations I use
    rApply = T.RandomApply
    rBright = T.RandomBrightness
    rCrop = T.RandomCrop_CategoryAreaConstraint
    rFlip = T.RandomFlip
    rContrast = T.RandomContrast
    rSaturation = T.RandomSaturation
    rRotate = T.RandomRotation
    Resize = T.ResizeShortestEdge

    # My Data Augmentation
    image, transforms = T.apply_transform_gens([
        rApply(rCrop(crop_type="relative_range", crop_size=(0.5, 0.5)),
               prob=0.50),
        rFlip(prob=0.50, horizontal=True, vertical=False),
        rApply(rBright(intensity_min=0.75, intensity_max=1.25), prob=0.20),
        rApply(rContrast(intensity_min=0.75, intensity_max=1.25), prob=0.20),
        rApply(rSaturation(intensity_min=0.75, intensity_max=1.25), prob=0.20),
        Resize((300, 500), 750, "range"),
        rApply(rRotate(angle=[-30, 30], expand=False, center=None,
                       sample_style="range", interp=None),
               prob=0.20),
    ], image)

    CHWimage = image.transpose(2, 0, 1).astype("float32")
    dataset_dict["image"] = torch.as_tensor(CHWimage)

    # transform instance annotations
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


def main():
    """ main function """
    cfg = get_cfg()
    model = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
    modelfile = model_zoo.get_config_file(model)
    cfg.merge_from_file(modelfile)
    cfg.merge_from_file('config.yaml')
    cfg.DATASETS.TRAIN = ("VOC_dataset",)
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # Initialize trainer and train
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    return trainer.train()


if __name__ == "__main__":
    # Train with {num_gpus} GPUs
    launch(
        main,
        num_gpus_per_machine=num_gpus,
        num_machines=1,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(),
    )
