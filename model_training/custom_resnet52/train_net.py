#!/usr/bin/env python3

"""
CartoonFace Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""

import os
from typing import ValuesView
import cv2
import resnet52

from LossEvalHook import LossEvalHook

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import detection_utils as utils

def get_board_dicts(imgdir, d):
    import json
    from detectron2.structures import BoxMode
    json_file = imgdir+"/datasets.json"
    with open(json_file) as f:
        dataset_dicts = json.load(f)
    for i in dataset_dicts:
        filename = i["file_name"]
        i["file_name"] = imgdir+"/"+filename 
        for j in i["annotations"]:
            j["bbox_mode"] = BoxMode.XYWH_ABS
            j["category_id"] = int(j["category_id"])
    if d == "train":
        return dataset_dicts
    elif d == "val":
        return dataset_dicts
    else:
        raise ValueError("Incorrect argument passed to function: get_board_dicts")

def register_datasets():
    from detectron2.data import DatasetCatalog, MetadataCatalog
    for d in ["train", "val"]:
        DatasetCatalog.register("cartoon_" + d, lambda d=d: get_board_dicts("datasets/CartoonDataset/" + d, d))
        MetadataCatalog.get("cartoon_" + d).set(thing_classes=["Face"])
    return


def build_img_train_aug(cfg):
    # Define the data augmentations to perform
    #mask = cv2.imread("mask.jpg")
    augs = [#T.RandomApply(T.RandomContrast(0.2, 5.0), prob=0.0),
            #T.RandomApply(T.RandomSaturation(0.0, 5.0), prob=0.0),
            #T.RandomApply(T.RandomBrightness(0.3, 2.0), prob=0.0),
            #T.RandomApply(T.RandomExtent((1, 4), (0,0)), prob=0.0),
            #T.RandomApply(T.RandomCrop("relative_range", ([0.3,0.3])), prob=0.0),
            #T.Resize((640, 640)),
            #T.RandomApply(T.BlendTransform(mask, 1.0, 1.0), prob=0.0),
            T.Resize((640, 640))]
    return augs


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
            os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, False, output_folder, use_fast_impl=False)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_img_train_aug(cfg))
        # mapper = None
        return build_detection_train_loader(cfg, mapper=mapper) # use this dataloader instead of default

    def build_hooks(self): # to track val loss as well as train loss
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # TODO: Remove this function and set the dataset in the config file once we have data in COCO format
    # Replace in config file like:
    #
    # DATASETS:
    #   TRAIN: ("coco_2017_train",)
    #   TEST: ("coco_2017_val",)
    register_datasets()

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg) # go in here to see how detectron2 works
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )