The dummy.yaml file is for quick testing annd is just the config file resulting from the standard Colab Detectron2 tutorial:

    from detectron2.engine import DefaultTrainer
    cfg = get_cfg()    # obtain detectron2's default config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) # https://github.com/facebookresearch/detectron2/tree/main/configs/COCO-Detection
    cfg.DATASETS.TRAIN = ("cartoon_train",)
    cfg.DATASETS.TEST = ("cartoon_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2 # (default: 2)
    cfg.SOLVER.BASE_LR = 0.00025   # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (face). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

    cfg.TEST.EVAL_PERIOD = 100  # After how many iterations to run the evaluation (COCO)