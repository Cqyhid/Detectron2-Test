import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

img_path = "./images/train/"
json_path = "./instances_train.json"
# register the coco data for training
register_coco_instances("mydata", {}, json_path, img_path)
mydata_metadata = MetadataCatalog.get("mydata")
dataset_dicts = DatasetCatalog.get("mydata")

cfg = get_cfg()
# get online model
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("mydata", )
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00001
cfg.SOLVER.MAX_ITER = (
    300
)  # 300 iterations seems good enough, but you can certainly train longer

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 class, not include background

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
