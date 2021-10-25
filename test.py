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
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

img_path = "./images/val/"
json_path = "./instances_val.json"

register_coco_instances("self_coco_val", {}, json_path, img_path)
mydata_metadata = MetadataCatalog.get("self_coco_val")
dataset_dicts = DatasetCatalog.get("self_coco_val")
print(mydata_metadata)
print(dataset_dicts)

# Missing some parameters here, solving...
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("self_coco_val",)
predictor = DefaultPredictor(cfg)

im = cv2.imread("./images/val/12.png")
outputs = predictor(im)
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)
