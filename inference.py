import os
import sys
import cv2
import json
import torch
import warnings
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.layers import batched_nms
from detectron2.engine import DefaultPredictor
from itertools import groupby
from pycocotools.coco import COCO
from pycocotools import mask as maskutil

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def binary_mask_to_rle(mask):
    """ Transform Masks to RLE (Do Compression) """
    rle = {'counts': [], 'size': list(mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    result = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    result['counts'] = str(result['counts'], encoding='utf-8')
    return result


if __name__ == "__main__":
    # Set Configs
    cfg = get_cfg()
    model = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
    modelfile = model_zoo.get_config_file(model)
    cfg.merge_from_file(modelfile)
    cfg.merge_from_file("config.yaml")
    cfg.MODEL.WEIGHTS = sys.argv[1]
    cfg.DATASETS.TEST = ("VOC_dataset",)

    # Define 2 pridictor
    # The first one is for larger object
    # And the second one is for smaller object
    cfg.INPUT.MAX_SIZE_TEST = 10000
    cfg.INPUT.MIN_SIZE_TEST = 400
    predictor1 = DefaultPredictor(cfg)

    cfg.INPUT.MIN_SIZE_TEST = 800
    predictor2 = DefaultPredictor(cfg)

    # Do prediction
    coco_test = COCO("test_images/test.json")
    prediction = []
    for imgid in coco_test.imgs:
        filename = coco_test.loadImgs(ids=imgid)[0]['file_name']
        print('predicting ' + filename)
        im = cv2.imread("test_images/" + filename)

        # Do prediction
        outputs1 = predictor1(im)
        outputs2 = predictor2(im)

        # Merge prediction
        boxes1 = outputs1['instances']._fields['pred_boxes'].tensor
        boxes2 = outputs2['instances']._fields['pred_boxes'].tensor
        scores1 = outputs1['instances']._fields['scores']
        scores2 = outputs2['instances']._fields['scores']
        classes1 = outputs1['instances']._fields['pred_classes']
        classes2 = outputs2['instances']._fields['pred_classes']
        masks1 = outputs1['instances']._fields['pred_masks']
        masks2 = outputs2['instances']._fields['pred_masks']

        boxes = torch.cat([boxes1, boxes2], dim=0).to('cpu')
        scores = torch.cat([scores1, scores2], dim=0).to('cpu')
        classes = torch.cat([classes1, classes2], dim=0).to('cpu')
        masks = torch.cat([masks1, masks2], dim=0).to('cpu')

        # Record predictions
        nms_idx = batched_nms(boxes, scores, classes, 0.7)
        for idx in nms_idx:
            pred = {}
            pred['image_id'] = imgid
            pred['category_id'] = int(classes[idx]) + 1
            pred['segmentation'] = binary_mask_to_rle(masks[idx].numpy())
            pred['score'] = float(scores[idx])
            prediction.append(pred)

    # Write json file
    with open("prediction.json", "w") as f:
        json.dump(prediction, f)
