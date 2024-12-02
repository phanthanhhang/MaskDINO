from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
# from detectron2.projects.idol import add_idol_config
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config
from detectron2.modeling import build_model
from demo.predictor import VisualizationDemo
import cv2
import torch
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
import os
from torchvision import ops
import numpy as np

def setup_cfg(config_file, pretrained_weight):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = pretrained_weight
    cfg.freeze()
    return cfg

class MaskDinoModel():
    def __init__(self, config_file, pretrained_weight):
        config = setup_cfg(config_file, pretrained_weight)
        self.cfg = config.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(self.cfg.DATASETS.TEST):
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(self.cfg.MODEL.WEIGHTS)
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = self.cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
        
    def predictor(self, original_image,threshold=0.4):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs],threshold)[0]
            return predictions
    def detect(self, img_path, confidence_thres):
        torch.cuda.empty_cache()
        img = read_image(img_path)
        if img.shape[-1] == 3:
            predictions = self.predictor(img,confidence_thres)
            boxes, masks, scores, labels = convert_pred_results(predictions, confidence_thres)
            # print(len(boxes),len(masks),len(scores),len(labels))
            # print(boxes.shape)
            if len(boxes) == 0:
                return [],[],[],[]
            
            keep_indices = ops.nms(
                boxes=torch.tensor(boxes),
                scores=torch.tensor(scores),
                iou_threshold=0.5  # Adjust this threshold as needed
            ).cpu().numpy()
            
            # Filter results using NMS indices
            boxes = np.array(boxes)[keep_indices]
            masks = np.array(masks)[keep_indices]
            scores = np.array(scores)[keep_indices]
            labels = np.array(labels)[keep_indices]
            return boxes, masks, scores, labels
        return [],[],[],[]

    
def convert_pred_results(predictions, confidence_thres):
    instances = predictions['instances']
    pred_boxes = instances.pred_boxes.tensor.detach().cpu().numpy()    
    pred_masks = instances.pred_masks.detach().cpu().numpy()
    pred_scores = instances.scores.detach().cpu().numpy()
    pred_classes = instances.pred_classes.detach().cpu().numpy()
    
    boxes, masks, scores, labels = [], [], [], []
     
    for b,m,s,l in zip(pred_boxes, pred_masks, pred_scores, pred_classes):
        if s >= confidence_thres:
            boxes.append(b)
            masks.append(m)
            scores.append(s)
            labels.append(l)
            
    # print(scores)
    # print(labels)
    return boxes, masks, scores, labels

if __name__ == '__main__':
    img_dir = '/MaskDINO/mnt_datasets/coco/luggage_parts/images/'
    img_fns = os.listdir(img_dir)
    config_file = '/MaskDINO/configs/coco/instance-segmentation/luggage_parts.yaml'
    confidence_thres = 0.4
    pretrained = '/MaskDINO/runs/luggage_parts/model_final.pth'
    for fn in img_fns[1:2]:
        test_img_path = os.path.join(img_dir, fn)
        model = Load_model(config_file, pretrained)
        results = model.detect(test_img_path, confidence_thres)
        
