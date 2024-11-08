from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
# from detectron2.projects.idol import add_idol_config
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config
from detectron2.modeling import build_model
from demo.predictor import VisualizationDemo
import cv2
import torch
import torch.nn as nn
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, detection_utils as utils
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm

# import warnings

# Treat all warnings as errors
# warnings.filterwarnings("error")


def setup_cfg(config_file, pretrained_weight):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = pretrained_weight
    cfg.freeze()
    return cfg


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    # logger = logging.getLogger(__name__)

    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip(horizontal=True, vertical=False))
        # tfm_gens.append(T.RandomFlip(horizontal=False, vertical=True))
        tfm_gens.append(T.RandomRotation(angle=[-30, 30]))
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    # if is_train:
    print("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class MissingViewDataset(Dataset):
    def __init__(self, csv_file,cfg, is_train=False):
        # used to prepare the labels and images path
        self.data = pd.read_csv(csv_file)
        self.transform = build_transform_gen(cfg,is_train)

    def __getitem__(self, index):
        file_name = self.data.iloc[index]['file_name']
        label = self.data.iloc[index]['label']

        raw_image = utils.read_image(file_name,format='RGB')
        
        # Apply image transformations
        if self.transform is not None:
            raw_image, _ = T.apply_transform_gens(self.transform,raw_image)
            raw_image = torch.as_tensor(np.ascontiguousarray(raw_image.transpose(2, 0, 1)))

        return raw_image, torch.tensor(label)

    def __len__(self):
        return len(self.data)

def custom_collate_fn(batch):
    # Separate the pairs of images and the classes
    image1_list = []
    labels = []

    for image1, label in batch:
        image1_list.append({'image':image1})
        labels.append(label)

    # Convert labels to tensor
    labels_tensor = torch.tensor(labels)

    return image1_list, labels_tensor

config_file = 'checkpoint/luggage_part/config.yaml'
pretrained = 'checkpoint/luggage_part/model_0094999.pth'
config = setup_cfg(config_file, pretrained)

batch_size = 3
train_dataset = MissingViewDataset('missing_view_train.csv',config,True)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=custom_collate_fn,num_workers=10)

valid_dataset = MissingViewDataset('missing_view_valid.csv',config,False)
valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,collate_fn=custom_collate_fn,num_workers=10)

# print(tfm_gens)
model = build_model(config)
# model.train()
device = model.device
for k,v in model.named_parameters():
    if 'multiScaleViewClassifyHead' in k:
        v.requires_grad = True
    else:
        v.requires_grad = False

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([p for name, p in model.named_parameters() if p.requires_grad], lr=1e-3, weight_decay=0.0005)

def train_1_epoch(model,loss_func,optimizer,dataloader,epoch, epochs,device):
    model.train()

    print(('\n' + '%10s   ' * 3) % ('Epoch', 'total_loss', 'targets'))
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar,total=len(dataloader))

    train_loss = 0
    for batch_id,(raw_images_list,labels) in pbar:
        optimizer.zero_grad()

        labels = labels.to(device)

        out_raw  =  model(raw_images_list)

        loss = loss_func(out_raw,labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        s = f'       {epoch}/{epochs}  {train_loss/(batch_id+1):4.4f}             {labels.shape[0]}'
        pbar.set_description(s)
        ckpt = {'model':model.state_dict()}
        torch.save(ckpt,'maskdino_swinL_view_head_last.pth')

def test(model,dataloader,device):
    model.eval()
    total= 0
    correct = 0
    with torch.no_grad():
        print(('\n' + '%10s   ' * 3) % ('acc', 'correct', 'total'))
        pbar = enumerate(dataloader)
        pbar = tqdm(pbar,total=len(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            s = f'      {correct/total: 4.4f}  (    {correct}  /      {total}) '
            pbar.set_description(s)

    return correct / total

# train_1_epoch(model,loss_fn,optimizer,train_dataloader,0,100,device)

epochs = 100
best_acc = 0
for epoch in range(epochs):
    train_1_epoch(model,loss_fn,optimizer,train_dataloader,epoch,epochs,device)
    acc = test(model,valid_dataloader,device)
    if acc > best_acc:
        ckpt = {'model':model.state_dict()}
        torch.save(ckpt,'maskdino_swinL_view_head_best.pth')