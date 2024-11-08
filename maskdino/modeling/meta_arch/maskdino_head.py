# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------------
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

from torch import nn
import torch

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.maskdino_decoder import build_transformer_decoder
from ..pixel_decoder.maskdino_encoder import build_pixel_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class MaskDINOHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        transformer_predictor: nn.Module,
    ):
        """
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor

        self.num_classes = num_classes
        
        self.multiScaleViewClassifyHead = MultiScaleClassificationHead([self.pixel_decoder.conv_dim for i in range(self.pixel_decoder.total_num_feature_levels)],4)
        # print(self.multiScaleViewClassifyHead)
        # if self.training:
        #     for param in self.
        # print('init number level: ',self.pixel_decoder.total_num_feature_levels)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
        }

    def forward(self, features, mask=None,targets=None):
        return self.layers(features, mask,targets=targets)

    def layers(self, features, mask=None,targets=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features, mask)
        # print('debug shape : ',[f.mean(dim=(-2, -1)).shape for f in multi_scale_features])
        print('debug shape : ',[f.shape for f in multi_scale_features])
        view_logit = self.multiScaleViewClassifyHead(multi_scale_features)
        
        return view_logit
        # print('debug shape : ',[f.mean(dim=(-2, -1)).shape for f in multi_scale_features])
        # predictions = self.predictor(multi_scale_features, mask_features, mask, targets=targets)

        # return predictions,


class MultiScaleClassificationHead(nn.Module):
    def __init__(self, input_dims, num_classes):
        """
        Args:
            input_dims (list of int): List of dimensions for each scale in multi_scale_features.
            num_classes (int): Number of classes for classification.
        """
        super(MultiScaleClassificationHead, self).__init__()
        
        # Define linear layers for each scale to bring features to a common dimension
        self.scale_fcs = nn.ModuleList([
            nn.Linear(dim, 256) for dim in input_dims  # Adjust output dim (e.g., 256) if needed
        ])
        
        # Final classification layer
        self.classifier = nn.Linear(256 * len(input_dims), num_classes)

    def forward(self, multi_scale_features):
        """
        Args:
            multi_scale_features (list of torch.Tensor): List of feature maps from different scales.
        
        Returns:
            torch.Tensor: Class logits of shape (batch_size, num_classes).
        """
        # Process each scale feature through its corresponding FC layer and pool spatial dimensions
        processed_features = []
        for feature, fc_layer in zip(multi_scale_features, self.scale_fcs):
            # Global average pooling to reduce spatial dimensions
            pooled_feature = feature.mean(dim=(-2, -1))  # (batch_size, feature_dim)
            # Apply fully connected layer
            processed_feature = fc_layer(pooled_feature)
            processed_features.append(processed_feature)

        # Concatenate processed features from all scales
        combined_features = torch.cat(processed_features, dim=-1)  # (batch_size, 256 * len(input_dims))

        # Pass through the classifier to get final class logits
        class_logits = self.classifier(combined_features)
        return class_logits
