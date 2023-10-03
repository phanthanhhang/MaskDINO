CUDA_VISIBLE_DEVICES=0,1,2 python train_net.py \
--num-gpus 3 --config-file /MaskDINO/configs/coco/instance-segmentation/swin/luggage_parts.yaml \
MODEL.WEIGHTS /MaskDINO/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth \
DATASETS.TEST "('valid_1109_newbody',)" DATASETS.TRAIN "('train_1109_newbody',)" \
OUTPUT_DIR runs/luggage_parts_1109_newbody_swinL \
INPUT.IMAGE_SIZE 650 SOLVER.IMS_PER_BATCH 3 \
SOLVER.MAX_ITER 100000 SOLVER.STEPS [80000,89000,95000] TEST.EVAL_PERIOD 1000