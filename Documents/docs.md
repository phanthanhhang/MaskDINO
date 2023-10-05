# Installation
## Setup git folder
- Clone repo:
`git clone https://github.com/phanthanhhang/MaskDINO.git`
- Checkout to the model-tuning branch:
`git checkout MCAI-695_tune_model_for_luggage_parts`

## Docker setup
- Build a docker image. In training machine:
`docker build -t maskdino_image -f docker/`
- Run a docker container
`nvidia-docker run --name maskdino_hang -it -v /mnt/ssd1/hang/MaskDINO:/MaskDINO -v /data2/hang/datasets:/MaskDINO/total_datasets -v    /mnt/ssd1/hang/datasets:/MaskDINO/mnt_datasets --shm-size=64g -p 3200:3200 -p 3201:3201 maskdino_image`
- Install detectron2: Inside the `/MaskDINO` dir:
  - Clone detectron2 repo: `https://github.com/phanthanhhang/detectron2.git`
  - Install : `python -m pip install -e detectron2`
- Install requirements
`pip install -r requirements.txt`

- CUDA kernel for MSDeformAttn: After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:
  CUDA_HOME must be defined and points to the directory of the installed CUDA toolkit.
```
    cd maskdino/modeling/pixel_decoder/ops
    sh make.sh
```

# Get started
- To train luggage parts segmentation model
    - Step1: Prepare valid, train, test according to coco format
    - Step2:Prepare a config file including structure model, training schedule, dataset name.
      Example: [resnet50](https://github.com/phanthanhhang/MaskDINO/blob/MCAI-695_tune_model_for_luggage_parts/configs/coco/instance-segmentation/luggage_parts.yaml), [swinL](https://github.com/phanthanhhang/MaskDINO/blob/MCAI-695_tune_model_for_luggage_parts/configs/coco/instance-segmentation/swin/luggage_parts.yaml)
    - Step3: register prepared dataset name in train_net.py file in setup(args) function (Line 332)
      ```
      register_coco_instances('dataset_name', {},'dataset_name.json', 'images_dir')
      cfg.DATASETS.TRAIN = ('dataset_name',)
      ```
    - Step4: run train_net.sh file (including training command)
  Adjust augmentation method at: [dataset_mapper](https://github.com/phanthanhhang/MaskDINO/blob/MCAI-695_tune_model_for_luggage_parts/maskdino/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py) 

  
  
  

  