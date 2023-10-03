python demo.py --config-file /MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml \
--input /MaskDINO/mnt_datasets/coco/luggage_parts/images/L3dwLWNvbnRlbnQvdXBsb2Fkcy8yMDIxLzEwL3NhbXNvbml0ZS1PbW5pLVBDLUhhcmRzaWRlLTIwLUluY2gtT25lLVNpemVlLVNwaW5uZXItNTczeDEwMjQuanBn.jpg \
--output /MaskDINO/demo/output \
--confidence-threshold 0.7 \
--opts MODEL.WEIGHTS /MaskDINO/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth \
#/MaskDINO/mnt_datasets/coco/luggage_parts/images/L3dwLWNvbnRlbnQvdXBsb2Fkcy8yMDIxLzExLzA1LTA1LTIwMjItTEVWRUw4LUZ1bGwtQWx1bWludW0tQ2FycnktT24tMDA0LmpwZw.jpg \
# /MaskDINO/mnt_datasets/coco/luggage_parts/images/Ly5pbWFnZS90X3NoYXJlL01UVTBNakEwTXpjeU5UTTVNREkwTnpJdy9hd2F5X2Jjb19hbHVtXzM2MF8wMS5qcGc.jpg \