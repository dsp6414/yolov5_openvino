:'
Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
'

# Remove cached labels
rm  /home/thanh_nguyen/Datasets/COCO2017_Person/labels/*.cache

python train.py \
        --img 512 \
        --cfg ./models/yolov5xs.yaml \
        --batch 64 \
        --epochs 500 \
        --data ./data/coco_person.yaml \
        --single-cls \
        --weights '' \
        --device 0 \
        --multi-scale