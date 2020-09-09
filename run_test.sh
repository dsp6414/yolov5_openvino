
# Remove cached labels
rm /Users/thanhnguyen/Documents/Datasets/COCO2017_Person/labels/*.cache

python test.py \
        --weights weights/org/best.pt \
        --data ./data/coco_person.yaml \
        --batch 1 \
        --img 512 \
        --task 'test' \
        --single-cls \
        --device 'cpu' \
        --verbose
