
python detect.py \
        --weights ./weights/best_new.pt \
        --conf 0.3 \
        --iou-thres 0.4 \
        --source inference/images \
        --device cpu \
        # --view-img

