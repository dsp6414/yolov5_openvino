
# export PYTHONPATH=$PYTHONPATH:$PWD
export PYTHONPATH="$PWD"
python conversion/export.py \
       --weights /Users/anhvu/Downloads/untitled_folder/best_best_m2_mix_person.onnx \
       --img-size 512 \
       --batch-size 1 \
       --out-dir weights/onnx

