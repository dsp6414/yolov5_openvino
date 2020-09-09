python /opt/intel/openvino_2020.4.287/deployment_tools/model_optimizer/mo_onnx.py \
--input_model /Users/anhvu/Downloads/untitled_folder/best_cont_m2_yolov5m.onnx \
--output_dir /Users/anhvu/Downloads/untitled_folder/ \
--input_shape="[1,3,512,512]" \
--log_level DEBUG --data_type FP16

# --mean_values="[128, 128, 128]" \
# --scale_values="[255, 255, 255]" \
