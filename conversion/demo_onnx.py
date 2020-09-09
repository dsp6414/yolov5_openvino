import onnx
import onnxruntime

from common import *


def build_argparser():
    parser = ArgumentParser(prog="demo_onnx.py")
    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/onnx/best_xs.onnx",
        help="model.xml path",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="inference/images/zidane.jpg",
        help="image path",
    )
    parser.add_argument(
        "--img-size", type=int, default=320, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.3, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.4, help="IOU threshold for NMS"
    )
    parser.add_argument("--cv", action="store_true", help="use PIL or opencv image")

    return parser


def main(args):
    session = onnxruntime.InferenceSession(args.model_path)

    if args.cv:
        image_src = cv2.imread(args.image_path).astype(np.float32)
        img_in = preprocess_cv_img(image_src, args.img_size, args.img_size)
    else:
        image_src = Image.open(args.image_path)
        img_in = preprocess_pil_img(image_src, args.img_size)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_in})

    detections = non_max_suppression(
        outputs[0], conf_thres=args.conf_thres, iou_thres=args.iou_thres, agnostic=False
    )

    return detections


if __name__ == "__main__":
    args = build_argparser().parse_args()

    with torch.no_grad():
        detections = main(args)
        if detections[0] is not None:
            display(
                detections[0],
                args.image_path,
                input_size=args.img_size,
                text_bg_alpha=0.6,
            )
