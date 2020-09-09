from common import *
from openvino.inference_engine import IENetwork, IEPlugin


def build_argparser():
    parser = ArgumentParser(prog="demo_openvino.py")
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified (CPU by default)",
    )
    parser.add_argument(
        "--model-xml",
        type=str,
        default="weights/openvino/best_xs.xml",
        help="model.xml path",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="inference/images/zidane.jpg",
        help="image path",
    )
    parser.add_argument(
        "--img-size", type=int, default=512, help="inference size (pixels)"
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
    model_bin = os.path.splitext(args.model_xml)[0] + ".bin"

    time.sleep(1)
    net = IENetwork(model=args.model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    plugin = IEPlugin(device=args.device)
    exec_net = plugin.load(network=net)

    start = time.time()

    if args.cv:
        image_src = cv2.imread(args.image_path).astype(np.float32)
        img_in = preprocess_cv_img(image_src, args.img_size, args.img_size)
    else:
        image_src = Image.open(args.image_path)
        img_in = preprocess_pil_img(image_src, args.img_size)

    outputs = exec_net.infer(inputs={input_blob: img_in})
    key = list(outputs.keys())[0]
    output = outputs[key]

    detections = non_max_suppression(
        output, conf_thres=args.conf_thres, iou_thres=args.iou_thres, agnostic=False
    )

    end = time.time()
    print(f"Processing time: {(end - start)}")

    if detections[0] is not None:
        display(
            detections[0], args.image_path, input_size=args.img_size, text_bg_alpha=0.6
        )

    del net
    del exec_net
    del plugin


if __name__ == "__main__":
    args = build_argparser().parse_args()

    sys.exit(main(args) or 0)
