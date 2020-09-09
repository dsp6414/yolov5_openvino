import argparse
import glob
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

import torch
from utils.datasets import create_dataloader
from utils.general import (
    ap_per_class,
    box_iou,
    check_file,
    check_img_size,
    clip_coords,
    coco80_to_coco91_class,
    compute_loss,
    non_max_suppression,
    output_to_target,
    plot_images,
    scale_coords,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.torch_utils import time_synchronized
from common import *


def test(
    data,
    batch_size=1,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.6,  # for NMS
    save_json=False,
    single_cls=True,
    augment=False,
    verbose=False,
    onnx=False,
    model=None,
    dataloader=None,
    save_dir="",
    merge=False,
    save_txt=False,
):
    merge, save_txt = opt.merge, opt.save_txt  # use Merge NMS, save *.txt labels
    if save_txt:
        out = Path("inference/output")
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Remove previous
    for f in glob.glob(str(Path(save_dir) / "test_batch*.jpg")):
        os.remove(f)

    # Load model
    if onnx:
        import onnx
        import onnxruntime

        session = onnxruntime.InferenceSession(opt.model_path)
        input_name = session.get_inputs()[0].name
    else:
        from openvino.inference_engine import IENetwork, IEPlugin

        model_bin = os.path.splitext(opt.model_xml)[0] + ".bin"
        time.sleep(1)
        net = IENetwork(model=opt.model_xml, weights=model_bin)
        input_blob = next(iter(net.inputs))
        plugin = IEPlugin(device=opt.device)
        exec_net = plugin.load(network=net)

    # Configure
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader

    path = (
        data["test"] if opt.task == "test" else data["val"]
    )  # path to val/test images
    dataloader, _ = create_dataloader(
        path,
        imgsz,
        batch_size,
        32,
        opt,
        hyp=None,
        augment=False,
        cache=False,
        pad=0.5,
        rect=False,
    )

    seen = 0
    names = class_names.copy()

    coco91class = coco80_to_coco91_class()
    s = ("%20s" + "%12s" * 6) % (
        "Class",
        "Images",
        "Targets",
        "P",
        "R",
        "mAP@.5",
        "mAP@.5:.95",
    )

    p, r, f1, mp, mr, map50, map, t0, t1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height])

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            if onnx:
                outputs = session.run(None, {input_name: img.numpy()})
                inf_out = outputs[0]
            else:
                outputs = exec_net.infer(inputs={input_blob: img})
                key = list(outputs.keys())[0]
                inf_out = outputs[key]
            t0 += time_synchronized() - t

            # Run NMS
            t = time_synchronized()

            output = non_max_suppression(
                inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge
            )

            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(
                        (
                            torch.zeros(0, niou, dtype=torch.bool),
                            torch.Tensor(),
                            torch.Tensor(),
                            tcls,
                        )
                    )
                continue

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[
                    [1, 0, 1, 0]
                ]  # normalization gain whwh
                txt_path = str(out / Path(paths[si]).stem)
                pred[:, :4] = scale_coords(
                    img[si].shape[1:], pred[:, :4], shapes[si][0], shapes[si][1]
                )  # to original
                for *xyxy, conf, cls in pred:
                    xywh = (
                        (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                        .view(-1)
                        .tolist()
                    )  # normalized xywh
                    with open(txt_path + ".txt", "a") as f:
                        f.write(("%g " * 5 + "\n") % (cls, *xywh))  # label format

            # Clip boxes to image bounds
            # print(pred)
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = Path(paths[si]).stem
                box = pred[:, :4].clone()  # xyxy
                scale_coords(
                    img[si].shape[1:], box, shapes[si][0], shapes[si][1]
                )  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append(
                        {
                            "image_id": int(image_id)
                            if image_id.isnumeric()
                            else image_id,
                            "category_id": coco91class[int(p[5])],
                            "bbox": [round(x, 3) for x in b],
                            "score": round(p[4], 5),
                        }
                    )

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (
                        (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    )  # prediction indices
                    pi = (
                        (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
                    )  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(
                            1
                        )  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if (
                                    len(detected) == nl
                                ):  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if batch_i < 1:
            f = Path(save_dir) / ("test_batch%g_gt.jpg" % batch_i)  # filename
            plot_images(img, targets, paths, str(f), names)  # ground truth
            f = Path(save_dir) / ("test_batch%g_pred.jpg" % batch_i)
            plot_images(
                img, output_to_target(output, width, height), paths, str(f), names
            )  # predictions

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = (
            p[:, 0],
            r[:, 0],
            ap[:, 0],
            ap.mean(1),
        )  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(
            stats[3].astype(np.int64), minlength=nc
        )  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = "%20s" + "%12.3g" * 6  # print format
    print(pf % ("all", seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1e3 for x in (t0, t1, t0 + t1)) + (
        imgsz,
        imgsz,
        batch_size,
    )  # tuple

    # Return results

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map), maps, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="eval.py")
    parser.add_argument(
        "--onnx", action="store_true", help="test openvino or onnx model"
    )
    parser.add_argument(
        "--model-xml",
        type=str,
        default="weights/openvino/best.xml",
        help="model.xml path",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/onnx/best.onnx",
        help="model.onnx path",
    )
    parser.add_argument(
        "--data", type=str, default="data/coco_person.yaml", help="*.data path"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="size of each image batch"
    )
    parser.add_argument(
        "--img-size", type=int, default=512, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.001, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.65, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="save a cocoapi-compatible JSON results file",
    )
    parser.add_argument("--task", default="val", help="'val', 'test'")
    parser.add_argument(
        "--device", default="CPU", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--single-cls", action="store_true", help="treat as single-class dataset"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--merge", action="store_true", help="use Merge NMS")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")

    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.data = check_file(opt.data)  # check file
    print(opt)

    test(
        data=opt.data,
        batch_size=opt.batch_size,
        imgsz=opt.img_size,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        save_json=opt.save_json,
        single_cls=opt.single_cls,
        augment=opt.augment,
        verbose=opt.verbose,
        onnx=opt.onnx,
    )
