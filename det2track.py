import json
import os
import argparse
from yolox.tracker.byte_tracker import BYTETracker
from yolox.exp import get_exp
import cv2
from tqdm import tqdm
import numpy as np


# TODO : Change Dirs
exp = get_exp('exps/example/mot/yolox_m_mix_det.py', None)
target = 'test'

# Test Inference result
infer = ''
datas = json.load(open(infer, 'r'))

# Test Annotation MSCOCO Format
test = ''
test_anno = json.load(open(test, 'r'))

result_dir = ''

def zero_division(a, b):
   return ( a / b ) if b != 0 else a

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
   #  parser.add_argument(
   #      "demo", default="image", help="demo type, eg. image, video and webcam"
   #  )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
   #  parser.add_argument(
   #      "-f",
   #      "--exp_file",
   #      default=None,
   #      type=str,
   #      help="pls input your expriment description file",
   #  )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

args = make_parser().parse_args()
id2instance = {}

for data in datas:
   if data['image_id'] in id2instance:
      id2instance[data['image_id']].append(data)
   else:
      id2instance[data['image_id']] = [data]

img_ids = sorted(list(id2instance.keys()))
tracker = BYTETracker(args, frame_rate=10)

total_results = {}
real_results = {}
for img_id in tqdm(img_ids):
   results = []
   for img in test_anno['images']:
      if img['id'] == img_id:
         anno = img
         break
   width = anno['width']
   height = anno['height']
   real_name = anno['file_name']
   frame_id = anno['frame_id']

   dets = []
   for instance in id2instance[img_id]:
      bbox = instance['bbox']
      new_object = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
      new_object.append(instance['score'])

      new_object.append(instance['category_id'])

      dets.append(new_object)
   
   dets_np = np.array(dets)
   online_targets = tracker.update(dets_np, [800, 800], exp.test_size)

   online_tlwhs = []
   online_ids = []
   online_scores = []
   candidates = []

   sum = 0
   for t in online_targets:
      tlwh = t.tlwh
      tid = t.track_id
      vertical = tlwh[2] / tlwh[3] > 1.6
      # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
      # if tlwh[2] * tlwh[3] > args.min_box_area:
      online_tlwhs.append(tlwh)
      online_ids.append(tid)
      online_scores.append(t.score)

      sum+=1
      found= 0
      candidate_category = 2

      # Find Possibly different candidates
      for item in dets:
         item_h = item[3] - item[1]
         item_w = item[2] - item[0]
         if (int(zero_division(tlwh[3], tlwh[1])) == int(zero_division(item_h,item[1]))) or \
         (int(zero_division(tlwh[2], tlwh[0])) == int(zero_division(item_w,item[0]))) \
            or (int(zero_division(tlwh[0], tlwh[1])) == int(zero_division(item[0],item[1]))) \
               or (int(zero_division(tlwh[3], tlwh[2])) == int(zero_division(item_h,item_w))) :
            candidates.append(item)
            candidate_category = item[5]
            break
      
      results.append(
            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},{candidate_category},-1,-1,-1\n"
      )
         
   if sum != len(candidates):
      print(sum, len(candidates))
   
      
   total_results[img_id] = results
   real_results[real_name] = results

json.dump(total_results, open(os.path.join(result_dir, target, 'total_result.json'), 'w'))
json.dump(real_results, open(os.path.join(result_dir, target, 'real_result.json'), 'w'))