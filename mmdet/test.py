import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from natsort import natsorted

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from mmdet.datasets import build_dataset, build_dataloader

from base_config import get_config
from mmdet_tools import test as mmdet_test

from train import get_training_datasets

_classes = ("Aortic_enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD", "Infiltration", "Lung_Opacity", "Nodule/Mass", "Other_lesion", "Pleural_effusion", "Pleural_thickening", "Pneumothorax", "Pulmonary_fibrosis")

def draw_bbox(image, box, label, color, thickness=3):
    alpha = 0.1
    alpha_box = 0.4
    overlay_bbox = image.copy()
    overlay_text = image.copy()
    output = image.copy()

    text_width, text_height = cv.getTextSize(label.upper(), cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv.rectangle(overlay_bbox, (box[0], box[1]), (box[2], box[3]),
                color, -1)
    cv.addWeighted(overlay_bbox, alpha, output, 1 - alpha, 0, output)
    cv.rectangle(overlay_text, (box[0], box[1]-7-text_height), (box[0]+text_width+2, box[1]),
                (0, 0, 0), -1)
    cv.addWeighted(overlay_text, alpha_box, output, 1 - alpha_box, 0, output)
    cv.rectangle(output, (box[0], box[1]), (box[2], box[3]),
                    color, thickness)
    cv.putText(output, label.upper(), (box[0], box[1]-5),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return output

def visualize(checkpoint):
  checkpoint_file = checkpoint

  cfg = get_config()
  model = init_detector(cfg, checkpoint_file, device='cuda:0')
  model.CLASSES = _classes

  os.makedirs('temp', exist_ok=True)

  test_dataset = build_dataset(cfg.data.test)
  for i in range(len(test_dataset)):
    annotation = test_dataset.get_ann_info(i)
    img = 'data/' + annotation['seg_map']
    result = inference_detector(model, img)
    # visualize the results in a new window
    model.show_result(img, result, score_thr=0.0, out_file='temp.png')
    temp = cv.imread('temp.png')
    for bbox_idx in range(len(annotation['bboxes'])):
      box = np.int_(annotation['bboxes'][bbox_idx])
      temp = draw_bbox(temp, list(box), _classes[annotation['labels'][bbox_idx]], (255, 0, 0))
    plt.imshow(temp)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig('temp/' + str(i) + '.png')
    plt.show()

def dataset_stats(cfg, checkpoint):
  pretrain, train = get_training_datasets(0.5)
  test = build_dataset(cfg.data.test)
  datasets = [pretrain, train, test]
  titles = ['pretrain', 'train', 'test']

  for i in range(len(datasets)):
    dataset = datasets[i]
    labels = []
    for data in dataset:
      if dataset == test:
        labels += np.array(data['gt_labels']).flatten().tolist()
      else:
        labels += data['gt_labels'].data.tolist()

    hist, edges = np.histogram(labels, bins=len(_classes))
    plt.bar(np.arange(len(_classes)), hist)
    plt.xticks(np.arange(len(_classes)), _classes, rotation=65)
    plt.tight_layout()
    plt.title(titles[i])
    plt.show()

def map_test(cfg, checkpoint):
  mmdet_test.test(cfg, checkpoint)

def parse_args():
  parser = argparse.ArgumentParser(description='Model testing')
  parser.add_argument('--mode', choices=['map', 'visualize', 'dataset-check'])
  parser.add_argument('--experiment-name')
  args, unknown = parser.parse_known_args()
  return args

def main():
  args = parse_args()
  checkpoints = os.listdir(os.path.join('../vinbig_output', args.experiment_name))
  checkpoints = natsorted(checkpoints)
  checkpoints = [p for p in checkpoints if 'epoch_' in p]
  print('Loading checkpoint ' + checkpoints[-1])
  checkpoint = os.path.join('../vinbig_output', args.experiment_name, checkpoints[-1])
  if args.mode == 'map':
    cfg = get_config()
    map_test(cfg, checkpoint)
  elif args.mode == 'visualize':
    visualize(checkpoint)
  elif args.mode == 'dataset-check':
    cfg = get_config()
    dataset_stats(cfg, checkpoint)

if __name__ == '__main__':
  main()