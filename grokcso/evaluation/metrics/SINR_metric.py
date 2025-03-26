# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
from collections import OrderedDict
from typing import List, Optional

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.registry import METRICS
from mmengine.logging import print_log


def caculate_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.sqrt(np.sum(np.square(point1 - point2)))

# 计算检测率和误检率
def det_rate_sinr(det_results, annotations, dis_thr, logger=None):
    """Calculate True Positive and False Positive.

    Args:
        preds (list): Detection bboxes.
        gts (list): Ground truth bboxes.
        dis_thr (float): distance threshold.
        logger (logging.Logger | str | None): The way to print the mAP

    Returns:
        tuple: True positives and false positives.
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)  # 图片数量

    total_targets = 0  # 总目标数
    true_positives = 0  # 真正例数
    false_positives = 0  # 假正例数

    for i in range(num_imgs):
        det_points = det_results[i]
        gt_points = annotations[i]
        gt_points = gt_points["targets"]

        matched_predictions = set()
        for gt_point in gt_points:
            for det_point in det_points:
                det_point_tuple = tuple(det_point)
                if det_point_tuple in matched_predictions:
                    continue
                if caculate_distance(det_point[:2], gt_point[:2]) < dis_thr:
                    matched_predictions.add(det_point_tuple)
                    break

        true_positives += len(matched_predictions)
        false_positives += len(det_points) - len(matched_predictions)
        total_targets += len(gt_points)

    det_rate = true_positives / total_targets if total_targets > 0 else 0.0      # 检测率，同样也是召回率
    false_alarm_rate = false_positives / (false_positives + true_positives) \
      if (false_positives + true_positives) > 0 else 0.0

    print_log(f'det_rate: {det_rate:.3f}', logger=logger)
    print_log(f'false_rate: {false_alarm_rate:.3f}', logger=logger)

    return det_rate, false_alarm_rate


@METRICS.register_module()
class SINR_DET_RATE(BaseMetric):

    default_prefix: Optional[str] = 'cso_metric'

    def __init__(self,
                 iou_thrs=(0.05, 0.1, 0.15, 0.2, 0.25),
                 brightness_threshold: float = 50,
                 c=3,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # iou_thrs是一个列表
        self.iou_thrs = list(iou_thrs)
        self.brightness_threshold = brightness_threshold
        self.c = c

    def process(self, data_batch: dict, outputs) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        preds的输出结果是在 c 超分倍数下的输出结果，需要将其转换为在 11 * 11 的图像中的位置，
        并与标注信息进行比较，计算指标。
        c超分倍率下位置 （i, j）对应到 11 * 11 的图像中的位置是
        ((i - c // 2)/c, (j - c // 2)/c)


        Args:
            data_batch (dict): A batch of data from the dataloader.
            outputs : A batch of data samples that
                contain annotations and x_output.
                x_output: torch.Tensor, shape (N, 1089)
                ann_list: list , shape (N, K, 3)
                N: batch size
                K: number of objects
                3: (x, y, brightness)
        """
        x_output = outputs[0].cpu().numpy()
        img_name = outputs[1]
        ann_list = outputs[2]

        for idx in range(x_output.shape[0]):
          # 一个样本的标注信息
          ann = dict(targets=ann_list[idx])   # 每个序列支持帧 gt

          dets = []    # 样本检测结果
          
          # 计算信噪比
          snr = self.calculate_snr(x_output[idx])
          matrix = x_output[idx].reshape(11 * self.c, 11 * self.c)
          matrix[matrix < self.brightness_threshold] = 0

          non_zero_indices = np.nonzero(matrix)

          for i in range(len(non_zero_indices[0])):
            row = non_zero_indices[0][i]
            col = non_zero_indices[1][i]
            value = matrix[row, col]
            dets.append([float(1.0 * (row - (self.c-1)//2) / self.c),
                         float(1.0 * (col - (self.c-1)//2) / self.c),
                        value])
          result_dir = '/opt/data/private/xubohan/work_dirs/temp'
          if not os.path.exists(result_dir):
                os.makedirs(result_dir)
          txt_name = img_name[idx+2][:-len(".png")] + ".txt"
          txt_path = os.path.join(result_dir, txt_name)
          with open(txt_path, "w") as f:
              for i in range(len(non_zero_indices[0])):
                f.write(str((non_zero_indices[0][i] - 1) / 3) + " " + str(
                  (non_zero_indices[1][i] - 1) / 3) + " " +
                  str(matrix[non_zero_indices[0][i], non_zero_indices[1][i]]) + "\n")

          self.results.append((ann, dets, snr))

    def compute_metrics(self, results: list) -> dict:

        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds, snrs = zip(*results)
        eval_results = OrderedDict()
        
        mean_det_rate = []
        mean_false_alarm_rate = []
        for iou_thr in self.iou_thrs:
            logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
            det_rate, false_alarm_rate = det_rate_sinr(
                preds,
                gts,
                dis_thr=iou_thr,
                logger=logger,
                )
            mean_det_rate.append(det_rate)
            mean_false_alarm_rate.append(false_alarm_rate)
            eval_results[f'det_rate{int(iou_thr * 100):02d}'] = round(
              det_rate, 3)
            eval_results[f'false_alarm_rate{int(iou_thr * 100):02d}'] = round(
              false_alarm_rate, 3)

        print('eval_results:', eval_results)
        
        # 计算信噪比
        eval_results['snr'] = round(sum(snrs) / len(snrs), 3)
        
        # 计算mAP
        eval_results['mean_det_rate'] = sum(mean_det_rate) / len(mean_det_rate)
        eval_results['mean_false_alarm_rate'] = sum(mean_false_alarm_rate) /\
                                                 len(mean_false_alarm_rate)
        # 将字典中mAP放在第一个位置
        eval_results.move_to_end('mean_det_rate', last=False)
        eval_results.move_to_end('mean_false_alarm_rate', last=False)
        eval_results.move_to_end('snr', last=False)
        return eval_results
    
    def calculate_snr(self, image):
      signal = image[image > self.brightness_threshold]
      nosie = image[image <= self.brightness_threshold]
      snr = np.var(signal) / np.var(nosie)
      snr = 10 * np.log10(snr)
      return snr

