# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import numpy as np

def compute_bboxes_IoU(bboxes_1, bboxes_2):
    area_1 = (bboxes_1[:, 2] - bboxes_1[:, 0] + 1) * (bboxes_1[:, 3] - bboxes_1[:, 1] + 1)
    area_2 = (bboxes_2[:, 2] - bboxes_2[:, 0] + 1) * (bboxes_2[:, 3] - bboxes_2[:, 1] + 1)

    iw = np.minimum(bboxes_1[:, 2][:, np.newaxis], bboxes_2[:, 2]) - np.maximum(bboxes_1[:, 0][:, np.newaxis], bboxes_2[:, 0]) + 1
    ih = np.minimum(bboxes_1[:, 3][:, np.newaxis], bboxes_2[:, 3]) - np.maximum(bboxes_1[:, 1][:, np.newaxis], bboxes_2[:, 1]) + 1

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    
    intersection = iw * ih
    union = (area_1[:, np.newaxis] + area_2) - iw * ih

    return intersection / np.maximum(union, 1e-10)
    