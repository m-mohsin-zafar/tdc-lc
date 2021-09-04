import cv2
import os
import numpy as np
import torch
from copy import deepcopy


class PostProcess:
    def __init__(self):
        self._id2cls = {
            0: 'Artifact',
            1: 'Normal',
            2: 'Lymphocyte'
        }

    def extract_label(self, y):
        _, predicted = torch.max(y.data, 1)
        return predicted.item()

    def label_to_class(self, k):
        return self._id2cls[k]

    def _bbox_to_cxy(self, bbox):
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return np.array((cx, cy))

    def _get_filtered_preds(self, pred, dist_thrs=13):  # dist_thr changed to 8 from 13
        r = {}
        indices_to_remove = []
        for i in range(len(pred)):

            if i + 1 == len(pred):
                already_exists = False
                for k in list(r):
                    if i in r[k]:
                        already_exists = True
                if already_exists:
                    r[i] = [-1]
                    indices_to_remove.append(i)
                else:
                    r[i] = [i]

                break

            for j in range(i + 1, len(pred)):
                dist = np.linalg.norm(self._bbox_to_cxy(pred[i]) - self._bbox_to_cxy(pred[j]))
                if dist <= dist_thrs:
                    if i in r.keys():
                        r[i].append(j)
                    else:
                        already_exists = False
                        for k in list(r):
                            if i in r[k]:
                                already_exists = True
                                r[k].append(j)
                                r[i] = [-1]
                                indices_to_remove.append(i)
                        if not already_exists:
                            r[i] = [i, j]
                        else:
                            r[i] = [i]

                elif i not in list(r):
                    already_exists = False
                    for k in r.keys():
                        if i in r[k]:
                            already_exists = True
                    if already_exists:
                        r[i] = [-1]
                        indices_to_remove.append(i)
                    else:
                        r[i] = [i]

        filtered_indexes = list(set(r.keys()) - set(indices_to_remove))
        filtered_preds = [pred[i] for i in filtered_indexes]

        return filtered_indexes, filtered_preds

    def process_detection_output(self, inp_img, y):
        count = 0
        viz = None

        predicted_boxes = y['instances'].pred_boxes.to('cpu').tensor.numpy().tolist()
        idx, filtered_boxes = self._get_filtered_preds(predicted_boxes)
        if len(predicted_boxes) >= 1:
            predicted_masks = y['instances'].pred_masks.to('cpu').numpy()[idx]
            count = len(predicted_boxes)
            viz = self._prepare_viz(inp_img, filtered_boxes, predicted_masks)

        return count, viz

    def _prepare_viz(self, inp_img, boxes, masks):
        tmp_img = deepcopy(inp_img)
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        for box, mask in zip(boxes, masks):
            tmp_img = cv2.rectangle(tmp_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (144, 238, 144), 2)
            tmp_img[mask] = tmp_img[mask] * 0.5 + np.array([136, 8, 8]) * 0.5

        return tmp_img
