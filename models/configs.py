import os
from detectron2 import model_zoo
from detectron2.config import get_cfg


class Configurations:

    def __init__(self, model_name, param_key, device=None):
        self.model_name = model_name
        self.device = device
        self.param_key = param_key
        self.cfg = get_cfg()
        self._set_device()
        self._set_model()
        self._set_commons()
        self._model_specific_params = self._get_model_params(self.param_key)
        self._set_model_specific_params(self._model_specific_params)

    def _set_device(self):
        if self.device.type == 'cpu':
            self.cfg.MODEL.DEVICE = 'cpu'

    def _set_model(self):
        if self.model_name == 'mrcnn':
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))

    def _set_commons(self):
        self.cfg.INPUT.MIN_SIZE_TEST = 256
        self.cfg.INPUT.MAX_SIZE_TEST = 256
        if self.model_name == 'mrcnn':
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    def _set_model_specific_params(self, args):
        self.cfg.MODEL.WEIGHTS = args['WEIGHTS']

    def get_configurations(self):
        return self.cfg

    def _get_model_params(self, key):
        self._args = {
            'mrcnn_circular_dab': {
                'WEIGHTS': './checkpoints/model_final.pth',
            }
        }

        return self._args[key]
