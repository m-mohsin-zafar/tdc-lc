from models.build import BuildModels
from pipeline.preprocess import PreProcess
from pipeline.postprocess import PostProcess


class Inference:
    def __init__(self):
        _models = BuildModels()
        self._device = _models.device
        self._clf = _models.get_classifier()
        self._det = _models.get_detector()
        self._preprocess = PreProcess()
        self._postprocess = PostProcess()

    def infer(self, fp):

        # 0. Default Returns
        count = 0
        viz = None

        # 1. Load Input Image
        x = self._preprocess.get_pil_image(fp)
        # 2. Apply transformations
        x = self._preprocess.apply_transformations()(x)
        # 3. Unsqueeze to change (C, H, W) to (1, C, H, W)
        x = x.unsqueeze(0)
        # 4. Pass from Classification Stage to remove Hard Negative Sample
        y_p1 = self._clf(x.to(self._device))
        # 5. Extract Label
        y_p1 = self._postprocess.extract_label(y_p1)
        # 6.Make Decision to get rid of Hard Negative Sample
        if y_p1 == 0:
            # 6A. Return count = 0 and Image=None
            count = 0
            viz = None
        else:
            # 6B. Apply Stain Deconvolution
            x_dab = self._preprocess.apply_stain_deconvolution(fp)
            # 7. Pass from Detection Stage to get Predictions
            y_p2 = self._det(x_dab)
            # 8. Postprocess output to get count, and visualization
            count, viz = self._postprocess.process_detection_output(x_dab, y_p2)

        # 9. Finally, return count and visualization
        return count, viz


if __name__ == "__main__":
    fp = '../data/training_112.png'
    _pipeline = Inference()
    count, viz = _pipeline.infer(fp)