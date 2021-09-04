import cv2
import os
import numpy as np
import matplotlib.image as mpimg
from torchvision import transforms
from matplotlib.colors import LinearSegmentedColormap
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
from skimage.io import imsave
from PIL import Image


class PreProcess:

    def get_pil_image(self, input_fp):
        return Image.open(input_fp)

    def apply_transformations(self):
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

    def apply_stain_deconvolution(self, input_fp):
        ihc_rgb = mpimg.imread(input_fp)
        ihc_hed = rgb2hed(ihc_rgb)

        # Rescale hematoxylin and DAB signals and give them a fluorescence look
        dab = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1))

        cmap = LinearSegmentedColormap.from_list('d', ['white', 'saddlebrown'])

        target_path = './pipeline/tmp/'+os.path.basename(input_fp)
        mpimg.imsave(fname=target_path, arr=dab, cmap=cmap)

        dab_img = cv2.imread(target_path)

        return dab_img
