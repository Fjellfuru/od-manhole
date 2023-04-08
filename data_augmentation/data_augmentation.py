import numpy as np
import os
from pathlib import Path
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance



class DataAugmentation:
    def __init__(self, image_path: str, da_image_dir: str):
        self.image_path = image_path
        self.da_image_dir = da_image_dir
        self.im = Image.open(self.image_path)

    def transform_images(self):
        self.transform_image_to_greyscale()
        self.switch_rgb_bands_to_bgr()
        self.switch_rgb_bands_to_bgr()
        self.increase_contrast()
        self.decrease_contrast()

    def save_image(self, im_transformed, transformer):
        image_name = os.path.splitext(os.path.basename(self.image_path))[0]
        # get image suffix
        ext = Path(self.image_path).suffix
        transformed_image_name = f"{image_name}{'_'}{transformer}{ext}"
        outputfile = os.path.join(self.da_image_dir, transformed_image_name)
        im_transformed.save(outputfile)

    def transform_image_to_greyscale(self):
        transformer = 'gray'
        gray_img = self.im.convert("L")
        self.save_image(gray_img, transformer)

    def switch_rgb_bands_to_bgr(self):
        transformer = 'bgr'
        r, g, b = self.im.split()
        bgr_img = Image.merge("RGB", (b, g, r))
        self.save_image(bgr_img, transformer)

    def switch_rgb_bands_to_grb(self):
        transformer = 'grb'
        r, g, b = self.im.split()
        grb_img = Image.merge("RGB", (g, r, b))
        self.save_image(grb_img, transformer)

    def increase_contrast(self):
        transformer = 'cp2'
        contrast_im = ImageEnhance.Contrast(self.im).enhance(2)
        self.save_image(contrast_im, transformer)

    def decrease_contrast(self):
        transformer = 'cm2'
        contrast_im = ImageEnhance.Contrast(self.im).enhance(-2)
        self.save_image(contrast_im, transformer)



#enhancer = ImageEnhance.Sharpness(im)
#enhancer.enhance(10.0)
#
#sharp_im = ImageEnhance.Sharpness(im)
#sharp_im.enhance(-10.0).show()
#
#
#contour_im = im.filter(ImageFilter.CONTOUR)
#contour_im.show()
#
#detail_im = im.filter(ImageFilter.DETAIL)
#detail_im.show()
#
#edgeenhance_im = im.filter(ImageFilter.EDGE_ENHANCE)
#edgeenhance_im.show()
#
#emboss_im = im.filter(ImageFilter.EMBOSS)
#emboss_im.show()
#
#findedges_im = im.filter(ImageFilter.FIND_EDGES)
#findedges_im.show()
#
#im.filter(ImageFilter.BoxBlur(5)).show()