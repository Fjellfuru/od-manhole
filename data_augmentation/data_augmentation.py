import numpy as np

from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance

image_path = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_splitted\swissimage-dop10_2021_2665-1258_0.1_2056_1.tif"
im = Image.open(image_path)
# im_array = np.array(im)
#im.show()

class DataAugmetation:
    def __init__(self, image_path: str, da_image_dir: str):
        self.image_path = image_path
        self.da_image_dir = da_image_dir

gray_img = im.convert("L")
gray_img.show()

#enhancer = ImageEnhance.Sharpness(im)
#enhancer.enhance(10.0)

contrast_im = ImageEnhance.Contrast(im)
contrast_im.enhance(-2).show()

sharp_im = ImageEnhance.Sharpness(im)
sharp_im.enhance(-10.0).show()


contour_im = im.filter(ImageFilter.CONTOUR)
contour_im.show()

detail_im = im.filter(ImageFilter.DETAIL)
detail_im.show()

edgeenhance_im = im.filter(ImageFilter.EDGE_ENHANCE)
edgeenhance_im.show()

emboss_im = im.filter(ImageFilter.EMBOSS)
emboss_im.show()

findedges_im = im.filter(ImageFilter.FIND_EDGES)
findedges_im.show()

im.filter(ImageFilter.BoxBlur(5)).show()