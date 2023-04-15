import os
from image_preparation.image_cropper import ImageCropper
from image_preparation.image_splitter import ImageSplitter
import image_preparation.sort_out_images as soi
from data_augmentation.data_augmentation import DataAugmentation

dir_orig = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10\grid"
dir_cropped = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_cropped"
dir_splitted = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_splitted"
dest_di_not_req = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_splitted\not_required"

if __name__ == "__main__":
    image_list_orig = [os.path.join(dir_orig, image) for image in os.listdir(dir_orig) if image.endswith(".tif")]
    # for image in os.listdir(source_dir):
    #    if image.endswith(".tif"):
    #        image_path = os.path.join(source_dir, image)
    #        image_list.append(image_path)
    for i in image_list_orig:
        image_cropper = ImageCropper(i, dir_cropped)
        image_cropper.crop_image()

    image_list_cropped = [os.path.join(dir_cropped, image) for image in os.listdir(dir_cropped) if image.endswith(".tif")]
    for i in image_list_cropped:
        image_splitter = ImageSplitter(i, dir_splitted)
        image_splitter.split()

    soi.move_image(dir_splitted, dest_di_not_req)

    image_list_da = [os.path.join(dir_splitted, image) for image in os.listdir(dir_splitted) if image.endswith(".tif")]
    for i in image_list_da:
        image_data_augmentation = DataAugmentation(i, dir_splitted)
        image_data_augmentation.transform_images()
