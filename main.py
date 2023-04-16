import os
from image_preparation.image_cropper import ImageCropper
from image_preparation.image_splitter import ImageSplitter
import image_preparation.sort_out_images as soi
from data_augmentation.data_augmentation import DataAugmentation
from labeling.image_labeler import ImageLabeler
from splitting_into_train_val_test.dataset_splitter import DatasetSplitter

dir_orig: str = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10\grid"
dir_cropped: str = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_cropped"
dir_splitted: str = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_splitted"
dir_not_req: str = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_splitted\not_required"
dir_label: str = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_splitted_label"
dir_dataset_train_model: str = r"D:\MAS_DataScience\aerial_images_train_model"

execute_cropper: str = 'no'
execute_image_splitter: str = 'no'
execute_labeler: str = 'no'
execute_dataset_splitter: str = 'yes'

if __name__ == "__main__":
    if execute_cropper == 'yes':
        image_list_orig = [os.path.join(dir_orig, image) for image in os.listdir(dir_orig) if image.endswith(".tif")]
        # for image in os.listdir(source_dir):
        #    if image.endswith(".tif"):
        #        image_path = os.path.join(source_dir, image)
        #        image_list.append(image_path)
        for i in image_list_orig:
            image_cropper = ImageCropper(i, dir_cropped)
            image_cropper.crop_image()
    else:
        pass

    if execute_image_splitter == 'yes':
        image_list_cropped = [os.path.join(dir_cropped, image) for image in os.listdir(dir_cropped) if image.endswith(".tif")]
        for i in image_list_cropped:
            image_splitter = ImageSplitter(i, dir_splitted)
            image_splitter.split()

        soi.move_image(dir_splitted, dir_not_req)

        image_list_da = [os.path.join(dir_splitted, image) for image in os.listdir(dir_splitted) if image.endswith(".tif")]
        for i in image_list_da:
            image_data_augmentation = DataAugmentation(i, dir_splitted)
            image_data_augmentation.transform_images()
    else:
        pass

    if execute_labeler == 'yes':
        image_label = ImageLabeler(dir_label)
        image_label.create_label_file()
    else:
        pass

    if execute_dataset_splitter == 'yes':
        dataset_split = DatasetSplitter(dir_splitted, dir_label, dir_dataset_train_model)
        dataset_split.split_dataset()
    else:
        pass


