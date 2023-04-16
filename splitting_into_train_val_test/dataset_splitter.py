import os
from pathlib import Path
import shutil
import random

dir_splitted = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_splitted"
dir_label = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_splitted_label"
dir_train_model = r"D:\MAS_DataScience\aerial_images_train_model"


class DatasetSplitter:
    def __init__(self, source_dir_images: str, source_dir_labels: str, dest_dir_train_model: str):
        self.source_dir_images = source_dir_images
        self.source_dir_labels = source_dir_labels
        self.dest_dir_train_model = dest_dir_train_model
        self.dir_train = os.path.join(self.dest_dir_train_model, 'train')
        self.dir_val = os.path.join(self.dest_dir_train_model, 'val')
        self.dir_test = os.path.join(self.dest_dir_train_model, 'test')


    def split_dataset(self):
        train_images, val_images, test_images = self._split_image_list()
        train_labels = self._create_corresponding_label_list(train_images, self.source_dir_labels)
        val_labels = self._create_corresponding_label_list(val_images, self.source_dir_labels)
        test_labels = self._create_corresponding_label_list(test_images, self.source_dir_labels)
        self.move_files(train_images, self.dir_train, 'images')
        self.move_files(train_labels, self.dir_train, 'labels')
        self.move_files(val_images, self.dir_train, 'images')
        self.move_files(val_labels, self.dir_train, 'labels')
        self.move_files(test_images, self.dir_train, 'images')
        self.move_files(test_labels, self.dir_train, 'labels')

    def _split_image_list(self):
        """
        splits a list of images into train, val and test (0.8, 0.1, 0.1)
        :return train_images:
        :return val_images:
        :return test_images:
        """
        image_list = [os.path.join(self.source_dir_images, image) for image in os.listdir(self.source_dir_images) if image.endswith(".tif")]

        image_list.sort()  # make sure that the filenames have a fixed order before shuffling
        random.seed(230)
        random.shuffle(image_list) # shuffles the ordering of filenames (deterministic given the chosen seed)

        split_1 = int(0.8 * len(image_list))
        split_2 = int(0.9 * len(image_list))
        train_images: list = image_list[:split_1]
        val_images: list = image_list[split_1:split_2]
        test_images: list = image_list[split_2:]

        print(len(train_images))
        print(len(val_images))
        print(len(test_images))

        return train_images, val_images, test_images

    def _create_corresponding_label_list(self, image_list: list, source_dir_label: str):
        """
        creates the corresponding label-list to the image-list (train, val, test)
        :param image_list:
        :param source_dir_label:
        :return label_list:
        """
        label_list: list = []
        for i in image_list:
            image_name = os.path.splitext(os.path.basename(i))[0]
            label_path = os.path.join(source_dir_label, image_name + '.txt')
            label_list.append(label_path)
        return label_list

    def move_files(self, file_list: list, dest_dir: str, file_type: str):
        for e in file_list:
            source_dir = e
            dest_dir = dest_dir
            if os.path.isfile(source_dir):
                ext = Path(source_dir).suffix
                file_name = os.path.splitext(os.path.basename(e))[0]
                dest_file = os.path.join(dest_dir, file_type, file_name + ext)
                print(dest_file)




