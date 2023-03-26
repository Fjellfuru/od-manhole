import os

from image_preparation.image_splitter import ImageSplitter
import image_preparation.sort_out_images as soi

source_dir = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10\grid"
dest_dir = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_splitted"
dest_di_not_req = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_splitted\not_required"

if __name__ == "__main__":
    image_list = [os.path.join(source_dir, image) for image in os.listdir(source_dir) if image.endswith(".tif")]
    # for image in os.listdir(source_dir):
    #    if image.endswith(".tif"):
    #        image_path = os.path.join(source_dir, image)
    #        image_list.append(image_path)
    for i in image_list:
        print(i)
        image_splitter = ImageSplitter(i, dest_dir)
        image_splitter.split()
        print(f'===================== Finished with {i}')

    soi.move_image(dest_dir, dest_di_not_req)
