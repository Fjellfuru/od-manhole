from tqdm import tqdm

from osgeo import gdal
from osgeo import gdal_array

import numpy as np
from pathlib import Path

input_image_path = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10\grid\swissimage-dop10_2021_2665-1258_0.1_2056.tif"
save_path = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_splitted"
crop_size = 1000
repetition_rate = 0
overwrite = False

class SplitImage:
    def __init__(self):
        repetition_rate = 10
        self.read_geotiff()
        self.get_image_properties()
        self.get_save_options()
        self.

    def read_geotiff(self):
        """
        read geotiff and properties
        :return:
        """
        dataset = gdal.Open(input_image_path, gdal.GA_ReadOnly)
        self.image = dataset.ReadAsArray()  # get the rasterArray
        # convert 2D raster to [1, H, W] format
        if len(self.image.shape) == 2:
            self.image = self.image[np.newaxis, :, :]
        self.proj = dataset.GetProjection()
        self.geotrans = dataset.GetGeoTransform()
        return self.image, self.proj, self.geotrans

    def count_files(self, folder_path):
        count = 0
        for path in Path(folder_path).iterdir():
            if path.is_file():
                count += 1
        return count

    def get_save_options(self):
        # get image suffix
        ext = Path(input_image_path).suffix
        # check output folder, if not exists, creat it.
        Path(save_path).mkdir(parents=True, exist_ok=True)

        if overwrite:
            new_name = 1
        else:
            cnt = self.count_files(save_path)
            new_name = cnt + 1
            print(f"There are {cnt} files in the {save_path}")
            print(f"New image name will start with {new_name}")

    def get_image_properties(self):
        self.stride = int(crop_size * (1 - repetition_rate))
        print(f"{crop_size=}, {self.stride=}")

        H = self.imgage.shape[1]
        W = self.imgage.shape[2]

        self.n_rows = int((H - crop_size)/self.stride + 1)
        self.n_cols = int((W - crop_size)/self.stride + 1)

        xmin = self.geotrans[0]
        ymax = self.geotrans[3]
        res = self.geotrans[1]

        xlen = res * self.dataset.RasterXSize
        ylen = res * self.dataset.RasterYSize

        xsize = xlen/self.n_rows
        ysize = ylen/self.n_cols

        self.xsteps = [xmin + xsize * i for i in range(self.n_rows+1)]
        self.ysteps = [ymax - ysize * i for i in range(self.n_cols+1)]

    def tile_generator(self):
        for idh in range(self.n_rows):
            h = idh * self.stride
            ymax = self.ysteps[idh]
            ymin = self.ysteps[idh + 1]
            for idw in range(self.n_cols):
                w = idw * self.stride
                xmin = self.xsteps[idw]
                xmax = self.xsteps[idw + 1]
                new_geotrans = (xmin, self.geotrans[1], self.geotrans[2], ymax, self.geotrans[4], self.geotrans[5])
                yield h, w, new_geotrans


    def save_image_GeoTiff(self, im_data, im_geotrans, im_proj, file_name):
        if Path(file_name).is_file():
            print(f"Overwrite existing file: {file_name}")

        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        elif len(im_data.shape) == 2:
            im_data = np.array([im_data])
            im_bands, im_height, im_width = im_data.shape

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(file_name, int(im_width), int(
            im_height), int(im_bands), datatype)
        if(dataset != None):
            dataset.SetGeoTransform(im_geotrans)
            dataset.SetProjection(im_proj)
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        del dataset

    with tqdm(total=n_rows * n_cols, desc='Generating', colour='green', leave=True, unit='img') as pbar:
        for n, (h, w, new_geotrans) in enumerate(tile_generator()):
            crop_img = padded_img[:, h:h + crop_size, w: w + crop_size]
            crop_image_name = f"{new_name:04d}{ext}"
            crop_image_path = Path(save_path) / crop_image_name
            self.save_image_GeoTiff(crop_img, new_geotrans, proj, str(crop_image_path))
            new_name = new_name + 1
            pbar.update(1)

