import os

from tqdm import tqdm

from osgeo import gdal
from osgeo import gdal_array

import numpy as np
from pathlib import Path

source_dir = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10\grid\swissimage-dop10_2021_2665-1258_0.1_2056.tif"
dest_dir = r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_splitted"

class SplitImage:
    def __init__(self, source_dir, dest_dir):
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.crop_size = 10
        self.repetition_rate = 0
        self.overwrite = False
        self.read_geotiff()
        self.get_image_properties()
        self.get_save_options()
        count = 0
        with tqdm(total=self.n_rows * self.n_cols, desc='Generating', colour='green', leave=True, unit='img') as pbar:
            for n, (self.h, self.w, self.new_geotrans) in enumerate(self.tile_generator()):
                count += 1
                crop_img = self.image[:, self.h:self.h + self.crop_size, self.w: self.w + self.crop_size]
                crop_image_name = f"{self.file_name}{'_'}{count}{self.ext}"
                crop_image_path = Path(dest_dir) / crop_image_name
                self.save_image_GeoTiff(crop_img, self.new_geotrans, self.proj, str(crop_image_path))
                pbar.update(1)

    def read_geotiff(self):
        """
        read geotiff and properties
        :return:
        """
        self.dataset = gdal.Open(self.source_dir, gdal.GA_ReadOnly)
        self.image =self.dataset.ReadAsArray()  # get the rasterArray
        # convert 2D raster to [1, H, W] format
        if len(self.image.shape) == 2:
            self.image = self.image[np.newaxis, :, :]
        self.proj = self.dataset.GetProjection()
        self.geotrans = self.dataset.GetGeoTransform()
        return self.image, self.proj, self.geotrans

    def get_image_properties(self):
        self.stride = int(self.crop_size * (1 - self.repetition_rate))
        print(f"{self.crop_size=}, {self.stride=}")

        H = self.image.shape[1]
        W = self.image.shape[2]

        self.n_rows = int((H - self.crop_size)/self.stride + 1)
        self.n_cols = int((W - self.crop_size)/self.stride + 1)

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
            self.h = idh * self.stride
            ymax = self.ysteps[idh]
            ymin = self.ysteps[idh + 1]
            for idw in range(self.n_cols):
                self.w = idw * self.stride
                xmin = self.xsteps[idw]
                xmax = self.xsteps[idw + 1]
                self.new_geotrans = (xmin, self.geotrans[1], self.geotrans[2], ymax, self.geotrans[4], self.geotrans[5])
                yield self.h, self.w, self.new_geotrans

    def count_files(self, folder_path):
        count = 0
        for path in Path(folder_path).iterdir():
            if path.is_file():
                count += 1
        return count

    def get_save_options(self):
        # get original filename
        self.file_name = os.path.splitext(os.path.basename(self.source_dir))[0]

        # get image suffix
        self.ext = Path(source_dir).suffix
        # check output folder, if not exists, creat it.
        Path(self.dest_dir).mkdir(parents=True, exist_ok=True)

        if self.overwrite:
            self.new_name = 1
        else:
            cnt = self.count_files(self.dest_dir)
            self.new_name = cnt + 1
            print(f"There are {cnt} files in the {self.dest_dir}")
            print(f"New image name will start with {self.new_name}")

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



