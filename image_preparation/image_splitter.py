import os
from pathlib import Path
from typing import Union

import numpy as np
from osgeo import gdal
from tqdm import tqdm


class ImageSplitter:
    def __init__(self, source_dir: str, dest_dir: str):
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.crop_size = 992
        self.repetition_rate = 0
        self.overwrite = False
        self.stride = int(self.crop_size * (1 - self.repetition_rate))

        self._read_geotiff()
        self._get_save_options()

    def split(self):
        n_rows, n_cols, xsteps, ysteps = self._get_image_properties()
        count = 0
        with tqdm(total=n_rows * n_cols, desc='Generating', colour='green', leave=True, unit='img') as pbar:
            for n, (h, w, new_geotrans) in enumerate(self._generate_tiles(n_rows, n_cols, xsteps, ysteps)):
                count += 1
                crop_img = self.image[:, h:h + self.crop_size, w: w + self.crop_size]
                crop_image_name = f"{self.file_name}{'_'}{count}{self.ext}"
                crop_image_path = Path(self.dest_dir) / crop_image_name
                self._save_image_geotiff(crop_img, new_geotrans, self.proj, str(crop_image_path))
                pbar.update(1)

    def _read_geotiff(self):
        """
        read geotiff and properties
        :return:
        """
        self.dataset = gdal.Open(self.source_dir, gdal.GA_ReadOnly)
        self.image = self.dataset.ReadAsArray()  # get the rasterArray
        # convert 2D raster to [1, H, W] format
        if len(self.image.shape) == 2:
            self.image = self.image[np.newaxis, :, :]
        self.proj = self.dataset.GetProjection()
        self.geotrans = self.dataset.GetGeoTransform()

    def _get_image_properties(self):
        """
        return image properties to generate tiles
        :return:
        """
        print(f"{self.crop_size=}, {self.stride=}")

        H = self.image.shape[1]
        W = self.image.shape[2]

        n_rows = int((H - self.crop_size) / self.stride + 1)
        n_cols = int((W - self.crop_size) / self.stride + 1)

        xmin = self.geotrans[0]
        ymax = self.geotrans[3]
        res = self.geotrans[1]

        xlen = res * self.dataset.RasterXSize
        ylen = res * self.dataset.RasterYSize

        xsize = xlen / n_rows
        ysize = ylen / n_cols

        xsteps = [xmin + xsize * i for i in range(n_rows + 1)]
        ysteps = [ymax - ysize * i for i in range(n_cols + 1)]

        return n_rows, n_cols, xsteps, ysteps

    def _generate_tiles(
            self,
            n_rows: int,
            n_cols: int,
            xsteps: list[int],
            ysteps: list[int]
    ) -> (int, int, Union[int, float]):
        for idh in range(n_rows):
            h = idh * self.stride
            ymax = ysteps[idh]
            for idw in range(n_cols):
                w = idw * self.stride
                xmin = xsteps[idw]
                new_geotrans = (xmin, self.geotrans[1], self.geotrans[2], ymax, self.geotrans[4], self.geotrans[5])
                yield h, w, new_geotrans

    def count_files(self, folder_path):
        count = 0
        for path in Path(folder_path).iterdir():
            if path.is_file():
                count += 1
        return count

    def _get_save_options(self):
        # get original filename
        self.file_name = os.path.splitext(os.path.basename(self.source_dir))[0]

        # get image suffix
        self.ext = Path(self.source_dir).suffix
        # check output folder, if not exists, creat it.
        Path(self.dest_dir).mkdir(parents=True, exist_ok=True)

        if self.overwrite:
            self.new_name = 1
        else:
            cnt = self.count_files(self.dest_dir)
            self.new_name = cnt + 1
            print(f"There are {cnt} files in the {self.dest_dir}")
            print(f"New image name will start with {self.new_name}")

    def _save_image_geotiff(self, im_data, im_geotrans, im_proj, file_name):
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
        else:
            raise Exception('Unknown number of bands')

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(file_name, int(im_width), int(
            im_height), int(im_bands), datatype)
        if dataset is not None:
            dataset.SetGeoTransform(im_geotrans)
            dataset.SetProjection(im_proj)
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        del dataset
