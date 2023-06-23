import os
from pathlib import Path
from PIL import Image
from osgeo import gdal
import rasterio as rio


class ImageCropper:
    def __init__(self, source_dir: str, dest_dir: str):
        self.source_dir = source_dir
        self.dest_dir = dest_dir

    def crop_image(self):
        """
        crop the original imagage and set georeferencing to cropped image
        :return:
        """
        new_minx, new_maxy, new_maxx, new_miny = self._get_new_geotransform()
        self._crop_10_10_image()
        self._set_georeferencing(new_minx, new_maxy, new_maxx, new_miny)
        self._remove_not_georeferencd_file()

    def _get_new_geotransform(self):
        """
        read orig image and get new geotransform
        :return:
        """
        dataset = gdal.Open(self.source_dir, gdal.GA_ReadOnly)
        geotrans = dataset.GetGeoTransform()

        new_minx = geotrans[0] + 4
        new_maxy = geotrans[3] - 4
        new_maxx = (geotrans[0] + geotrans[1] * dataset.RasterXSize) - 4
        new_miny = (geotrans[3] + geotrans[5] * dataset.RasterYSize) + 4

        # get fil name
        self.file_name_orig = os.path.splitext(os.path.basename(self.source_dir))[0]

        return new_minx, new_maxy, new_maxx, new_miny

    def _crop_10_10_image(self):
        """
        crop the original image with 10000x10000 pixels to 9920x9920 pixels
        :return:
        """
        #set max of image size
        Image.MAX_IMAGE_PIXELS = 100000000

        # read image
        im = Image.open(self.source_dir)

        # crop image to 9920x9920
        im_crop = im.crop((40, 40, 9960, 9960))

        # define destination path
        self.crop_not_geo_dest = Path(self.dest_dir) / f"{self.file_name_orig}{'_'}{'cropped_not_geo.tif'}"
        # save image with quality of 100%
        im_crop.save(self.crop_not_geo_dest, quality=100)

    def _set_georeferencing(self, new_minx: int, new_maxy: int, new_maxx: int, new_miny: int):
        """
        georeferences the image
        :param new_minx:
        :param new_maxy:
        :param new_maxx:
        :param new_miny:
        :return:
        """

        # read not georeferenced image
        dataset = rio.open(self.crop_not_geo_dest)

        # define 3 bands (rgb)
        bands = [1, 2, 3]
        data = dataset.read(bands)

        # set geotransform for georeferencing
        transform = rio.transform.from_bounds(
            new_minx,
            new_miny,
            new_maxx,
            new_maxy,
            data.shape[1],
            data.shape[2]
        )

        # set the output image kwargs
        kwargs = {
            "driver": "GTiff",
            "width": data.shape[1],
            "height": data.shape[2],
            "count": len(bands),
            "dtype": data.dtype,
            "transform": transform,
            "crs": "EPSG:2056"
        }

        # save georeferenced image
        self.crop_geo_dest = Path(self.dest_dir) / f"{self.file_name_orig}{'_'}{'cropped.tif'}"

        with rio.open(self.crop_geo_dest, "w", **kwargs) as dst:
            dst.write(data, indexes=bands)
            dst.close()
            dataset.close()

    def _remove_not_georeferencd_file(self):
        # delete original cropped file, we only need the georeferenced image
        if os.path.exists(self.crop_not_geo_dest):
            os.remove(self.crop_not_geo_dest)
        else:
            print("The file does not exist")