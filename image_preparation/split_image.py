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

dataset = gdal.Open(input_image_path, gdal.GA_ReadOnly)
image = dataset.ReadAsArray()  # get the rasterArray
# convert 2D raster to [1, H, W] format
if len(image.shape) == 2:
    image = image[np.newaxis, :, :]
proj = dataset.GetProjection()
geotrans = dataset.GetGeoTransform()
print(f'Proj: {proj}')
print(f'Geotransform: {geotrans}')

if image is None:
    print("Image not found")
# get image suffix
ext = Path(input_image_path).suffix
# check output folder, if not exists, creat it.
Path(save_path).mkdir(parents=True, exist_ok=True)

print(f"Input Image File Shape (D, H, W):{image.shape}")

stride = int(crop_size * (1 - repetition_rate))
print(f"{crop_size=}, {stride=}")

D = image.shape[0]  # this one is for (D, H, W) format Channel First.
height = image.shape[1]
width = image.shape[2]
# get the minial padding image size
H = int(np.ceil(height / stride) * stride)
W = int(np.ceil(width / stride) * stride)

padded_img = np.zeros((D, H, W), dtype=image.dtype)
for d in range(D):  # padding every layer
    onelayer = image[d, :, :]
    padded_img[d, :, :] = np.pad(
        onelayer, ((0, H - height), (0, W - width)), "reflect"
    )

H = padded_img.shape[1]
W = padded_img.shape[2]

print(f"Padding Image File Shape (D, H, W):{ padded_img.shape}")

def count_files(folder_path):
    count = 0
    for path in Path(folder_path).iterdir():
        if path.is_file():
            count += 1
    return count

if overwrite:
    new_name = 1
else:
    cnt = count_files(save_path)
    new_name = cnt + 1
    print(f"There are {cnt} files in the {save_path}")
    print(f"New image name will start with {new_name}")

n_rows = int((H - crop_size)/stride + 1)
n_cols = int((W - crop_size)/stride + 1)

print(n_rows)
print(n_cols)

xmin = geotrans[0]
ymax = geotrans[3]
res = geotrans[1]

xlen = res * dataset.RasterXSize
ylen = res * dataset.RasterYSize

xsize = xlen/n_rows
ysize = ylen/n_cols

xsteps = [xmin + xsize * i for i in range(n_rows+1)]
ysteps = [ymax - ysize * i for i in range(n_cols+1)]

print(xsteps)
print(ysteps)

def tile_generator():
    for idh in range(n_rows):
        h = idh * stride
        ymax = ysteps[idh]
        ymin = ysteps[idh + 1]
        for idw in range(n_cols):
            w = idw * stride
            xmin = xsteps[idw]
            xmax = xsteps[idw + 1]
            new_geotrans = (xmin, geotrans[1], geotrans[2], ymax, geotrans[4], geotrans[5])
            yield h, w, new_geotrans


def save_rasterGeoTIF(im_data, im_geotrans, im_proj, file_name):
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
        # save_rasterArray(crop_img,  str(crop_image_path)) # just save the raster image
        save_rasterGeoTIF(crop_img, new_geotrans, proj, str(crop_image_path))
        new_name = new_name + 1
        pbar.update(1)

