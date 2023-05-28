
# check if weight and height are divisible by 32
num = 700
number = num - (num % 32)
print(number)


import os
from pathlib import Path
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
from osgeo import gdal
import rasterio as rio
import pandas as pd
from ultralytics import YOLO

# from IPython.display import display, Image


model = YOLO(f"D:\MAS_DataScience\yolo_manhole\yolov8m_4524_150_16.pt")
image_orig = f'D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_splitted\swissimage-dop10_2021_2602-1200_0.1_2056_cropped_70.tif'
image = f'D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_splitted\swissimage-dop10_2021_2602-1200_0.1_2056_cropped_70_cm1.tif'

dataset = gdal.Open(image_orig, gdal.GA_ReadOnly)
geotrans = dataset.GetGeoTransform()

minx = geotrans[0]
maxy = geotrans[3]

pix_size = geotrans[1]


#image = f"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_cropped\swissimage-dop10_2021_2665-1258_0.1_2056_cropped.tif"
results = model.predict(source=image, line_width=1, show_labels=False, conf=0.05, imgsz=992)

for result in results:
    # detection
    #print(result.boxes.xyxy)   # box with xyxy format, (N, 4)
    #print(result.boxes.xywh)    # box with xywh format, (N, 4)
    xywh = result.boxes.xywh
    xywh_list = xywh.tolist()
    x = [i[0] for i in xywh_list]
    y = [i[1] for i in xywh_list]
    w = [i[2] for i in xywh_list]
    h = [i[3] for i in xywh_list]
    #print(result.boxes.xyxyn)   # box with xyxy format but normalized, (N, 4)
    #print(result.boxes.xywhn)   # box with xywh format but normalized, (N, 4)
    #print(result.boxes.conf)    # confidence score, (N, 1)
    conf = result.boxes.conf
    conf_list = conf.tolist()
    #print(result.boxes.cls)     # cls, (N, 1)
    cls = result.boxes.cls
    cls_list = cls.tolist()

    df = pd.DataFrame(list(zip(cls_list, conf_list, x, y, w, h)),
                      columns=['class', 'confidence', 'pixel_x', 'pixel_y', 'pixel_w', 'pixel_h'])

    df['origin_x'] = minx
    df['origin_y'] = maxy
    df['d_x'] = df['pixel_x'] * pix_size
    df['d_y'] = df['pixel_y'] * pix_size
    df['x'] = df['origin_x'] + df['d_x']
    df['y'] = df['origin_y'] - df['d_y']

    df['class'] = df['class'].astype('int')

    df.to_csv(r"D:\MAS_DataScience\Test\yolov8m_4524_150_16_cm1_70.csv", sep=';', index=False)