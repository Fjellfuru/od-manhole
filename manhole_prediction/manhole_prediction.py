from osgeo import gdal
import pandas as pd
from ultralytics import YOLO



model = YOLO(f"D:\MAS_DataScience\yolo_manhole\yolov8m_6040_150_16_721.pt")
image_orig = f'D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_splitted\swissimage-dop10_2021_2602-1200_0.1_2056_cropped_70.tif'
image = f'D:\MAS_DataScience\Luftbilder_Swisstopo_10_10_splitted\swissimage-dop10_2021_2602-1200_0.1_2056_cropped_70_cm1.tif'

# read image and get minx, maxy and pix_size
dataset = gdal.Open(image_orig, gdal.GA_ReadOnly)
geotrans = dataset.GetGeoTransform()

minx = geotrans[0]
maxy = geotrans[3]

pix_size = geotrans[1]

# make prediction
results = model.predict(source=image, line_width=1, show_labels=False, conf=0.05, imgsz=992)

for result in results:
    # get box coordinates, height, width
    xywh = result.boxes.xywh
    xywh_list = xywh.tolist()
    x = [i[0] for i in xywh_list]
    y = [i[1] for i in xywh_list]
    w = [i[2] for i in xywh_list]
    h = [i[3] for i in xywh_list]

    # get confidence
    conf = result.boxes.conf
    conf_list = conf.tolist()

    # get class
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

    classes = {0: 'Abwasser-eckig', 1: 'Abwasser-rund', 2: 'Abwasser-Einlaufschacht-eckig',
               3: 'Abwasser-Einlaufschacht-rund', 4: 'andere-eckig', 5: 'andere-rund'}

    df['class_name'] = df['class'].map(classes)
    df['id'] = df.index + 1

    df_prediction = df[['id', 'class_name', 'x', 'y', 'confidence']]

    # save result as csv
    df_prediction.to_csv(r"D:\MAS_DataScience\Test\test_prediction.csv", sep=';', index=False)