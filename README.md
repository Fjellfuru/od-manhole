# Manhole cover detection

The project manhole detection (od_manhole = object detection manhole)
shows a way to detect manhole covers from the newest swisstopo aerial images with YOLOv8. The project was created as part of my master's thesis. 
The project contains the labeling-process, a notebook for training, some scripts to compare 
different trained models and a script for manhole cover prediction in new images.

## Labeling-Process
The aim of the labeling process is to generate labeled aerial images that can be used for training and validation.
### What do you need
 * PosrgreSQL 15 incl. PostGIS 3.3.2
 * QGIS 3.28.4 
 * FME 2022.1.3
 * Python 3.9
 * swisstopo aerial images
 * if available, manhole cover data

### Peparation and Workflow
1. Install PosrgreSQL incl. PostGIS
2. Create database with https://github.com/Fjellfuru/od-manhole/blob/master/utilities/database/od_manhole_db_schema.backup
3. Download some swisstopo aerial images
4. Import manhole cover data into the database (public.manhole_point) or manually digitize manhole covers (e.g. with QGIS https://github.com/Fjellfuru/od-manhole/blob/master/utilities/qgis_project/od_manhole.qgz) 
5. Crop swisstopo original images --> 9920 x 9920 pixels
6. Create grid with 100 cells per cropped original image (e.g. with QGIS) and insert into database (public.grid)
7. Intersect cropped images and grid (e.g. with FME https://github.com/Fjellfuru/od-manhole/blob/master/utilities/fme_workbench/clipp_grid_image.fmw) and update database (public.grid)
8. Update image_nr for every grid_cell (public.grid). You can find an example at https://github.com/Fjellfuru/od-manhole/blob/master/utilities/sql_scripts/Example_UPDATE_image_nr.sql
9. The following script can be used for the data augmentation, the actual labeling and the preparation of the training data set: https://github.com/Fjellfuru/od-manhole/blob/master/labeling_process.py


## Training
### What do you need
 * Google Colab
 * Goggle Drive for data management
 * Dataset created during the labeling process

### Workflow
You can find everything you need in the Jupyter Notebook https://github.com/Fjellfuru/od-manhole/blob/master/utilities/jupyter_notebook/od_manhole_yolov8-object-detection.ipynb

## Compare different traind models (Training)
To compare different trained models after experimentally training
### What do you need
 * Google Colab
 * Goggle Drive for data management
 * Results from the training
### Workflow
To compare the metrics of the different models, you can use the following script:
   * https://github.com/Fjellfuru/od-manhole/blob/master/model_comparison/model_comparison.py

## Compare different traind models (Validation)
To compare different trained models, I performed a validation with a test data set for all models.
### What do you need
 * Google Colab
 * Goggle Drive for data management
 * created during the labeling process
### Workflow
You can find the validation workflow Jupyter Notebook https://github.com/Fjellfuru/od-manhole/blob/master/utilities/jupyter_notebook/od_manhole_yolov8-object-detection.ipynb \
After validation, the predicted manhole covers can be compared to the true manhole covers. The following steps must be carried out for this:
1. Import the predicted manhole covers into the database (https://github.com/Fjellfuru/od-manhole/blob/master/model_comparison/prediction_comparison_data_preparation.py)
2. Create point geometry for predicted manhole covers (https://github.com/Fjellfuru/od-manhole/blob/master/utilities/sql_scripts/SELECT_UPDATE_geom_manhole_test_prediction.sql)
3. To compare the metrics and data augmentation methods, you can use the following scripts (Please customize the intput results):
   * https://github.com/Fjellfuru/od-manhole/blob/master/model_comparison/model_comparison.py
   * https://github.com/Fjellfuru/od-manhole/blob/master/model_comparison/prediction_comparison_plots.py


## Manhole cover prediction
### What do you need
 * Python 3.9
 * QGIS 3.28.4
 * swisstopo aerial images

### Workflow
For manhole cover detection you can use https://github.com/Fjellfuru/od-manhole/blob/master/manhole_prediction/manhole_prediction.py. \
Please customize the model you want and the input image.
You can find some models that were experimentally trained as part of the master’s thesis under https://github.com/Fjellfuru/od-manhole/tree/master/utilities/yolov8_manhole_cover_models.
