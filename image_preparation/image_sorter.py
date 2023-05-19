import os
import shutil
from pathlib import Path
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text


class ImageSorter:
    def __init__(self):
        self.engine = create_engine("postgresql+psycopg2://postgres@localhost/mas_ds")
        self.conn = self.engine.connect()

    def _move_prepared_image(self, image_list: list, source_dir: str, dest_dir: str):

        for i in image_list:
            name = i
            source_dir = source_dir
            dest_dir = dest_dir
            source_file = os.path.join(source_dir, name)
            if os.path.isfile(source_file):
                dest_file = os.path.join(dest_dir, name)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                shutil.move(source_file, dest_file)


    def move_not_required_image(self, source_dir: str, dest_dir: str):
        query = text("SELECT CONCAT(image, '_cropped_', image_nr, '.tif') FROM public.grid WHERE required = 1")
        not_required_cell = pd.read_sql_query(query, self.conn)
        image_orig_list = not_required_cell.values.tolist()

        image_list = []

        for i in image_orig_list:
            image = i[0]

            # image name for orig-image
            image_orig = f"{image}{'.tif'}"

            # add all image names to image-list
            image_list.append(image_orig)

        self._move_prepared_image(image_list, source_dir, dest_dir)

    def move_corrected_image(self, source_dir:str, dest_dir:str):
        query = text("SELECT CONCAT(image, '_cropped_', image_nr) FROM public.grid WHERE corrected = 1")
        corrected_grid_cell = pd.read_sql_query(query, self.conn)

        image_orig_list = corrected_grid_cell.values.tolist()
        self.conn.close()

        image_list = []

        for i in image_orig_list:
            image = i[0]

            # image name for orig-image
            image_orig = f"{image}{'.tif'}"

            # image name for gray-image
            image_gray = f"{image}{'_gray'}{'.tif'}"

            # image name for bgr-image
            image_bgr = f"{image}{'_bgr'}{'.tif'}"

            # image name for gbr-image
            image_grb = f"{image}{'_grb'}{'.tif'}"

            # image name for cp2-image
            image_cp2 = f"{image}{'_cp2'}{'.tif'}"

            # image name for cm1-image
            image_cm1 = f"{image}{'_cm1'}{'.tif'}"

            # image name for sh-image
            image_sh = f"{image}{'_sh'}{'.tif'}"

            # image name for escp2-image
            image_escp2 = f"{image}{'_escp2'}{'.tif'}"

            # add all image names to image-list
            image_list.append(image_orig)
            image_list.append(image_gray)
            image_list.append(image_bgr)
            image_list.append(image_grb)
            image_list.append(image_cp2)
            image_list.append(image_cm1)
            image_list.append(image_sh)
            image_list.append(image_escp2)

        print(len(image_list))
        self._move_prepared_image(image_list, source_dir, dest_dir)



