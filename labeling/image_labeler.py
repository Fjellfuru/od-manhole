from pathlib import Path
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text


class ImageLabeler:
    def __init__(self, dest_dir: str):
        self.dest_dir = dest_dir
        self.engine = create_engine("postgresql+psycopg2://postgres@localhost/mas_ds")
        self.conn = self.engine.connect()

    def create_label_file(self):
        query_image = text("SELECT CONCAT(image, '_cropped_', image_nr) AS image FROM public.grid WHERE corrected = 1")
        not_required_cell = pd.read_sql_query(query_image, self.conn)

        image_list = not_required_cell.values.tolist()

        for i in image_list:
            image = i[0]
            query = text(f"""SELECT label_id, nxpos_center_px, nypos_center_px, nwidth_px, nheigth_px "
                         "FROM public.v_manhole_label_yolo WHERE corrected = 1 AND image = '{image}'""")
            image_labels = pd.read_sql_query(query, self.conn)

            # label-file for orig-image
            label_file = Path(self.dest_dir) / f"{image}{'.txt'}"
            image_labels.to_csv(label_file, sep=' ', header=False, index=False)

            # label-file for gray-image
            label_file_gray = Path(self.dest_dir) / f"{image}{'_gray'}{'.txt'}"
            image_labels.to_csv(label_file_gray, sep=' ', header=False, index=False)

            # label-file for bgr-image
            label_file_bgr = Path(self.dest_dir) / f"{image}{'_bgr'}{'.txt'}"
            image_labels.to_csv(label_file_bgr, sep=' ', header=False, index=False)

            # label-file for gbr-image
            label_file_grb = Path(self.dest_dir) / f"{image}{'_grb'}{'.txt'}"
            image_labels.to_csv(label_file_grb, sep=' ', header=False, index=False)

            # label-file for cp2-image
            label_file_cp2 = Path(self.dest_dir) / f"{image}{'_cp2'}{'.txt'}"
            image_labels.to_csv(label_file_cp2, sep=' ', header=False, index=False)

            # label-file for cm1-image
            label_file_cm1 = Path(self.dest_dir) / f"{image}{'_cm1'}{'.txt'}"
            image_labels.to_csv(label_file_cm1, sep=' ', header=False, index=False)

            # label-file for sh-image
            label_file_sh = Path(self.dest_dir) / f"{image}{'_sh'}{'.txt'}"
            image_labels.to_csv(label_file_sh, sep=' ', header=False, index=False)

            # label-file for escp2-image
            label_file_escp2 = Path(self.dest_dir) / f"{image}{'_escp2'}{'.txt'}"
            image_labels.to_csv(label_file_escp2, sep=' ', header=False, index=False)

        print('label-files created')


