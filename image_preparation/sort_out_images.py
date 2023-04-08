import os
import shutil
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text


def move_image(source_dir, dest_dir):
    engine = create_engine("postgresql+psycopg2://postgres@localhost/mas_ds")
    conn = engine.connect()
    query = text("SELECT CONCAT(image, '_cropped_', image_nr, '.tif') FROM public.grid WHERE required = 1")
    not_required_cell = pd.read_sql_query(query, conn)

    result_list = not_required_cell.values.tolist()

    for result in result_list:
        name = result[0]
        source_dir = source_dir
        dest_dir = dest_dir
        source_file = os.path.join(source_dir, name)
        if os.path.isfile(source_file):
            dest_file = os.path.join(dest_dir, name)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            shutil.move(source_file, dest_file)

    conn.close()

