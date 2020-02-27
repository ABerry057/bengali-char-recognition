import pandas as pd
import os
from pathlib import Path
import shutil
import cv2 
import numpy as np
import pyarrow.parquet as pq

parent_dir = os.path.dirname(os.getcwd())
data_dir = f'{parent_dir}/data'

for split in ["train", "test"]:
    split_dir = f'{data_dir}/{split}'
    split_dir_path = Path(split_dir)
    # remove dir if already exists
    if split_dir_path.exists():
        shutil.rmtree(split_dir)
    # now create split dir
    split_dir_path.mkdir(parents=True)
    # read key table
    key_df = pd.read_csv(f'{data_dir}/{split}.csv')
    dir_pairs = 0
    # read through pq files
    for i in range(4):
        # read current pq file, already proccessed
        current_pq = pq.read_table(f'{data_dir}/{split}_image_data_{i}.parquet').to_pandas()

        for row in current_pq.iterrows():
            ID = row[1]["image_id"]
            key_row = key_df[key_df.image_id == ID]
            labels = (key_row["grapheme_root"].values[0],
                      key_row["vowel_diacritic"].values[0],
                      key_row["consonant_diacritic"].values[0])
            im = row[1].drop("image_id").to_numpy().reshape(87,106) * 255.0
            cv2.imwrite(f'{split_dir}/{ID}.png', np.float32(im))
            with open(f'{split_dir}/{ID}.text', "w") as text_file:
                print(f"{labels}", file=text_file)
            dir_pairs += 1
            del im
            del labels
            del key_row
            del ID
        del current_pq
    del key_df
    del split_dir_path
    del split_dir
    print(f"Created {split} directory, containing {dir_pairs} image-label pairs")
    break
        
