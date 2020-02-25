import pandas as pd
import os
from pathlib import Path
import shutil
import cv2 
import numpy as np

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
    # read through pq files
    for i in range(4):
        # read current pq file
        # current_pq = pd.read_parquet(f'{data_dir}/{split}_image_data_{i}.parquet')
        current_pq = pd.read_parquet("/home/alex/Desktop/deep_learning/midterm/bengali-char-recognition/data/train_toy.parquet")
        print(current_pq.head())
        # do magical preprocessing
        for row in current_pq.iterrows():
            ID = row[1]["image_id"]
            key_row = key_df[key_df.image_id == ID]
            labels = (key_row["grapheme_root"],
                      key_row["vowel_diacritic"],
                      key_row["consonant_diacritic"])
            im = row[1].drop("image_id").to_numpy()
            # im = row[1].drop("image_id").to_numpy().reshape(127,97)
            print(im)
            print(im.shape)
            break
            cv2.imwrite(f'{split_dir}/{ID}.png', im)
            with open(f'{split_dir}/{ID}.text', "w") as text_file:
                print(f"{labels}", file=text_file)
            del im
            del labels
            del key_row
            del ID
        break
        del current_pq
    del key_df
    del split_dir_path
    del split_dir
    break
        
