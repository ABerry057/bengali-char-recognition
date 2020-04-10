import pandas as pd
import os
from pathlib import Path
import shutil

parent_dir = os.path.dirname(os.getcwd())
data_dir = f'{parent_dir}/data'

for split in ["train"]:
    split_dir = f'{data_dir}/{split}'
    split_dir_path = Path(split_dir)
    # remove dir if already exists
    if split_dir_path.exists():
        shutil.rmtree(split_dir)
    # now create split dir
    split_dir_path.mkdir(parents=True)
pq = pd.read_parquet(f'{data_dir}/{split}_image_data_0.parquet')
pq = pq.loc[:10,:]
pq.to_parquet(f'{data_dir}/{split}_toy.parquet')