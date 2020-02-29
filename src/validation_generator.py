import os
import glob
import shutil
from pathlib import Path
import random

random.seed(19)

parent_dir = os.path.dirname(os.getcwd())
data_dir = str(parent_dir) + '/data'

images = sorted(glob.glob(data_dir + "/train/*.png"))
txts = sorted(glob.glob(data_dir + "/train/*.txt"))

valid_dir = data_dir + "/validation"
valid_dir_path = Path(valid_dir)
# remove dir if already exists
if valid_dir_path.exists():
    shutil.rmtree(valid_dir)
# now create valid dir
valid_dir_path.mkdir(parents=True)

move_count = 0
for i in range(len(images)):
    image = images[i]
    text = txts[i]
    condition_number = random.randrange(0,5)
    if condition_number == 0:
        #shutil.move: src is moved to dest dir
        shutil.move(image, valid_dir)
        shutil.move(text, valid_dir)
        move_count += 1
print("Moved " + str(move_count) + " image-text pairs into validation directory")

