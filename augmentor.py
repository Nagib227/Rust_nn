import os
from os import walk
import shutil
import random
import Augmentor

path = 'datasets/rust/dataset_nn'
num_augment = 3000
first_name = 3005

os.mkdir(f"{path}/augment")
files_img = [i for i in next(walk(path), (None, None, []))[2] if ".jpg" in i]

for i in random.choices(files_img, k=num_augment):
        shutil.copy(f"{path}/{i}", f"{path}/augment/{first_name}.jpg")
        first_name += 1

p = Augmentor.Pipeline(f"{path}/augment")

p.flip_left_right(0.5)
p.rotate(0.3, 10, 10)
p.skew(0.4, 0.5)
p.zoom(probability = 0.2, min_factor = 1.1, max_factor = 1.5)
p.sample(num_augment)

