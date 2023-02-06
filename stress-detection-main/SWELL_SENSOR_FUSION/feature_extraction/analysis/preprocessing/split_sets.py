import os
import shutil

xx = "/mnt/sdb/thesis/SWELL/results_ecg_spec"
d1 = "data_2"

finpath_n = "/mnt/sdb/thesis/SWELL/datasets/" + d1 + "/no_stress"
finpath_s = "/mnt/sdb/thesis/SWELL/datasets/" + d1 + "/stress"

os.makedirs(finpath_n, exist_ok=True)
os.makedirs(finpath_s, exist_ok=True)

pp = [x for x in os.listdir(xx)]

for id in pp:
    data = id.split("_")[1]
    if data in ["cI", "cT"]:
        pat = finpath_s
    else:
        pat = finpath_n
    src = xx + "/" + id
    dst = pat + "/" + id
    shutil.copyfile(src, dst)


import random

d1 = "/mnt/sdb/thesis/SWELL/datasets/dataset_3"

data_211 = d1 + "/train/no_stress"
os.makedirs(data_211, exist_ok=True)
data_212 = d1 + "/train/stress"
os.makedirs(data_212, exist_ok=True)

data_221 = d1 + "/eval/no_stress"
os.makedirs(data_221, exist_ok=True)
data_222 = d1 + "/eval/stress"
os.makedirs(data_222, exist_ok=True)


lx1 = os.listdir(finpath_n)

lx2_ = os.listdir(finpath_s)
lx2 = random.sample(lx2_, len(lx1))



data_split = 0.85
px1 = int(data_split*len(lx1))
px2 = int(data_split*len(lx2))

sx1 = random.sample(lx1, px1)
sx2 = random.sample(lx2, px1)


## Organise stress samples
for i, im in enumerate(sx1):
    src  = finpath_n + "/" + im
    dst = data_212 + "/" + im
    shutil.copyfile(src, dst)
for i, im in enumerate(lx1):
    if im in sx1: continue
    src  = finpath_n + "/" + im
    dst = data_222 + "/" + im
    shutil.copyfile(src, dst)

## Organise neutral samples
for i, im in enumerate(sx2):
    src  = finpath_s + "/" + im
    dst = data_211 + "/" + im
    shutil.copyfile(src, dst)
for i, im in enumerate(lx2):
    if im in sx2: continue
    src  = finpath_s + "/" + im
    dst = data_221 + "/" + im
    shutil.copyfile(src, dst)


