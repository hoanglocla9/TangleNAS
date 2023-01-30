import os
import shutil
import numpy as np

shutil.rmtree(
    '/work/dlclarge2/sukthank-naslib_one_shot/Cream/AutoFormer/imagenet_subsample'
)
if not os.path.exists(
        "/work/dlclarge2/sukthank-naslib_one_shot/Cream/AutoFormer/imagenet_subsample"
):
    os.mkdir(
        "/work/dlclarge2/sukthank-naslib_one_shot/Cream/AutoFormer/imagenet_subsample"
    )
if not os.path.exists(
        "/work/dlclarge2/sukthank-naslib_one_shot/Cream/AutoFormer/imagenet_subsample/train"
):
    os.mkdir(
        "/work/dlclarge2/sukthank-naslib_one_shot/Cream/AutoFormer/imagenet_subsample/train"
    )
if not os.path.exists(
        "/work/dlclarge2/sukthank-naslib_one_shot/Cream/AutoFormer/imagenet_subsample/val"
):
    os.mkdir(
        "/work/dlclarge2/sukthank-naslib_one_shot/Cream/AutoFormer/imagenet_subsample/val"
    )

for d in os.listdir(
        "/work/dlclarge2/sukthank-naslib_one_shot/Cream/AutoFormer/imagenet/train"
):
    current_dir = "/work/dlclarge2/sukthank-naslib_one_shot/Cream/AutoFormer/imagenet/train/" + d
    dir_new = "/work/dlclarge2/sukthank-naslib_one_shot/Cream/AutoFormer/imagenet_subsample/train/" + d
    if not os.path.exists(dir_new):
        os.mkdir(dir_new)
    imgs = []
    if os.path.isdir(current_dir):
        for f in os.listdir(current_dir):
            imgs.append(current_dir + "/" + f)
        num_imgs = len(imgs)
        frac = 0.1
        to_choose = int(frac * num_imgs)
        choice_array = list(
            np.random.choice(num_imgs, to_choose, replace=False))
        for c in choice_array:
            shutil.copy(imgs[c], dir_new)
            print("Copied", imgs[c])

for d in os.listdir(
        "/work/dlclarge2/sukthank-naslib_one_shot/Cream/AutoFormer/imagenet/val"
):
    current_dir = "/work/dlclarge2/sukthank-naslib_one_shot/Cream/AutoFormer/imagenet/val/" + d
    dir_new = "/work/dlclarge2/sukthank-naslib_one_shot/Cream/AutoFormer/imagenet_subsample/val/" + d
    if not os.path.exists(dir_new):
        os.mkdir(dir_new)
    imgs = []
    if os.path.isdir(current_dir):
        for f in os.listdir(current_dir):
            imgs.append(current_dir + "/" + f)
        num_imgs = len(imgs)
        frac = 0.025
        to_choose = int(frac * num_imgs)
        choice_array = list(
            np.random.choice(num_imgs, to_choose, replace=False))
        for c in choice_array:
            shutil.copy(imgs[c], dir_new)
            print("Copied", imgs[c])
