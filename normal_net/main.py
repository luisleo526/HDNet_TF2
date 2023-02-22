import glob
import numpy as np
import argparse
import cv2

import sys
sys.path.insert(0,'./lib/')
from lib.norm_pred import predNormals

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-imgdir", type=str, default='./example_data/color/')
    # '+90' means rotating clockwise, '-90' means rotating counterclockwise, '0' means unchanged.
    parser.add_argument("-rotate", type=str, default='0')
    parser.add_argument("-format", type=str, default='jpg')
    args = parser.parse_args()

    img_dir = args.imgdir
    rotate = args.rotate
    format = args.format
    colorname = '{}*.'+format

    imgs = glob.glob(colorname.format(img_dir))
    for img_path in imgs:
        norm_output, vis = predNormals(img_path, ifRotate=rotate)

        normal_path = img_path.replace('color','normal')
        normalimg_path = normal_path
        normalimg_path = normalimg_path.replace(format, 'png')
        normal_path = normal_path.replace(format, 'npy')
        cv2.imwrite(normalimg_path, vis)
        np.save(normal_path, norm_output)

        print(img_path + " done")