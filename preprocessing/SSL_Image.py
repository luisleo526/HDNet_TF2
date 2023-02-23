import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from norm_pred import predNormals

sys.path.append("/detectron2/projects/DensePose")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--pickle_path', type=str, default='data/celeba')
    parser.add_argument('--output', type=str, default='data/celeba_self_supervisied')
    parser.add_argument('--img_size', type=int, default=256)
    return parser.parse_args()


def write_matrix_txt(a, filename):
    mat = np.matrix(a)
    with open(filename, 'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.5f')


def padding_size(box_coordinates, img_shape, map_shape):
    H, W = img_shape
    Hm, Wm = map_shape
    x0, y0, x1, y1 = box_coordinates

    Ys = y0
    Ye = H - y1
    Ye = H - Hm - Ys

    Xs = x0
    Xe = W - x1
    Xe = W - Wm - Xs

    return ((Ys, Ye), (Xs, Xe))


def resize_image(img, size=(28, 28)):
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1

    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0] + size[1]) // 2 else cv2.INTER_CUBIC

    x_pos = (dif - w) // 2
    y_pos = (dif - h) // 2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)


if __name__ == '__main__':

    args = parse_args()
    args.img_size = (args.img_size, args.img_size)

    data = torch.load(args.pickle_path)

    Path(os.path.join(args.output, 'color')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output, 'color_WO_bg')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output, 'densepose')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output, 'mask')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output, 'pred_normals')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output, 'pred_normals_png')).mkdir(parents=True, exist_ok=True)

    netF = None
    for info in data:

        try:
            original_img = imageio.imread(info['file_name'])
            H, W, _ = original_img.shape
            original_img = np.array(original_img)

            u, v = (info['pred_densepose'][0].uv * 255).to(torch.uint8).cpu().numpy()
            i = (info['pred_densepose'][0].labels).to(torch.uint8).cpu().numpy()
            single_mask = ((i != 0) * 255).astype(np.uint8)

            pad_size = padding_size(info['pred_boxes_XYXY'][0].to(torch.int).tolist(), (H, W), u.shape)

            u = np.pad(u, pad_size, 'constant', constant_values=((0, 0), (0, 0)))
            v = np.pad(v, pad_size, 'constant', constant_values=((0, 0), (0, 0)))
            i = np.pad(i, pad_size, 'constant', constant_values=((0, 0), (0, 0)))
            single_mask = np.pad(single_mask, pad_size, 'constant', constant_values=((0, 0), (0, 0)))

            densepose = np.stack((v, u, i), axis=-1)
            basename = os.path.basename(info["file_name"])

            Image.fromarray(resize_image(densepose, args.img_size), "RGB").save(
                os.path.join(args.output, 'densepose', basename))
            Image.fromarray(resize_image(original_img, args.img_size), "RGB").save(
                os.path.join(args.output, 'color', basename))

            norm_output, mask, netF = predNormals(os.path.join(args.output, 'color', basename), ifRotate="0", netF=netF)
            mask_image = imageio.imread(os.path.join(args.output, 'color', basename))
            mask_image[mask == 0] = 255

            vis = np.copy((1 + norm_output) / 2)
            vis[mask == 0] = 255

            for idx in range(3):
                write_matrix_txt(norm_output[:, :, 0],
                                 os.path.join(args.output, 'pred_normals', basename + f"_{idx + 1}.txt"))

            Image.fromarray(resize_image(vis, args.img_size), "RGB").save(
                os.path.join(args.output, 'pred_normals_png', basename))
            Image.fromarray(resize_image(mask_image, args.img_size), "RGB").save(
                os.path.join(args.output, 'color_WO_bg', basename))
            Image.fromarray(resize_image(mask, args.img_size), "RGB").save(
                os.path.join(args.output, 'mask', basename))


        except Exception as e:

            print(str(e))
            print(info['file_name'])
