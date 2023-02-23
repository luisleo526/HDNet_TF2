# affine transform part based on https://github.com/xingyizhou/pytorch-pose-hg-3d/blob/master/src/lib/utils/image.py
import mediapipe as mp
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import os


# Remove background
def remove_bg(img_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5)

    rgb_im = cv2.imread(img_path)
    rgb = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
    rgb_mask_res = pose.process(rgb)
    rgb_mask = rgb_mask_res.segmentation_mask

    mask = rgb_mask.copy()
    mask = (mask * 255.).astype(np.uint8)
    mask[mask >= 50] = 255
    mask[mask <= 50] = 0
    mask = (mask / 255.).astype(np.uint8)

    save_mask = mask.copy()
    save_mask[save_mask == 1] = 255
    suffix = img_path.split('.')[-1]
    mask_path = img_path.replace('color', 'mask')
    mask_path = mask_path.replace('rgb', 'mask')
    mask_path = mask_path.replace(suffix, 'png')
    if not os.path.exists(mask_path.split('\\')[0]):
        os.makedirs(mask_path.split('\\')[0])
    cv2.imwrite(mask_path, np.uint8(save_mask))

    return mask


# Find bounding box
def find_bbx(mask, color, buffer=25):
    height, width = color.shape[0:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = np.concatenate(cnts)
    x, y, w, h = cv2.boundingRect(cnts)

    buffer = 25
    h_upper = np.max((0, y - buffer))
    h_bottom = np.min((height, y + h + buffer))
    w_left = np.max((0, x - buffer))
    w_right = np.min((width, x + w + buffer))

    mask_bbx = mask[h_upper:h_bottom, w_left:w_right]
    color_bbx = color[h_upper:h_bottom, w_left:w_right, :]

    return mask_bbx, color_bbx, [h_upper, h_bottom, w_left, w_right]


# get affine transform
def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans