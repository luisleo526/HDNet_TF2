import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, './lib/')
from preprocess import remove_bg, find_bbx, get_affine_transform
from norm_net import define_G, load_pretrained_model


def load_network(pretrained_model_path='./pretrained_models/normal_net/netF.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netF = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance").to(device)  # normal network
    load_pretrained_model(pretrained_model_path, netF)

    return netF


def predNormals(img_path, ifRotate='0', netF=None):
    if netF is None:
        netF = load_network()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inp_color = cv2.imread(img_path)
    get_mask = remove_bg(img_path)
    mask, img, bbox = find_bbx(get_mask, inp_color)
    img[mask == 0] = 0

    # rotate
    if ifRotate == '+90':  # clockwise rotation
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    elif ifRotate == '-90':  # counterclockwise rotation
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        img = img
        mask = mask

    # transfer to RGB
    img_h, img_w, img_c = img.shape
    if img_c == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #   affain
    input_size = max(img.shape[0], img.shape[1])
    s = max(img.shape[0], img.shape[1]) * 1.0
    s = np.array([s, s])
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    trans_input = get_affine_transform(c, s, 0, [input_size, input_size])

    cropped_img = cv2.warpAffine(img, trans_input, (input_size, input_size), flags=cv2.INTER_LINEAR)
    cropped_mask = cv2.warpAffine(mask, trans_input, (input_size, input_size), flags=cv2.INTER_LINEAR)
    #     ori_shape = (cropped_img.shape[1], cropped_img.shape[0])
    cropped_img_512 = cv2.resize(cropped_img, (512, 512), interpolation=cv2.INTER_LINEAR)
    cropped_mask_512 = cv2.resize(cropped_mask, (512, 512), interpolation=cv2.INTER_LINEAR)
    cropped_mask_512 = np.expand_dims(cropped_mask_512, axis=2)

    cropped_img_512 = torch.from_numpy(cropped_img_512).float()
    cropped_img_512 = cropped_img_512.permute(2, 0, 1).unsqueeze(0)
    cropped_img_512 = cropped_img_512.to(device)

    with torch.no_grad():
        normal_img = netF(cropped_img_512)
    normal_img = normal_img.squeeze(0).permute((1, 2, 0))
    normal_img_512 = normal_img.cpu().detach().numpy()

    # normalize
    norm_output = normal_img_512.copy()
    # norm_output[:, :, 0] = (normal_img_512[:, :, 0]+1)/2
    # norm_output[:, :, 1] = (normal_img_512[:, :, 1]+1)/2
    # norm_output[:, :, 2] = (normal_img_512[:, :, 2]+1)/2
    # norm_output = np.multiply(norm_output, cropped_mask_512)

    ### back to original size
    # resize from 512x512 to input_size x input_size
    norm_output = cv2.resize(norm_output, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    # crop to fit bbox
    col_start = int((input_size - img.shape[1]) / 2.)
    col_end = int((input_size + img.shape[1]) / 2.)
    col_len = col_end - col_start
    ori_len = bbox[3] - bbox[2]
    # if(col_len!=ori_len):
    #     print("shape not match")
    #     col_end = col_end + (ori_len - col_len)
    norm_output = norm_output[:, col_start:col_end, :]
    # rotate back
    if ifRotate == '+90':
        norm_output = cv2.rotate(norm_output, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif ifRotate == '-90':
        norm_output = cv2.rotate(norm_output, cv2.ROTATE_90_CLOCKWISE)
    else:
        norm_output = norm_output
    # back to original
    norm_output_ori = np.zeros(inp_color.shape)
    norm_output_ori[bbox[0]:bbox[1], bbox[2]:bbox[3], :] = norm_output
    # masking and visualization
    mask_ = np.expand_dims(get_mask, axis=-1)
    norm_output_ori = norm_output_ori * mask_
    vis = (norm_output_ori[:, :, ::-1] + 1) / 2 * 255
    vis = (vis * mask_).astype(np.uint8)

    # mask
    segmentation_mask = np.concatenate((mask_, mask_, mask_), axis=-1) * 255

    # normalization
    norm_mag = np.expand_dims(np.sqrt(np.square(norm_output_ori).sum(axis=2)), -1)
    norm_mag[norm_mag == 0] = 1.0
    norm_output_ori = np.divide(norm_output_ori, norm_mag)

    return norm_output_ori, segmentation_mask, netF
