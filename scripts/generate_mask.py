import os
import cv2
import numpy as np
import os.path as osp

r=0.95

in_path = '../dataset/test_set/test_sdr'
out_path = '../dataset/test_set/test_sdr_mask'

if not osp.exists(out_path):
    os.mkdir(out_path)

for filename in sorted(os.listdir(in_path)):
    img_LQ = cv2.imread(osp.join(in_path, filename), -1)
    img_LQ = img_LQ.astype(np.float32) / 255.

    H, W, C = img_LQ.shape
    if H%32!=0 or W%32!=0:
        H_new = int(np.ceil(H / 32) * 32)
        W_new = int(np.ceil(W / 32) * 32)
        img_LQ = cv2.resize(img_LQ, (W_new, H_new))

    mask = np.max(img_LQ, 2)
    mask = np.minimum(1.0, np.maximum(0, mask - r) / (1 - r))
    # mask = np.expand_dims(mask, 2).repeat(C, axis=2)
    
    np.save(osp.join(out_path, filename[:-4]+'.npy'), mask)
    # print(filename)
