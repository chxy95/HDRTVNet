import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
import os.path as osp

class LQGT_dataset(data.Dataset):

    def __init__(self, opt):
        super(LQGT_dataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None

        self.sizes_GT, self.paths_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.sizes_LQ, self.paths_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))

        self.mask_folder = opt['dataroot_mask']

    def __getitem__(self, index):
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        img_GT = util.read_img(None, GT_path)

        # get LQ image
        LQ_path = self.paths_LQ[index]
        img_LQ = util.read_img(None, LQ_path)

        # get mask when mask folder is not None
        if self.mask_folder is not None:
            mask_name = osp.basename(LQ_path)[:-4] + '.npy'
            mask_path = osp.join(self.mask_folder, mask_name)
            mask = util.read_npy(mask_path)
            mask = np.expand_dims(mask, 2).repeat(3, axis=2)

        if self.opt['phase'] == 'train':
            
            H, W, C = img_LQ.shape
            H_gt, W_gt, C = img_GT.shape
            if H != H_gt:
                print('*******wrong image*******:{}'.format(LQ_path))
            LQ_size = GT_size // scale

            # randomly crop
            if GT_size is not None:
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # resize for alignment
        H, W, C = img_LQ.shape
        if H%32!=0 or W%32!=0:
            H_new = int(np.ceil(H / 32) * 32)
            W_new = int(np.ceil(W / 32) * 32)
            img_LQ = cv2.resize(img_LQ, (W_new, H_new))
            img_GT = cv2.resize(img_GT, (W_new, H_new))
        
        # use the input LQ to calculate the mask.
        if self.mask_folder is None:
            r = 0.95
            mask = np.max(img_LQ, 2)
            mask = np.minimum(1.0, np.maximum(0, mask - r) / (1 - r))
            mask = np.expand_dims(mask, 2).repeat(3, axis=2)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        mask = torch.from_numpy(np.ascontiguousarray(np.transpose(mask, (2, 0, 1)))).float()

        if LQ_path is None:
            LQ_path = GT_path
        return {'LQ': img_LQ, 'GT': img_GT, 'mask': mask, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)
