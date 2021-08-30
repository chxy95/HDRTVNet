import random
import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import os.path as osp

class LQGT_dataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

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

        self.cond_folder = opt['dataroot_cond']

    def __getitem__(self, index):
        GT_path, LQ_path = None, None
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        img_GT = util.read_img(None, GT_path)

        # get LQ image
        LQ_path = self.paths_LQ[index]
        img_LQ = util.read_img(None, LQ_path)

        # # get condition
        cond_scale = self.opt['cond_scale']
        if self.cond_folder is not None:
            if '_' in osp.basename(LQ_path):
                cond_name = '_'.join(osp.basename(LQ_path).split('_')[:-1])+'_bicx'+str(cond_scale)+'.png'
            else: cond_name = osp.basename(LQ_path).split('.')[0]+'_bicx'+str(cond_scale)+'.png'
            cond_path = osp.join(self.cond_folder, cond_name)
            cond_img = util.read_img(None, cond_path)
        else:
            cond_img = util.imresize_np(img_LQ, 1/cond_scale)

        if self.opt['phase'] == 'train':

            H, W, C = img_LQ.shape
            H_gt, W_gt, C = img_GT.shape
            if H != H_gt:
                print('*******wrong image*******:{}'.format(LQ_path))

            # randomly crop
            if GT_size is not None:
                LQ_size = GT_size
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                img_GT = img_GT[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
            cond_img = cond_img[:, :, [2, 1, 0]]

        H, W, _ = img_LQ.shape
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        cond = torch.from_numpy(np.ascontiguousarray(np.transpose(cond_img, (2, 0, 1)))).float()

        if LQ_path is None:
            LQ_path = GT_path
            
        return {'LQ': img_LQ, 'GT': img_GT, 'cond': cond, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)
