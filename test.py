"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config
from trainer import UNIT_Trainer
# from trainer_new import UNIT_Trainer
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os,cv2
from torchvision import transforms
from PIL import Image
# from skimage.measure import compare_psnr, compare_ssim
import numpy as np
import scipy.io as sio
from networks import make_mask, downsampling
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/unit_noise2clear-bn.yaml', help="net configuration")
parser.add_argument('--input', type=str, default = '/data/simulation/test/motion/', help="input image path")

parser.add_argument('--output_folder', type=str, default='/result/', help="output image path")
parser.add_argument('--checkpoint', type=str, default='/checkpoints/gen_00070000.pt',
                    help="checkpoint of autoencoders")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--trainer', type=str, default='UNIT', help="MUNIT|UNIT")
opts = parser.parse_args()


torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)

# Setup model and data loaderopts.trainer == 'UNIT':
trainer = UNIT_Trainer(config)
state_dict = torch.load(opts.checkpoint, map_location='cpu')
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])
# trainer.gen_mask.load_state_dict(state_dict['mask'])

# trainer.cuda()
trainer.cuda(trainer.gpuid)
trainer.eval()
encode = trainer.gen_a.encode_cont  # encode function
decode = trainer.gen_b.dec_cont  # decode function

for motion_name in sorted(os.listdir(opts.input)):
    motion_path = opts.input + motion_name

    motion_mat = sio.loadmat(motion_path)
    # motion = motion_mat['test_motion']
    motion = motion_mat['dicomV1']
    # motion = motion_mat['data']
    motion = np.float32(motion)
    # motion = np.transpose(motion, [2,0,1])

    num = len(motion)

    mid = np.zeros((trainer.N, motion.shape[-2], motion.shape[-1]))

    # out = np.zeros_like(motion[:,1,:,:])
    out = np.zeros_like(motion)

    for i in range(num):
        print(i)
        motion_slice_tem = motion[i,:,:]


        motion_slice = np.zeros(motion_slice_tem.shape, dtype=np.float32)
        cv2.normalize(motion_slice_tem, motion_slice, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        motion_slice = np.expand_dims(np.expand_dims(motion_slice,0),0)

        input = torch.from_numpy(motion_slice).cuda(trainer.gpuid)
        # input = torch.from_numpy(motion_slice).cuda()

        for j in range(trainer.N):
            mask_test = make_mask(input, trainer.R, trainer.gpuid)
            input_d = downsampling(input, mask_test)
            # content = encode(input_d)
            content = encode(input)
            outputs = decode(content)

            final = outputs.data.cpu().numpy().squeeze()


            mid[j,:,:] = final[:,:]

        out[i,:,:] = np.mean(mid, axis=0)
    out = np.float64(out)
    sio.savemat(os.path.join(opts.output_folder, motion_name),{'corrected':out})
