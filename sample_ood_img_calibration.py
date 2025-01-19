import json
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4" 
os.environ['http_proxy'] = '10.106.130.4:3128'
os.environ['https_proxy'] = '10.106.130.4:3128'
import time
import socket
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import MultivariateNormal
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pixel_generator.mage import models_mage
from PIL import Image
from imagenet_clstolabel import IMGNET_CLASS2LABEL
from IPython.display import display
import lpips


torch.manual_seed(7)
np.random.seed(7)

# imagenet100 class idx
base_count = 0
in_100_idx = [6, 8, 18, 21, 23, 42, 78, 85, 94, 105, 
              107, 113, 124, 130, 148, 151, 272, 281, 288, 291, 
              292, 301, 308, 310, 311, 323, 327, 331, 333, 337, 
              340, 341, 345, 352, 357, 360, 366, 388, 394, 401, 
              403, 407, 411, 414, 417, 420, 425, 429, 430, 437, 
              446, 462, 470, 483, 488, 492, 497, 498, 516, 525, 
              526, 538, 541, 549, 561, 562, 578, 587, 608, 609, 
              616, 620, 621, 624, 629, 643, 646, 650, 651, 657, 
              677, 711, 721, 742, 748, 764, 774, 783, 819, 827, 
              843, 851, 866, 872, 879, 889, 902, 913, 937, 949]

def viz_torchimage(image):
    image = torch.clamp(image, 0, 1)
    image_np = image.detach().cpu().numpy().transpose([1, 2, 0])
    image_np = Image.fromarray(np.uint8(image_np*255))
    display(image_np)

def save_torchimg(image, class_label):
    global base_count
    image = torch.clamp(image, 0, 1)
    image_np = image.detach().cpu().numpy().transpose([1, 2, 0])
    image_np = Image.fromarray(np.uint8(image_np*255))
    save_path = os.path.join('outputs/ood_samples_val/', str(in_100_idx.index(class_label)))
    os.makedirs(save_path, exist_ok=True)
    class_name = IMGNET_CLASS2LABEL[class_label].split(',')[0]
    image_np.save(os.path.join(save_path, f'{class_name}_{base_count:05}.png'))
    base_count += 1


# --------------------------- 1. initialize arguments -------------------------- #
# Initialize RCG-L
class_cond = True
if class_cond:
    rdm_ckpt_path = 'final_ckpts/rdm-mocov3vitl-clscond.pth'
    rdm_cfg = 'config/rdm/mocov3vitl_simplemlp_l12_w1536_classcond.yaml'
else:
    rdm_ckpt_path = 'final_ckpts/rdm-mocov3vitl.pth'
    rdm_cfg = 'config/rdm/mocov3vitl_simplemlp_l12_w1536.yaml'

# minority guidance args
mg_kwargs = {
    'use_ms_grad': True,
    'norm_for_mg': 2.0,
    't_mid': -1.0,  # -1.0 for no early stop
    'mg_scale': 0.15,
    'p_ratio': 0.5,
    'num_mc_samples': 1,
    'mg_scale_type': 'var',
    'use_normed_grad': True,
    'use_lpips': False,
    'inter_rate': 1,
}
if mg_kwargs['use_lpips']:
    loss_lpips = lpips.LPIPS(net='alex').cuda()
    mg_kwargs['loss_lpips'] = loss_lpips

# mage infer args
n_image_to_gen = 1
rdm_steps = 250
rdm_eta = 1.0
mage_temp = 11.0
mage_steps = 20
cfg = 6.0  # 6.0


# ---------------------------- 2. initialize model --------------------------- #
model = models_mage.mage_vit_large_patch16(mask_ratio_mu=0.75, mask_ratio_std=0.25,
                                           mask_ratio_min=0.5, mask_ratio_max=1.0,
                                           vqgan_ckpt_path='vqgan-ckpts/vqgan_jax_strongaug.ckpt',
                                           use_rep=True, rep_dim=256, rep_drop_prob=0.1,
                                           use_class_label=False,
                                           pretrained_enc_arch='mocov3_vit_large',
                                           pretrained_enc_path='pretrained_enc_ckpts/mocov3/vitl.pth.tar',
                                           pretrained_enc_proj_dim=256,
                                           pretrained_enc_withproj=True,
                                           pretrained_rdm_ckpt=rdm_ckpt_path,
                                           pretrained_rdm_cfg=rdm_cfg,
                                           mg_kwargs=mg_kwargs)
checkpoint = torch.load(os.path.join('final_ckpts/mage-l.pth'), map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=True)
model.cuda()
_ = model.eval()


# ------------------------------- 3. mage infer ------------------------------ #
if class_cond:
    for _ in range(50):
        for class_label in in_100_idx[75:100]:  # [1, 323, 985], [:25], [25:50], [50:75], [75:100]
            print("{}: {}".format(class_label, IMGNET_CLASS2LABEL[class_label]))
            class_label = class_label * torch.ones(1).cuda().long()
            gen_images, _ = model.gen_image(1, num_iter=mage_steps, choice_temperature=mage_temp, sampled_rep=None, rdm_steps=rdm_steps, eta=rdm_eta, cfg=cfg, class_label=class_label)
            # viz_torchimage(gen_images[0])
            save_torchimg(gen_images[0], class_label.item())

else:
    for i in range(n_image_to_gen):
        gen_images, _ = model.gen_image(1, num_iter=mage_steps, choice_temperature=mage_temp, sampled_rep=None, rdm_steps=rdm_steps, eta=rdm_eta, cfg=cfg, class_label=None)
        viz_torchimage(gen_images[0])
