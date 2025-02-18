# %%
import json
import numpy as np
import sys
sys.path.append('/root/rcg')
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
import lpips
from tqdm import tqdm


# in_100_class_label_map = {'n01514859': 1, 'n01608432': 3, 'n02123045': 17, 'n02510455': 37, 'n02701002': 41, 'n07714990': 98}


# def viz_torchimage(image):
#     image = torch.clamp(image, 0, 1)
#     image_np = image.detach().cpu().numpy().transpose([1, 2, 0])
#     image_np = Image.fromarray(np.uint8(image_np*255))
#     display(image_np)

def save_torchimg(image_batch, class_label_batch):
    global base_count
    for image, class_label in zip(image_batch, class_label_batch):
        image = torch.clamp(image, 0, 1)
        image_np = image.detach().cpu().numpy().transpose([1, 2, 0])
        image_np = Image.fromarray(np.uint8(image_np*255))
        save_path = os.path.join('outputs/inversion_ood/', f"inv{inv_rate}_mg{mg_kwargs['mg_scale']}",
                                 str(in_100_idx.index(class_label.item())))
        os.makedirs(save_path, exist_ok=True)
        # class_name = IMGNET_CLASS2LABEL[in_100_idx[class_label]].split(',')[0]
        class_name = IMGNET_CLASS2LABEL[class_label.item()].split(',')[0]
        image_np.save(os.path.join(save_path, f'{class_name}_{base_count:05}.png'))
        base_count += 1

# %%
# --------------------------- 1. initialize arguments -------------------------- #
# imagenet100 class idx
inv_rate = 0.3
bsz = 5
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

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
    'mg_scale': 0.1,
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

# %%
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
                                           mg_kwargs=mg_kwargs)  # None
checkpoint = torch.load(os.path.join('final_ckpts/mage-l.pth'), map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=True)
model.cuda()
_ = model.eval()

# ---------------------------------------------------------------------------- #
# 反演得到表征
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# load ID data
train_data_in = datasets.ImageFolder(
    os.path.join('/root/datasets/ood/imagenet-benchmark/imagenet-100/', 'train'),
    transform=transform)
train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=bsz, shuffle=True,
    num_workers=4, pin_memory=True)

dict_values = torch.tensor(in_100_idx, dtype=torch.long)
for img, label in tqdm(train_loader_in):
    image_tensor = img.cuda()
    # class_label = in_100_idx[label.item()] * torch.ones(1).cuda().long()
    class_label = dict_values[label].cuda()
    
    rep = model.pretrained_encoder.forward_features(image_tensor)
    # import pdb; pdb.set_trace()
    if model.pretrained_enc_withproj:
        rep = model.pretrained_encoder.head(rep)
    rep_std = torch.std(rep, dim=1, keepdim=True)
    rep_mean = torch.mean(rep, dim=1, keepdim=True)
    rep = (rep - rep_mean) / rep_std
    
    # import pdb; pdb.set_trace()
    rdm = model.rdm_sampler.model
    rdm_sampler = model.rdm_sampler
    
    
    # get ddim timestep
    rdm_sampler.make_schedule(ddim_num_steps=rdm_steps, ddim_eta=rdm_eta, verbose=False)
    timesteps = rdm_sampler.ddim_timesteps[:int(len(rdm_sampler.ddim_timesteps)*inv_rate)]
    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]
    # print('total_steps:', total_steps)
    
    # import pdb; pdb.set_trace()
    tp = timesteps[-1] * torch.ones((1, )).long().cuda()
    r_t = rdm.q_sample(rep, tp).unsqueeze(-1).unsqueeze(-1)
    
    # get cond
    cond = rdm.get_learned_conditioning({"class_label": class_label})
    
    for i, step in enumerate(time_range):
        index = total_steps - i - 1
        ts = torch.full((1,), step, device=rep.device, dtype=torch.long)
        # r_t, _ = model.rdm_sampler.p_sample_ddim(r_t, cond, ts, index)
        r_t, _ = rdm_sampler.p_sample_ddim(r_t, cond, ts, index)
    # import pdb; pdb.set_trace()
    r_t = r_t.squeeze(-1).squeeze(-1)
    
    gen_images, _ = model.gen_image(bsz, num_iter=mage_steps, choice_temperature=mage_temp, sampled_rep=r_t, rdm_steps=rdm_steps, eta=rdm_eta, cfg=cfg, class_label=class_label)
    # import pdb; pdb.set_trace()
    save_torchimg(gen_images, class_label)
    
