import json
import numpy as np
import sys
sys.path.append('/root/rcg')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" 
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
# from IPython.display import display
import lpips
from tqdm import tqdm


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
inv_rate = 0.6

in_100_path = [
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01514859/ILSVRC2012_val_00012528.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01514859/ILSVRC2012_val_00032169.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01514859/ILSVRC2012_val_00025324.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01514859/ILSVRC2012_val_00042568.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01514859/ILSVRC2012_val_00008125.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01514859/ILSVRC2012_val_00041334.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01514859/ILSVRC2012_val_00001114.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01514859/ILSVRC2012_val_00026508.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01514859/ILSVRC2012_val_00028107.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01514859/ILSVRC2012_val_00045547.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01514859/ILSVRC2012_val_00039943.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01514859/ILSVRC2012_val_00041852.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01514859/ILSVRC2012_val_00031481.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01514859/ILSVRC2012_val_00010162.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01514859/ILSVRC2012_val_00014917.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01608432/ILSVRC2012_val_00020694.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01608432/ILSVRC2012_val_00046092.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01608432/ILSVRC2012_val_00013393.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01608432/ILSVRC2012_val_00001813.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01608432/ILSVRC2012_val_00031676.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01608432/ILSVRC2012_val_00036125.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01608432/ILSVRC2012_val_00025636.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01608432/ILSVRC2012_val_00013095.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01608432/ILSVRC2012_val_00033214.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01608432/ILSVRC2012_val_00013432.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01608432/ILSVRC2012_val_00030483.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01608432/ILSVRC2012_val_00026444.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01608432/ILSVRC2012_val_00000888.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01608432/ILSVRC2012_val_00014196.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n01608432/ILSVRC2012_val_00023128.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02123045/ILSVRC2012_val_00033490.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02123045/ILSVRC2012_val_00013673.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02123045/ILSVRC2012_val_00004902.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02123045/ILSVRC2012_val_00025650.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02123045/ILSVRC2012_val_00016240.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02123045/ILSVRC2012_val_00008876.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02123045/ILSVRC2012_val_00022853.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02123045/ILSVRC2012_val_00027695.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02123045/ILSVRC2012_val_00009921.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02123045/ILSVRC2012_val_00022080.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02123045/ILSVRC2012_val_00011015.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02123045/ILSVRC2012_val_00011142.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02123045/ILSVRC2012_val_00027013.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02123045/ILSVRC2012_val_00007006.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02123045/ILSVRC2012_val_00039877.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02510455/ILSVRC2012_val_00026307.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02510455/ILSVRC2012_val_00009025.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02510455/ILSVRC2012_val_00031896.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02510455/ILSVRC2012_val_00014901.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02510455/ILSVRC2012_val_00042615.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02510455/ILSVRC2012_val_00034442.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02510455/ILSVRC2012_val_00021267.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02510455/ILSVRC2012_val_00021488.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02510455/ILSVRC2012_val_00015426.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02510455/ILSVRC2012_val_00032206.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02510455/ILSVRC2012_val_00021336.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02510455/ILSVRC2012_val_00023272.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02510455/ILSVRC2012_val_00022314.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02510455/ILSVRC2012_val_00047796.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02510455/ILSVRC2012_val_00017598.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02701002/ILSVRC2012_val_00024658.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02701002/ILSVRC2012_val_00018957.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02701002/ILSVRC2012_val_00013686.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02701002/ILSVRC2012_val_00049889.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02701002/ILSVRC2012_val_00045969.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02701002/ILSVRC2012_val_00020826.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02701002/ILSVRC2012_val_00032219.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02701002/ILSVRC2012_val_00032293.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02701002/ILSVRC2012_val_00024948.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02701002/ILSVRC2012_val_00042503.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02701002/ILSVRC2012_val_00038491.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02701002/ILSVRC2012_val_00025159.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02701002/ILSVRC2012_val_00049302.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02701002/ILSVRC2012_val_00001910.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n02701002/ILSVRC2012_val_00041942.JPEG",
    
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n07714990/ILSVRC2012_val_00027637.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n07714990/ILSVRC2012_val_00015318.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n07714990/ILSVRC2012_val_00027503.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n07714990/ILSVRC2012_val_00016725.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n07714990/ILSVRC2012_val_00038518.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n07714990/ILSVRC2012_val_00033534.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n07714990/ILSVRC2012_val_00029722.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n07714990/ILSVRC2012_val_00014546.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n07714990/ILSVRC2012_val_00025693.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n07714990/ILSVRC2012_val_00028740.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n07714990/ILSVRC2012_val_00047538.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n07714990/ILSVRC2012_val_00048012.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n07714990/ILSVRC2012_val_00003600.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n07714990/ILSVRC2012_val_00004241.JPEG",
    "/root/datasets/ood/imagenet-benchmark/imagenet-100/val/n07714990/ILSVRC2012_val_00048833.JPEG"
]

in_100_class_label_map = {'n01514859': 1, 'n01608432': 3, 'n02123045': 17, 'n02510455': 37, 'n02701002': 41, 'n07714990': 98}


# def viz_torchimage(image):
#     image = torch.clamp(image, 0, 1)
#     image_np = image.detach().cpu().numpy().transpose([1, 2, 0])
#     image_np = Image.fromarray(np.uint8(image_np*255))
#     display(image_np)

def save_torchimg(image, class_label):
    global base_count
    image = torch.clamp(image, 0, 1)
    image_np = image.detach().cpu().numpy().transpose([1, 2, 0])
    image_np = Image.fromarray(np.uint8(image_np*255))
    save_path = os.path.join('outputs/inversion_ood/', f'inv{inv_rate}', str(class_label))
    os.makedirs(save_path, exist_ok=True)
    class_name = IMGNET_CLASS2LABEL[in_100_idx[class_label]].split(',')[0]
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

for img_path in tqdm(in_100_path):
    image = Image.open(img_path)
    image_tensor = transform(image).unsqueeze(0).cuda()
    rep = model.pretrained_encoder.forward_features(image_tensor)
    import pdb; pdb.set_trace()
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
    raw_label = img_path.split('/')[-2]
    class_label = in_100_class_label_map[raw_label] * torch.ones(1).cuda().long()
    cond = rdm.get_learned_conditioning({"class_label": class_label})
    
    for i, step in enumerate(time_range):
        index = total_steps - i - 1
        ts = torch.full((1,), step, device=rep.device, dtype=torch.long)
        # r_t, _ = model.rdm_sampler.p_sample_ddim(r_t, cond, ts, index)
        r_t, _ = rdm_sampler.p_sample_ddim(r_t, cond, ts, index)
    # import pdb; pdb.set_trace()
    r_t = r_t.squeeze(-1).squeeze(-1)
    for _ in range(10):
        gen_images, _ = model.gen_image(1, num_iter=mage_steps, choice_temperature=mage_temp, sampled_rep=r_t, rdm_steps=rdm_steps, eta=rdm_eta, cfg=cfg, class_label=class_label)
        save_torchimg(gen_images[0], class_label.item())
    
# # ------------------------------- 3. mage infer ------------------------------ #
# if class_cond:
#     for _ in range(1000):
#         for class_label in in_100_idx[75:100]:  # [1, 323, 985], [:25], [25:50], [50:75], [75:100]
#             print("{}: {}".format(class_label, IMGNET_CLASS2LABEL[class_label]))
#             class_label = class_label * torch.ones(1).cuda().long()
#             gen_images, _ = model.gen_image(1, num_iter=mage_steps, choice_temperature=mage_temp, sampled_rep=None, rdm_steps=rdm_steps, eta=rdm_eta, cfg=cfg, class_label=class_label)
#             # viz_torchimage(gen_images[0])
#             save_torchimg(gen_images[0], class_label.item())

# else:
#     for i in range(n_image_to_gen):
#         gen_images, _ = model.gen_image(1, num_iter=mage_steps, choice_temperature=mage_temp, sampled_rep=None, rdm_steps=rdm_steps, eta=rdm_eta, cfg=cfg, class_label=None)
#         viz_torchimage(gen_images[0])
