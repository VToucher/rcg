import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1" 
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pixel_generator.mage import models_mage
from PIL import Image
from imagenet_clstolabel import IMGNET_CLASS2LABEL
from IPython.display import display


# ----------------------------- get train loader ----------------------------- #
torch.manual_seed(0)
np.random.seed(0)
bsz = 1
class_label = 1

transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(256),
            transforms.ToTensor()])

dataset = datasets.ImageFolder('/root/datasets/imagenet1k_train', transform=transform)
print('num_data:', len(dataset))
print('num_classes:', len(dataset.classes))
data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=bsz,
        num_workers=10,
        pin_memory=True,
        drop_last=True,
)
data_iterator = enumerate(data_loader)

# ----------------------------- Initialize RCG ----------------------------- #
class_cond = False
large_scale = True

if large_scale:
    print('large scale')
    if class_cond:
        rdm_ckpt_path = f'final_ckpts/rdm-mocov3vitl-clscond.pth'
        rdm_cfg = f'config/rdm/mocov3vitl_simplemlp_l12_w1536_classcond.yaml'
    else:
        rdm_ckpt_path = f'final_ckpts/rdm-mocov3vitl.pth'
        rdm_cfg = f'config/rdm/mocov3vitl_simplemlp_l12_w1536.yaml'
    
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
                                           pretrained_rdm_cfg=rdm_cfg)
    checkpoint = torch.load(os.path.join('final_ckpts/mage-l.pth'), map_location='cpu')
    
else:
    print('base scale')
    if class_cond:
        rdm_ckpt_path = f'final_ckpts/rdm-mocov3vitb-clscond.pth'
        rdm_cfg = f'config/rdm/mocov3vitb_simplemlp_l12_w1536_classcond.yaml'
    else:
        rdm_ckpt_path = f'final_ckpts/rdm-mocov3vitb.pth'
        rdm_cfg = f'config/rdm/mocov3vitb_simplemlp_l12_w1536.yaml'

    model = models_mage.mage_vit_base_patch16(mask_ratio_mu=0.75, mask_ratio_std=0.25,
                                            mask_ratio_min=0.5, mask_ratio_max=1.0,
                                            vqgan_ckpt_path='vqgan-ckpts/vqgan_jax_strongaug.ckpt',
                                            use_rep=True, rep_dim=256, rep_drop_prob=0.1,
                                            use_class_label=False,
                                            pretrained_enc_arch='mocov3_vit_base',
                                            pretrained_enc_path='pretrained_enc_ckpts/mocov3/vitb.pth.tar',
                                            pretrained_enc_proj_dim=256,
                                            pretrained_enc_withproj=True,
                                            pretrained_rdm_ckpt=rdm_ckpt_path,
                                            pretrained_rdm_cfg=rdm_cfg)
    checkpoint = torch.load(os.path.join('final_ckpts/mage-b.pth'), map_location='cpu')
    
model.load_state_dict(checkpoint['model'], strict=True)
model.cuda()
model.eval()

# ------------------------------- get moco rep ------------------------------- #
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
rep_list = []

with torch.no_grad():
    for _, (images, label) in data_iterator:
        if label == class_label:
            x_normalized = (images.cuda() - mean) / std
            x_normalized = torch.nn.functional.interpolate(x_normalized, 224, mode='bicubic')
            rep = model.pretrained_encoder.forward_features(x_normalized)
            if model.pretrained_enc_withproj:
                rep = model.pretrained_encoder.head(rep)
            rep_std = torch.std(rep, dim=1, keepdim=True)
            rep_mean = torch.mean(rep, dim=1, keepdim=True)
            rep = (rep - rep_mean) / rep_std  # [1, 256]
            rep_list.append(rep.cpu().numpy())

rep_np = np.concatenate(rep_list, axis=0)
print(rep_np.shape)
np.save(f'outputs/rep_large_class_{class_label}.npy', rep_np)
