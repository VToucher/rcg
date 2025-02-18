# RCG PyTorch Implementation

## 1. environment
```bash
conda env create -f environment.yaml
conda activate rcg
pip install gdown
```
### 1.1 config after dreamood
```bash
pip install timm==0.3.2
```

## 2. checkpoint download
```bash
## 1. pretrained encoder
cd pretrained_enc/moco_v3
# moco_v3_large
gdown --fuzzy https://drive.google.com/file/d/1Foa2-FqhwIFYjcAAbY9sXyO-1Vwwx-_9/view?usp=sharing
# moco_v3_base
wget https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar

## 2. RDM
cd final_ckpts
# uncond RDM (moco_v3_base)
gdown --fuzzy https://drive.google.com/file/d/1gdsvzKLmmBWuF4Ymy4rQ_T1t6dDHnTEA/view?usp=sharing
# cond RDM (moco_v3_base)
gdown --fuzzy https://drive.google.com/file/d/1roanmVfg-UaddVehstQErvByqi0OYs2R/view?usp=sharing

# uncond RDM (moco_v3_large)
gdown --fuzzy https://drive.google.com/file/d/1E5E3i9LRpSy0tVF7NA0bGXEh4CrjHAXz/view?usp=sharing
# cond RDM (moco_v3_large)
gdown --fuzzy https://drive.google.com/file/d/1lZmXOcdHE97Qmn2azNAo2tNVX7dtTAkY/view?usp=sharing

## 3. MAGE
# MAGE-base
gdown --fuzzy https://drive.google.com/file/d/1iZY0ujWp5GVochTLj0U6j4HgVTOyWPUI/view?usp=sharing
# MAGE-large
gdown --fuzzy https://drive.google.com/file/d/1nQh9xCqjQCd78zKwn2L9eLfLyVosb1hp/view?usp=sharing

## 4. VQGAN
gdown --fuzzy https://drive.google.com/file/d/13S_unB87n6KKuuMdyMnyExW0G1kplTbP/view?usp=sharing
```