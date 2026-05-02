# MedSAM2

**Segment Anything in 3D Medical Images and Videos**

## Updates

- 20250705: Release Efficient MedSAM2 baseline for FLARE 2025 Pan-cancer segmentation challenge [RECIST-to-3D](https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task1-PancancerRECIST-to-3D)
- 20250423: Release lung lesion segmentation dataset [LUNA25-MedSAM2](https://huggingface.co/datasets/wanglab/LUNA25-MedSAM2) for [LUNA25](https://luna25.grand-challenge.org/)

## Installation 

- Create a virtual environment: `uv venv --python 3.12.12` and `.venv/Scripts/activate` 
- Install [PyTorch](https://pytorch.org/get-started/locally/): `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124` (Linux CUDA 12.4)
- Download code `git clone https://github.com/bowang-lab/MedSAM2.git && cd MedSAM2` and run `pip install -e ".[dev]"`
- Download checkpoints: `bash download.sh`
  - `MedSAM2_2411.pt`: The based model trained in Nov. 2024
  - `MedSAM2_US_Heart.pt`: Fine-tuned model for heart ultrasound video segmentation
  - `MedSAM2_MRI_LiverLesion.pt`: Fine-tuned model for liver lesion MRI segmentation
  - `MedSAM2_CTLesion.pt`: Fine-tuned model for CT lesion segmentation
  - `MedSAM2_latest.pt` (recommended): Latest model trained on the combination of existing public datasets and newly annotated datasets
- Optional: Please install the following dependencies for gradio

```bash
sudo apt-get update
sudo apt-get install ffmpeg
pip install gradio==3.38.0
pip install numpy==1.26.3 
pip install ffmpeg-python 
pip install moviepy
```

## Download annotated datasets

- [CT_DeepLesion-MedSAM2](https://huggingface.co/datasets/wanglab/CT_DeepLesion-MedSAM2)

- [LLD-MMRI-MedSAM2](https://huggingface.co/datasets/wanglab/LLD-MMRI-MedSAM2) 

- [RVENet-MedSAM2](https://huggingface.co/datasets/wanglab/RVENet-MedSAM2)

Note: Please also cite the raw [DeepLesion](https://doi.org/10.1117/1.JMI.5.3.036501), [LLD-MMRI](https://www.sciencedirect.com/science/article/pii/S0893608025001078) and [RVENET](https://rvenet.github.io/dataset/) papers when using these datasets. 


## Inference

### 3D medical image segmentation

- [Colab](https://colab.research.google.com/drive/1MKna9Sg9c78LNcrVyG58cQQmaePZq2k2?usp=sharing): [MedSAM2_inference_CT_Lesion_Demo.ipynb](notebooks/MedSAM2_inference_CT_Lesion.ipynb)

- CMD

```bash
python medsam2_infer_3D_CT.py -i CT_DeepLesion/images -o CT_DeepLesion/segmentation
```

### Medical video segmentation

- [Colab](https://colab.research.google.com/drive/16niRHqdDZMCGV7lKuagNq_r_CEHtKY1f?usp=sharing): [MedSAM2_Inference_Video_Demo.ipynb](notebooks/MedSAM2_Inference_Video.ipynb)


- CMD

```bash
python medsam2_infer_video.py -i input_video_path -m input_mask_path -o output_video_path 
```




### Gradio demo

```bash
python app.py
```

## Training MedSAM2
Use [FLARE25 pan-cancer CT dataset](https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task1-PancancerRECIST-to-3D) as an example. 
- Download [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) to `checkpoints`
- Add dataset information in `sam2/configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml`: `data` -> `train` -> `datasets`
- Set `train_video_batch_size` based on the GPU memory


```bash
sh single_node_train_medsam2.sh
```

- multi-node training

```bash
sbatch multi_node_train.sh
```

- inference with RECIST marker (simulate a box prompt on middle slice)

```bash
python medsam2_infer_CT_lesion_npz_recist.py
```

## Training Efficient MedSAM2

- Train Efficient MedSAM2 on [FLARE25 pan-cancer CT dataset](https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task1-PancancerRECIST-to-3D) for CPU-based inference

```bash
sh single_node_train_eff_medsam2_FLARE25.sh
```

- Inference with RECIST marker on the FLARE25 pan-cancer [validation dataset](https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task1-PancancerRECIST-to-3D/tree/main/validation_npz/validation_public_npz). 

```python
npz = np.load('path to/CT_Lesion_FLARE23Ts_0057.npz', allow_pickle=True)
print(npz.keys())
imgs = npz['imgs'] # (D, W, H), [0, 255]
recist = npz['recist'] # (D, W, H), binary RECIST marker on tumor middle slice {0, 1}
gts = npz['gts'] # (D, W, H), 3D tumor ground truth mask. It will be not available in the testing set
```

> simulate a box prompt on middle slice

```bash
python eff_medsam2_infer_CT_lesion_npz_recist.py
```


## Acknowledgements

- We highly appreciate all the challenge organizers and dataset owners for providing the public datasets to the community.
- We thank Meta AI for making the source code of [SAM2](https://github.com/facebookresearch/sam2) and [EfficientTAM](https://github.com/yformer/EfficientTAM) publicly available. Please also cite these papers when using MedSAM2. 


## Bibtex

```bash
@article{MedSAM2,
    title={MedSAM2: Segment Anything in 3D Medical Images and Videos},
    author={Ma, Jun and Yang, Zongxin and Kim, Sumin and Chen, Bihui and Baharoon, Mohammed and Fallahpour, Adibvafa and Asakereh, Reza and Lyu, Hongwei and Wang, Bo},
    journal={arXiv preprint arXiv:2504.03600},
    year={2025}
}
```
Please also cite SAM2
```
@inproceedings{SAM2,
title={{SAM} 2: Segment Anything in Images and Videos},
    author={Nikhila Ravi and Valentin Gabeur and Yuan-Ting Hu and Ronghang Hu and Chaitanya Ryali and Tengyu Ma and Haitham Khedr and Roman R{\"a}dle and Chloe Rolland and Laura Gustafson and Eric Mintun and Junting Pan and Kalyan Vasudev Alwala and Nicolas Carion and Chao-Yuan Wu and Ross Girshick and Piotr Dollar and Christoph Feichtenhofer},
    booktitle={International Conference on Learning Representations},
    year={2025}
}
```

and EfficientTAM

```
@article{xiong2024efficienttam,
    title={Efficient Track Anything},
    author={Yunyang Xiong, Chong Zhou, Xiaoyu Xiang, Lemeng Wu, Chenchen Zhu, Zechun Liu, Saksham Suri, Balakrishnan Varadarajan, Ramya Akula, Forrest Iandola, Raghuraman Krishnamoorthi, Bilge Soran, Vikas Chandra},
    journal={preprint arXiv:2411.18933},
    year={2024}
}
```

