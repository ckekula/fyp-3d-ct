# SegVol: Universal and Interactive Volumetric Medical Image Segmentation
<div align="center">
  
  <img src="https://github.com/BAAI-DCAI/SegVol/assets/60123629/d2b82996-7f2c-4de5-bfc8-8ccd115bfcdf" width="85%" height="85%">

 | 🌟**Quickstart([ModelScope](https://www.modelscope.cn/models/yuxindu/SegVol/summary) / [🤗HF](https://huggingface.co/BAAI/SegVol))** | 📃 [**Paper**](https://arxiv.org/abs/2311.13385) | [**Web Tool**](https://www.modelscope.cn/studios/YuxinDu/SegVol/summary) | 📂 **Datasets([ModelScope](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-Seg/summary)/[🤗HF](https://huggingface.co/datasets/GoodBaiBai88/M3D-Seg))** |
</div>

🎉🎉🎉**Our paper has been accepted at NeurIPS 2024 as a spotlight!**

[**SegVol repo for CVPR SegFM**](https://github.com/Yuxin-Du-Lab/SegVol-for-SegFM)

The SegVol is a universal and interactive model for volumetric medical image segmentation. SegVol accepts **point**, **box** and **text** prompt while output volumetric segmentation. By training on 90k unlabeled Computed Tomography (CT) volumes and 6k labeled CTs, this foundation model supports the segmentation of over 200 anatomical categories.

We have released SegVol's **inference code**, **training code**, **model params** and **ViT pre-training params** (pre-training is performed over 2,000 epochs on 96k  CTs). 

**Keywords**: 3D medical SAM, volumetric image segmentation

## Quickstart: Enable easy training and testing
### 🌟[Quickstart](https://www.modelscope.cn/models/yuxindu/SegVol/summary) with ModelScope (无需代理)
### 🌟[Quickstart](https://huggingface.co/BAAI/SegVol) with HuggingFace

## Start with source code
### Requirements
The [pytorch v1.11.0](https://pytorch.org/get-started/previous-versions/) (or a higher version) is needed first. Following install key requirements using commands:

```
pip install 'monai[all]==0.9.0'
pip install einops==0.6.1
pip install transformers==4.18.0
pip install matplotlib
```

### Guideline for training and inference
[How to infer a demo case](./documents/inference_demo.md).

[How to train SegVol](./documents/training.md).

[How to use our pre-trained ViT as your model encoder](./documents/pretrained_vit.md).

## [Web Tool](https://www.modelscope.cn/studios/YuxinDu/SegVol/summary) of SegVol 📽

## News🚀
(2024.01.03) *A radar map about [**zero-shot experiment**](#jump) has been reported.* 🏆

(2023.12.25) *Our web tool **supports download results** now! You can use it as an online tool.* 🔥🔥🔥

(2023.12.15) *The training code has been uploaded!*

(2023.12.04) ***A web tool of SegVol is [here](https://www.modelscope.cn/studios/YuxinDu/SegVol/summary)! Just enjoy it!*** 🔥🔥🔥

(2023.11.28) *Our model and demo case have been open-source at [huggingface/BAAI/SegVol](https://huggingface.co/BAAI/SegVol/tree/main).* 🤗🤗

(2023.11.28) *The usage of pre-trained ViT has been uploaded.* 

(2023.11.24) *You can download weight files of SegVol and ViT(CTs pre-train) from [huggingface/BAAI/SegVol](https://huggingface.co/BAAI/SegVol/tree/main) or [Google Drive](https://drive.google.com/drive/folders/1TEJtgctH534Ko5r4i79usJvqmXVuLf54?usp=drive_link).* 🔥🔥🔥

(2023.11.23) *The brief introduction and instruction have been uploaded.*

(2023.11.23) *The inference demo code has been uploaded.*

(2023.11.22) *The first edition of our paper has been uploaded to arXiv.* 📃

## Citation
If you find this repository helpful, please consider citing:
```
@article{du2023segvol,
  title={SegVol: Universal and Interactive Volumetric Medical Image Segmentation},
  author={Du, Yuxin and Bai, Fan and Huang, Tiejun and Zhao, Bo},
  journal={arXiv preprint arXiv:2311.13385},
  year={2023}
}
@misc{bai2024m3dadvancing3dmedical,
      title={M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models}, 
      author={Fan Bai and Yuxin Du and Tiejun Huang and Max Q. -H. Meng and Bo Zhao},
      year={2024},
      eprint={2404.00578},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.00578}, 
}
```
  
## Acknowledgement

This work is supported by the National Science and Technology Major Project (No. 2022ZD0116314).

Thanks for the following amazing works:

[HuggingFace](https://huggingface.co/).

[CLIP](https://github.com/openai/CLIP).

[MONAI](https://github.com/Project-MONAI/MONAI).

[3D Slicer](https://www.slicer.org/).

[Image by brgfx](https://www.freepik.com/free-vector/anatomical-structure-human-bodies_26353260.htm) on Freepik.

[Image by muammark](https://www.freepik.com/free-vector/people-icon-collection_1157380.htm#query=user&position=2&from_view=search&track=sph) on Freepik.



