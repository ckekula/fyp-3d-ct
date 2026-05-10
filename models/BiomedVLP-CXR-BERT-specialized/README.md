---
language: en
tags:
- exbert
license: mit
pipeline_tag: fill-mask
widget:
- text: "Left pleural effusion with adjacent [MASK]."
  example_title: "Radiology 1"
- text: "Heart size normal and lungs are [MASK]."
  example_title: "Radiology 2"
inference: false
---

# CXR-BERT-specialized

[CXR-BERT](https://arxiv.org/abs/2204.09817) is a chest X-ray (CXR) domain-specific language model that makes use of an improved vocabulary, novel pretraining procedure, weight regularization, and text augmentations. The resulting model demonstrates improved performance on radiology natural language inference, radiology masked language model token prediction, and downstream vision-language processing tasks such as zero-shot phrase grounding and image classification.

First, we pretrain [**CXR-BERT-general**](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-general) from a randomly initialized BERT model via Masked Language Modeling (MLM) on abstracts [PubMed](https://pubmed.ncbi.nlm.nih.gov/) and clinical notes from the publicly-available [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) and [MIMIC-CXR](https://physionet.org/content/mimic-cxr/). In that regard, the general model is expected be applicable for research in clinical domains other than the chest radiology through domain specific fine-tuning.

**CXR-BERT-specialized** is continually pretrained from CXR-BERT-general to further specialize in the chest X-ray domain. At the final stage, CXR-BERT is trained in a multi-modal contrastive learning framework, similar to the [CLIP](https://arxiv.org/abs/2103.00020) framework. The latent representation of [CLS] token is utilized to align text/image embeddings.

## Model variations

| Model                                             | Model identifier on HuggingFace                                                                             | Vocabulary     | Note                                                      |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------- | --------------------------------------------------------- |
| CXR-BERT-general                                  | [microsoft/BiomedVLP-CXR-BERT-general](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-general)         | PubMed & MIMIC | Pretrained for biomedical literature and clinical domains |
| CXR-BERT-specialized (after multi-modal training) | [microsoft/BiomedVLP-CXR-BERT-specialized](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized) | PubMed & MIMIC | Pretrained for chest X-ray domain                         |

## Image model

**CXR-BERT-specialized** is jointly trained with a ResNet-50 image model in a multi-modal contrastive learning framework. Prior to multi-modal learning, the image model is pre-trained on the same set of images in MIMIC-CXR using [SimCLR](https://arxiv.org/abs/2002.05709). The corresponding model definition and its loading functions can be accessed through our [HI-ML-Multimodal](https://github.com/microsoft/hi-ml/blob/main/hi-ml-multimodal/src/health_multimodal/image/model/model.py) GitHub repository. The joint image and text model, namely [BioViL](https://arxiv.org/abs/2204.09817), can be used in phrase grounding applications as shown in this python notebook [example](https://mybinder.org/v2/gh/microsoft/hi-ml/HEAD?labpath=hi-ml-multimodal%2Fnotebooks%2Fphrase_grounding.ipynb). Additionally, please check the [MS-CXR benchmark](https://physionet.org/content/ms-cxr/0.1/) for a more systematic evaluation of joint image and text models in phrase grounding tasks.

## Citation

The corresponding manuscript is accepted to be presented at the [**European Conference on Computer Vision (ECCV) 2022**](https://eccv2022.ecva.net/)

```bibtex
@misc{https://doi.org/10.48550/arxiv.2204.09817,
  doi = {10.48550/ARXIV.2204.09817},
  url = {https://arxiv.org/abs/2204.09817},
  author = {Boecking, Benedikt and Usuyama, Naoto and Bannur, Shruthi and Castro, Daniel C. and Schwaighofer, Anton and Hyland, Stephanie and Wetscherek, Maria and Naumann, Tristan and Nori, Aditya and Alvarez-Valle, Javier and Poon, Hoifung and Oktay, Ozan},
  title = {Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing},
  publisher = {arXiv},
  year = {2022},
}
```

## Model Use

### Intended Use

This model is intended to be used solely for (I) future research on visual-language processing and (II) reproducibility of the experimental results reported in the reference paper.

#### Primary Intended Use

The primary intended use is to support AI researchers building on top of this work. CXR-BERT and its associated models should be helpful for exploring various clinical NLP & VLP research questions, especially in the radiology domain.

#### Out-of-Scope Use

**Any** deployed use case of the model --- commercial or otherwise --- is currently out of scope. Although we evaluated the models using a broad set of publicly-available research benchmarks, the models and evaluations are not intended for deployed use cases. Please refer to [the associated paper](https://arxiv.org/abs/2204.09817) for more details.

### How to use

Here is how to use this model to extract radiological sentence embeddings and obtain their cosine similarity in the joint space (image and text):

```python
import torch
from transformers import AutoModel, AutoTokenizer

# Load the model and tokenizer
url = "microsoft/BiomedVLP-CXR-BERT-specialized"
tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
model = AutoModel.from_pretrained(url, trust_remote_code=True)

# Input text prompts (e.g., reference, synonym, contradiction)
text_prompts = ["There is no pneumothorax or pleural effusion",
                "No pleural effusion or pneumothorax is seen",
                "The extent of the pleural effusion is constant."]

# Tokenize and compute the sentence embeddings
tokenizer_output = tokenizer.batch_encode_plus(batch_text_or_text_pairs=text_prompts,
                                               add_special_tokens=True,
                                               padding='longest',
                                               return_tensors='pt')
embeddings = model.get_projected_text_embeddings(input_ids=tokenizer_output.input_ids,
                                                 attention_mask=tokenizer_output.attention_mask)

# Compute the cosine similarity of sentence embeddings obtained from input text prompts.
sim = torch.mm(embeddings, embeddings.t())
```

## Data

This model builds upon existing publicly-available datasets:

- [PubMed](https://pubmed.ncbi.nlm.nih.gov/)
- [MIMIC-III](https://physionet.org/content/mimiciii/)
- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/)

These datasets reflect a broad variety of sources ranging from biomedical abstracts to intensive care unit notes to chest X-ray radiology notes. The radiology notes are accompanied with their associated chest x-ray DICOM images in MIMIC-CXR dataset.  

## Performance

We demonstrate that this language model achieves state-of-the-art results in radiology natural language inference through its improved vocabulary and novel language pretraining objective leveraging semantics and discourse characteristics in radiology reports.

A highlight of comparison to other common models, including [ClinicalBERT](https://aka.ms/clinicalbert) and [PubMedBERT](https://aka.ms/pubmedbert):

|                                                 | RadNLI accuracy (MedNLI transfer) | Mask prediction accuracy | Avg. # tokens after tokenization | Vocabulary size |
| ----------------------------------------------- | :-------------------------------: | :----------------------: | :------------------------------: | :-------------: |
| RadNLI baseline                                 |               53.30               |            -             |                -                 |        -        |
| ClinicalBERT                                    |               47.67               |          39.84           |         78.98 (+38.15%)          |     28,996      |
| PubMedBERT                                      |               57.71               |          35.24           |         63.55 (+11.16%)          |     28,895      |
| CXR-BERT (after Phase-III)                      |               60.46               |          77.72           |          58.07 (+1.59%)          |     30,522      |
| **CXR-BERT (after Phase-III + Joint Training)** |             **65.21**             |        **81.58**         |        **58.07 (+1.59%)**        |     30,522      |

CXR-BERT also contributes to better vision-language representation learning through its improved text encoding capability. Below is the zero-shot phrase grounding performance on the **MS-CXR** dataset, which evaluates the quality of image-text latent representations.

| Vision–Language Pretraining Method | Text Encoder | MS-CXR Phrase Grounding (Avg. CNR Score) |
| ---------------------------------- | ------------ | :--------------------------------------: |
| Baseline                           | ClinicalBERT |                  0.769                   |
| Baseline                           | PubMedBERT   |                  0.773                   |
| ConVIRT                            | ClinicalBERT |                  0.818                   |
| GLoRIA                             | ClinicalBERT |                  0.930                   |
| **BioViL**                         | **CXR-BERT** |                **1.027**                 |
| **BioViL-L**                       | **CXR-BERT** |                **1.142**                 |

Additional details about performance can be found in the corresponding paper, [Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing](https://arxiv.org/abs/2204.09817).

## Limitations

This model was developed using English corpora, and thus can be considered English-only.

## Further information

Please refer to the corresponding paper, ["Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing", ECCV'22](https://arxiv.org/abs/2204.09817) for additional details on the model training and evaluation.

For additional inference pipelines with CXR-BERT, please refer to the [HI-ML-Multimodal GitHub](https://aka.ms/biovil-code) repository.
