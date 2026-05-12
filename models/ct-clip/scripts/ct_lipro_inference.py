import os
import sys
from pathlib import Path
import copy
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ct_lipro_inference.log')
    ]
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CT_CLIP_DIR = ROOT / "CT_CLIP"
TRANSFORMER_MASKGIT_DIR = ROOT / "transformer_maskgit"
for path in (CT_CLIP_DIR, TRANSFORMER_MASKGIT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
logger.info(f"Added paths: {CT_CLIP_DIR}, {TRANSFORMER_MASKGIT_DIR}")

from src.args import parse_arguments
logger.info("Imported parse_arguments")
from transformers import BertTokenizer, BertModel
logger.info("Imported BertTokenizer and BertModel")
from transformer_maskgit import CTViT
logger.info("Imported CTViT")
from ct_clip import CTCLIP
logger.info("Imported CTCLIP")
from data_inference_nii import CTReportDatasetinfer
logger.info("Imported CTReportDatasetinfer")
from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis
logger.info("Imported evaluation functions")
import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
logger.info("Imported all dependencies successfully")

def sigmoid(tensor):
    return 1 / (1 + torch.exp(-tensor))

def _save_interim_results(plotdir, realall, predictedall, accs, batch_count):
    """Save interim results without overwriting final results"""
    try:
        # Filter to 4 pathologies
        pathology_indices = [8, 9, 10, 15]
        
        if len(realall) > 0:
            real_arr = np.array(realall)[:, pathology_indices] if np.array(realall).ndim > 1 else np.array(realall)
            pred_arr = np.array(predictedall)[:, pathology_indices] if np.array(predictedall).ndim > 1 else np.array(predictedall)
            
            # Save interim versions (not overwriting final)
            interim_dir = os.path.join(plotdir, "interim")
            os.makedirs(interim_dir, exist_ok=True)
            
            np.savez(f"{interim_dir}/labels_interim_{batch_count}.npz", data=real_arr)
            np.savez(f"{interim_dir}/predicted_interim_{batch_count}.npz", data=pred_arr)
            
            # Also save latest interim as current
            np.savez(f"{interim_dir}/labels_interim_latest.npz", data=real_arr)
            np.savez(f"{interim_dir}/predicted_interim_latest.npz", data=pred_arr)
            
            # Save accessions list
            with open(f"{interim_dir}/accessions_interim_{batch_count}.txt", "w") as f:
                for item in accs:
                    f.write(item[0] + "\n")
            
            logger.info(f"Saved interim results for {batch_count} samples at {interim_dir}")
    except Exception as e:
        logger.warning(f"Failed to save interim results: {e}")

class ImageLatentsClassifier(nn.Module):
    def __init__(self, trained_model, latent_dim, num_classes, dropout_prob=0.3):
        super(ImageLatentsClassifier, self).__init__()
        self.trained_model = trained_model
        self.dropout = nn.Dropout(dropout_prob)  # Add dropout layer
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(latent_dim, num_classes)  # Assuming trained_model.image_latents_dim gives the size of the image_latents

    def forward(self, latents=False, *args, **kwargs):
        kwargs['return_latents'] = True
        _, image_latents, _ = self.trained_model(*args, **kwargs)
        image_latents = self.relu(image_latents)
        if latents:
            return image_latents
        image_latents = self.dropout(image_latents)  # Apply dropout on the latents

        return self.classifier(image_latents)

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)
    def load(self, file_path):
        loaded_state_dict = torch.load(file_path)
        self.load_state_dict(loaded_state_dict)

def evaluate_model(args, model, dataloader, device):
    logger.info("Starting model evaluation")
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")
    
    plotdir = args.save
    os.makedirs(plotdir, exist_ok=True)
    logger.info(f"Created output directory: {plotdir}")
    
    correct = 0
    total = 0
    predictedall=[]
    realall=[]
    accs = []
    batch_count = 0
    save_interval = 50  # Save after every N samples
    logger.info(f"Will save results incrementally every {save_interval} samples")
    
    with torch.no_grad():

        for batch in tqdm.tqdm(dataloader):
            batch_count += 1
            if batch_count % 10 == 0:
                logger.info(f"Processing batch {batch_count}")
            inputs, _, labels, acc_no = batch
            labels = labels.float().to(device)
            inputs = inputs.to(device)
            # Assuming your model takes in the same inputs as during training
            text_tokens = tokenizer("", return_tensors="pt", padding="max_length", truncation=True, max_length=200).to(device)
            output = model(False, text_tokens, inputs,  device=device, return_latents=True)
            realall.append(labels.detach().cpu().numpy()[0])
            save_out = sigmoid(torch.tensor(output)).cpu().numpy()
            predictedall.append(save_out[0])
            accs.append(acc_no[0])
            print(acc_no[0], flush=True)
            
            # Save incrementally every N samples
            if batch_count % save_interval == 0:
                logger.info(f"Saving interim results for {batch_count} samples")
                _save_interim_results(plotdir, realall, predictedall, accs, batch_count)

        logger.info(f"Completed processing {batch_count} batches")

        # Full 18 pathologies from model
        all_pathologies = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']
        
        # Filter to 4 pathologies only
        pathologies = ['Atelectasis', 'Lung nodule', 'Lung opacity', 'Consolidation']
        pathology_indices = [8, 9, 10, 15]  # indices in original 18-class order
        logger.info(f"Filtering to 4 pathologies: {pathologies}")

        realall=np.array(realall)
        predictedall=np.array(predictedall)
        logger.info(f"Real labels shape: {realall.shape}, Predicted shape: {predictedall.shape}")
        
        # Filter predictions and labels to only the 4 selected pathologies
        realall = realall[:, pathology_indices]
        predictedall = predictedall[:, pathology_indices]
        logger.info(f"After filtering - Real shape: {realall.shape}, Predicted shape: {predictedall.shape}")

        logger.info(f"Saving final results for {len(accs)} total samples")
        np.savez(f"{plotdir}/labels_weights.npz", data=realall)
        np.savez(f"{plotdir}/predicted_weights.npz", data=predictedall)
        logger.info(f"Saved final NPZ files to {plotdir}")

        with open(f"{plotdir}/accessions.txt", "w") as file:
            for item in accs:
                file.write(item[0] + "\n")
        logger.info(f"Saved {len(accs)} accessions to {plotdir}/accessions.txt")

        dfs=evaluate_internal(predictedall,realall,pathologies, plotdir)
        logger.info("Computed final evaluation metrics")

        writer = pd.ExcelWriter(f'{plotdir}/aurocs.xlsx', engine='xlsxwriter')

        dfs.to_excel(writer, sheet_name='Sheet1', index=False)

        writer.close()
        logger.info(f"Saved final AUROC results to {plotdir}/aurocs.xlsx")
        logger.info("Model evaluation completed successfully")




if __name__ == '__main__':
    logger.info("="*80)
    logger.info("Starting CT-CLIP LiPro Inference Pipeline")
    logger.info("="*80)
    
    args = parse_arguments()  # Assuming this function provides necessary arguments
    logger.info(f"Arguments parsed: pretrained={args.pretrained}, device={args.device}")
    logger.info(f"Data folder: {args.data_folder}")
    logger.info(f"Output directory: {args.save}")

    logger.info("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
    logger.info("Tokenizer loaded")
    
    logger.info("Loading text encoder...")
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    logger.info("Text encoder loaded")

    text_encoder.resize_token_embeddings(len(tokenizer))
    logger.info(f"Text encoder tokenizer embeddings resized to {len(tokenizer)}")

    logger.info("Creating image encoder (CTViT)...")
    image_encoder = CTViT(
        dim = 512,
        codebook_size = 8192,
        image_size = 480,
        patch_size = 20,
        temporal_patch_size = 10,
        spatial_depth = 4,
        temporal_depth = 4,
        dim_head = 32,
        heads = 8
    )
    logger.info("Image encoder created")

    logger.info("Creating CTCLIP model...")
    clip = CTCLIP(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        dim_image = 294912,
        dim_text = 768,
        dim_latent = 512,
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False

    )
    logger.info("CTCLIP model created")

    num_classes = 18  # Keep 18 to match checkpoint; filter output to 4 pathologies
    logger.info(f"Creating ImageLatentsClassifier with {num_classes} classes")
    image_classifier = ImageLatentsClassifier(clip, 512, num_classes)
    zero_shot = copy.deepcopy(image_classifier)
    logger.info("ImageLatentsClassifier created")

    if args.pretrained is None:
        logger.error("--pretrained must be provided (path to checkpoint)")
        raise ValueError("--pretrained must be provided (path to checkpoint)")

    logger.info(f"Loading checkpoint from: {args.pretrained}")
    # Load checkpoint with proper map_location to avoid GPU/CPU mismatch
    state = torch.load(args.pretrained, map_location=torch.device(args.device))
    logger.info("Checkpoint file loaded into memory")

    # Determine the actual state dict (may be nested under a key)
    state_dict = state
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "weights"):
            if key in state:
                state_dict = state[key]
                logger.info(f"Using state dict from nested key: '{key}'")
                break

    # Use strict=False to handle version mismatches:
    #   - 'position_ids' is a persistent buffer in older BERT but non-persistent in newer versions
    #   - 'embed_avg' may be added by newer vector-quantization codebook code
    result = image_classifier.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        logger.warning(f"Missing keys (will use initialized values): {result.missing_keys}")
    if result.unexpected_keys:
        logger.warning(f"Unexpected keys (ignored from checkpoint): {result.unexpected_keys}")

    logger.info("Checkpoint loaded successfully")

    logger.info(f"Creating dataset from folder: {args.data_folder}")
    # Prepare the evaluation dataset
    ds = CTReportDatasetinfer(data_folder=args.data_folder, reports_file=args.reports_file, meta_file=args.meta_file, labels = args.labels)
    logger.info(f"Dataset created with {len(ds)} samples")
    
    logger.info("Creating DataLoader...")
    dl = DataLoader(ds, num_workers=8, batch_size=1, shuffle=False)
    logger.info("DataLoader created")

    logger.info("="*80)
    logger.info("Starting evaluation...")
    logger.info("="*80)
    # Evaluate the model using the parsed device
    device = torch.device(args.device)
    evaluate_model(args, image_classifier, dl, device)
    logger.info("="*80)
    logger.info("Inference pipeline completed successfully!")
    logger.info("="*80)