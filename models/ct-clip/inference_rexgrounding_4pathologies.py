"""
CT-CLIP Inference Script - RexGrounding-CT (4 Pathologies)

Inference for:
- Lung Nodule
- Lung Opacity
- Consolidation
- Atelectasis

Usage:
    python inference_rexgrounding_4pathologies.py \
        --model path/to/model.pt \
        --image path/to/ct_volume.nii.gz \
        --output results/
"""

import torch
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from transformers import BertTokenizer, BertModel
from transformer_maskgit import CTViT
from ct_clip import CTCLIP
import json


class PathologyDetector:
    """Detect 4 lung pathologies in CT volumes"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize pathology detector
        
        Args:
            model_path: Path to trained CT-CLIP model
            device: "cuda" or "cpu"
        """
        self.device = torch.device(device)
        
        # Pathologies to detect
        self.pathologies = [
            "Lung Nodule",
            "Lung Opacity",
            "Consolidation",
            "Atelectasis"
        ]
        
        print(f"Loading model from {model_path}...")
        
        # Create model
        image_encoder = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8
        )
        
        tokenizer = BertTokenizer.from_pretrained(
            'microsoft/BiomedVLP-CXR-BERT-specialized',
            do_lower_case=True
        )
        
        text_encoder = BertModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized"
        )
        text_encoder.resize_token_embeddings(len(tokenizer))
        
        self.model = CTCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            dim_image=294912,
            dim_text=768,
            dim_latent=512,
            use_mlm=False,
            downsample_image_embeds=False,
            use_all_token_embeds=False
        )
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        self.tokenizer = tokenizer
        
        print("✓ Model loaded successfully")
    
    def load_ct_volume(self, volume_path: str) -> torch.Tensor:
        """Load and preprocess CT volume"""
        print(f"Loading CT volume: {volume_path}")
        
        nii_img = nib.load(volume_path)
        img_data = nii_img.get_fdata()
        
        # Normalize
        img_data = np.clip(img_data, -1000, 1000) / 1000
        
        # Resize to 240x480x480
        target_shape = (240, 480, 480)
        
        # Crop if too large
        crop_z = min(img_data.shape[0], target_shape[0])
        crop_y = min(img_data.shape[1], target_shape[1])
        crop_x = min(img_data.shape[2], target_shape[2])
        
        img_data = img_data[:crop_z, :crop_y, :crop_x]
        
        # Pad if too small
        pad_z = max(0, target_shape[0] - img_data.shape[0])
        pad_y = max(0, target_shape[1] - img_data.shape[1])
        pad_x = max(0, target_shape[2] - img_data.shape[2])
        
        img_data = np.pad(
            img_data,
            ((0, pad_z), (0, pad_y), (0, pad_x)),
            mode='constant',
            constant_values=-1
        )
        
        tensor = torch.tensor(img_data, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        return tensor
    
    def detect_pathologies(self, volume: torch.Tensor) -> dict:
        """
        Detect pathologies in CT volume
        
        Returns:
            dict: {pathology_name: probability}
        """
        volume = volume.to(self.device)
        results = {}
        
        with torch.no_grad():
            for pathology in self.pathologies:
                # Create prompts
                positive_prompt = f"There is {pathology}."
                negative_prompt = f"There is no {pathology}."
                
                # Tokenize
                pos_tokens = self.tokenizer(
                    positive_prompt,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                neg_tokens = self.tokenizer(
                    negative_prompt,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Get latent embeddings
                with torch.no_grad():
                    _, pos_latent = self.model(
                        pos_tokens,
                        volume,
                        device=self.device,
                        return_encodings=True
                    )
                    _, neg_latent = self.model(
                        neg_tokens,
                        volume,
                        device=self.device,
                        return_encodings=True
                    )
                
                # Compute similarity
                pos_sim = torch.nn.functional.cosine_similarity(
                    pos_latent.unsqueeze(1),
                    pos_latent.unsqueeze(0),
                    dim=-1
                ).squeeze()
                
                neg_sim = torch.nn.functional.cosine_similarity(
                    neg_latent.unsqueeze(1),
                    neg_latent.unsqueeze(0),
                    dim=-1
                ).squeeze()
                
                # Get probability
                logits = torch.tensor([pos_sim, neg_sim])
                prob = torch.softmax(logits, dim=0)[0].item()
                
                results[pathology] = prob
        
        return results
    
    def infer(self, volume_path: str) -> dict:
        """Run inference on a single CT volume"""
        
        # Load volume
        volume = self.load_ct_volume(volume_path)
        
        # Detect pathologies
        print("\nDetecting pathologies...")
        results = self.detect_pathologies(volume)
        
        return results
    
    def print_results(self, results: dict, threshold: float = 0.5):
        """Print results in human-readable format"""
        print("\n" + "=" * 60)
        print("PATHOLOGY DETECTION RESULTS")
        print("=" * 60)
        
        print(f"\nPathology Detection (threshold={threshold}):\n")
        
        detected = []
        for pathology, prob in results.items():
            status = "✓ DETECTED" if prob > threshold else "✗ NOT DETECTED"
            print(f"  {pathology:20s}: {prob:.2%} [{status}]")
            
            if prob > threshold:
                detected.append(pathology)
        
        if detected:
            print(f"\n✓ Detected abnormalities: {', '.join(detected)}")
        else:
            print(f"\n✓ No significant abnormalities detected")
        
        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Detect pathologies in CT volumes using CT-CLIP"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained CT-CLIP model"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to CT volume (NIfTI format)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for pathology detection (default: 0.5)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--output",
        default="./inference_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("CT-CLIP Inference - 4 Lung Pathologies")
    print("=" * 60)
    
    print(f"\n📋 Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Image: {args.image}")
    print(f"  Device: {args.device}")
    print(f"  Threshold: {args.threshold}")
    
    # Initialize detector
    detector = PathologyDetector(args.model, device=args.device)
    
    # Run inference
    results = detector.infer(args.image)
    
    # Print results
    detector.print_results(results, threshold=args.threshold)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    results_file = output_dir / "pathology_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
