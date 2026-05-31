from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


DEFAULT_DISEASES = [
    "Lung nodule",
    "Lung opacity",
    "Consolidation",
    "Atelectasis",
]


# These keywords serve two purposes:
# 1. dataset_adapter.py uses them to find relevant ReXGroundingCT cases from findings text.
# 2. default_prompt_bundles() uses them to build BiomedParse text prompts.
TARGET_DISEASES: Dict[str, List[str]] = {
    "Lung nodule": [
        "lung nodule",
        "lung nodules",
        "pulmonary nodule",
        "pulmonary nodules",
        "subcentimeter pulmonary nodule",
        "subcentimeter pulmonary nodules",
        "small lung nodule",
        "small pulmonary nodule",
        "nodule",
        "nodules",
        "nodular lesion",
        "nodular lesions",
        "nonspecific pulmonary nodule",
        "nonspecific pulmonary nodules",
        "micronodule",
        "micronodules",
        "reticulonodular density",
        "reticulonodular density increase",
    ],
    "Lung opacity": [
        "lung opacity",
        "lung opacities",
        "pulmonary opacity",
        "pulmonary opacities",
        "opacity",
        "opacities",
        "ground glass opacity",
        "ground glass opacities",
        "ground-glass opacity",
        "ground-glass opacities",
        "GGO",
        "mosaic attenuation",
        "parenchymal opacity",
        "parenchymal opacities",
        "density increase",
        "density increases",
    ],
    "Consolidation": [
        "consolidation",
        "consolidations",
        "pulmonary consolidation",
        "lobar consolidation",
        "segmental consolidation",
        "airspace consolidation",
        "consolidation area",
        "consolidation areas",
        "consolidative opacity",
        "consolidative opacities",
        "nodular consolidation",
    ],
    "Atelectasis": [
        "atelectasis",
        "linear atelectasis",
        "segmental atelectasis",
        "subsegmental atelectasis",
        "atelectatic change",
        "atelectatic changes",
        "fibroatelectatic change",
        "fibroatelectatic changes",
        "fibroatelectatic",
        "lung collapse",
    ],
}


@dataclass(frozen=True)
class PromptBundle:
    disease: str
    prompts: List[str]

    @property
    def text(self) -> str:
        # BiomedParse supports multiple text prompts separated by [SEP].
        return " [SEP] ".join(self.prompts)


def normalize_disease_name(name: str) -> str:
    normalized = " ".join(str(name).strip().lower().replace("-", " ").split())

    aliases = {
        "lung nodule": "Lung nodule",
        "lung nodules": "Lung nodule",
        "pulmonary nodule": "Lung nodule",
        "pulmonary nodules": "Lung nodule",
        "nodule": "Lung nodule",
        "nodules": "Lung nodule",
        "nodular lesion": "Lung nodule",
        "nodular lesions": "Lung nodule",

        "lung opacity": "Lung opacity",
        "lung opacities": "Lung opacity",
        "pulmonary opacity": "Lung opacity",
        "pulmonary opacities": "Lung opacity",
        "opacity": "Lung opacity",
        "opacities": "Lung opacity",
        "ground glass opacity": "Lung opacity",
        "ground glass opacities": "Lung opacity",
        "ground-glass opacity": "Lung opacity",
        "ground-glass opacities": "Lung opacity",
        "ggo": "Lung opacity",

        "consolidation": "Consolidation",
        "consolidations": "Consolidation",
        "pulmonary consolidation": "Consolidation",

        "atelectasis": "Atelectasis",
        "atelectatic": "Atelectasis",
        "linear atelectasis": "Atelectasis",
        "fibroatelectatic": "Atelectasis",
    }

    return aliases.get(normalized, name)


def default_prompt_bundles() -> List[PromptBundle]:
    """
    Build default disease-level prompt bundles for the selected FYP abnormalities.

    The prompts are intentionally disease-level, not case-specific. For evaluation,
    each bundle produces one disease mask per CT case.
    """
    return [
        PromptBundle(disease=disease, prompts=prompts)
        for disease, prompts in TARGET_DISEASES.items()
    ]
