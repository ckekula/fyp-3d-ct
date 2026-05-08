from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


TARGET_DISEASES: Dict[str, List[str]] = {
    "Lung nodule": [
        "lung nodule",
        "pulmonary nodule",
        "subcentimeter pulmonary nodule",
        "nodule in the lung",
    ],
    "Lung opacity": [
        "lung opacity",
        "pulmonary opacity",
        "ground-glass opacity",
        "diffuse lung opacity",
    ],
    "Consolidation": [
        "consolidation",
        "pulmonary consolidation",
        "lobar consolidation",
        "nodular consolidation",
    ],
    "Atelectasis": [
        "atelectasis",
        "linear atelectasis",
        "segmental atelectasis",
        "fibroatelectatic change",
    ],
}


@dataclass(frozen=True)
class DiseasePromptBundle:
    disease: str
    prompts: List[str]

    @property
    def text(self) -> str:
        return "[SEP]".join(self.prompts)


def default_prompt_bundles() -> List[DiseasePromptBundle]:
    return [DiseasePromptBundle(disease=name, prompts=prompts) for name, prompts in TARGET_DISEASES.items()]


def prompt_text_for_diseases(diseases: List[str] | None = None) -> str:
    selected = diseases if diseases else list(TARGET_DISEASES.keys())
    prompts: List[str] = []
    for disease in selected:
        prompts.extend(TARGET_DISEASES[disease])
    return "[SEP]".join(prompts)


def normalize_disease_name(text: str) -> str | None:
    lowered = text.lower()
    for disease, prompts in TARGET_DISEASES.items():
        if disease.lower() in lowered:
            return disease
        if any(prompt in lowered for prompt in prompts):
            return disease
    return None