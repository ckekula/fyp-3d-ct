from __future__ import annotations

DEFAULT_DISEASES = [
    "Lung nodule",
    "Lung opacity",
    "Consolidation",
    "Atelectasis",
]


def build_disease_prompt(disease: str) -> str:
    """Build a simple prompt template for a disease."""
    return f"Are there any {disease}?"


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






