"""
Drug Discovery Evaluation Module
=================================

Molecular property prediction and protein-ligand interaction tasks
for AIDD (Applied AI for Drug Discovery) evaluation.

Reference: AAAI AIDD 2026 workshop requirements
"""

from ten.evaluation.molecular import (
    MolecularPropertyPrediction,
    MoleculeNetDataset,
    SMILESTokenizer,
    TENForMolecularPrediction,
)
from ten.evaluation.metrics import (
    calculate_auroc,
    calculate_auprc,
    calculate_rmse,
    calculate_mae,
)

__all__ = [
    "MolecularPropertyPrediction",
    "MoleculeNetDataset",
    "SMILESTokenizer",
    "TENForMolecularPrediction",
    "calculate_auroc",
    "calculate_auprc",
    "calculate_rmse",
    "calculate_mae",
]
