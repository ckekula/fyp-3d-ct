import logging
import pickle

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from lc_ksvd.config import MODELS_DIR, SPARSE_CODE_DIR
from lc_ksvd.patch_extractor.patch_extraction import load_unified_patch_matrix
from lc_ksvd.metrics import normalise_columns
from lc_ksvd.inference.evaluate import evaluate

from reppi import OMP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DICT_MODEL_PATH = MODELS_DIR / "unified_frozen.pkl"
SVM_MODEL_PATH = MODELS_DIR / "svm_model.pkl"
LOG_REG_MODEL_PATH = MODELS_DIR / "log_reg_model.pkl"

N_NONZERO_COEFS = 10


def load_dictionary(path=DICT_MODEL_PATH) -> np.ndarray:
    """Load the trained IncrementalFrozenDictionary payload and return D."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    D = model.D_
    if D is None:
        raise ValueError(f"Loaded model at {path} has no fitted dictionary (D_ is None).")
    logger.info(f"Loaded dictionary D of shape {D.shape} from {path}")
    return D


def encode_patches(X: np.ndarray, D: np.ndarray, n_nonzero_coefs: int = N_NONZERO_COEFS) -> np.ndarray:
    """Sparse-code X against D using Batch-OMP."""
    X_norm, _, _ = normalise_columns(X)
    omp = OMP(n_nonzero_coefs=n_nonzero_coefs, mode="batch", check_dict=True)
    Gamma = omp.encode(X_norm, D)
    logger.info(f"Encoded {X.shape[1]} patches -> Gamma shape {Gamma.shape}")
    return Gamma


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SPARSE_CODE_DIR.mkdir(parents=True, exist_ok=True)

    split = "test"
    # -- Load patches and labels ------------------------------------------------
    X, labels, _scan_ids, _coords = load_unified_patch_matrix(split=split)

    # -- Load trained dictionary --------------------------------------------------
    D = load_dictionary(DICT_MODEL_PATH)

    # -- Sparse-code patches against the dictionary -------------------------------
    if (SPARSE_CODE_DIR / f"{split}_sparse_codes.npz").exists():
        logger.info(f"Loading existing sparse codes from {SPARSE_CODE_DIR / f'{split}_sparse_codes.npz'}")
        data = np.load(SPARSE_CODE_DIR / f"{split}_sparse_codes.npz")
        Gamma = data["Gamma"]
    else:
        Gamma = encode_patches(X, D)

        # -- Save sparse codes (dense) alongside labels for reuse ---------------------
        np.savez_compressed(SPARSE_CODE_DIR / f"{split}_sparse_codes.npz", Gamma=Gamma, labels=labels)
        logger.info(f"Saved sparse codes -> {SPARSE_CODE_DIR / 'sparse_codes.npz'}")


    # -- Inference and evaluate using SVM ------------------------------------------------------------------
    logger.info("Inferencing from LinearSVC...")
    svm_clf = joblib.load(SVM_MODEL_PATH)
    y_pred_svm = svm_clf.predict(Gamma.T)
    evaluate(y_pred_svm, labels)



    # -- Inference and evaluate using Logistic Regression ---------------------------------------------------
    logger.info("Inferencing from LogisticRegression...")
    log_reg_clf = joblib.load(LOG_REG_MODEL_PATH)
    y_pred_log_reg = log_reg_clf.predict(Gamma.T)
    evaluate(y_pred_log_reg, labels)


if __name__ == "__main__":
    main()