import logging
import pickle
import torch
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from lc_ksvd.config import MODELS_DIR, SPARSE_CODE_DIR
from lc_ksvd.patch_extractor.patch_extraction import load_unified_patch_matrix, extract_unified
from lc_ksvd.metrics import normalise_columns
from lc_ksvd.inference.evaluate import evaluate, save_classification_results

from reppi import OMP
# from reppi import fista_core
from reppi.sparse.fista.utils import soft_threshold
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DICT_MODEL_PATH = MODELS_DIR / "unified_lcksvd2.pkl"
SVM_MODEL_PATH = MODELS_DIR / "lcksvd2_svm_model.pkl"
ALPHA = 0.1
N_NONZERO_COEFS = 10
SPLIT = "test"
MODEL = "lcksvd2"


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


def load_dictionary_and_boundaries(path=DICT_MODEL_PATH):
    """
    Load D together with its per-class atom-index ranges, for reconstructing
    a signal from only one class's block of the dictionary.

    Both LCKSVD and IncrementalFrozenDictionary partition D_'s columns into
    contiguous per-class blocks and expose that partition as
    ``model.class_boundaries_`` : {class_idx: (start, end)} (end exclusive),
    with class_idx matching CLASS_ORDER's integer indices (see
    lc_ksvd.model_fitting._fit_lcksvd / _fit_frozen, which build/label each
    stage from CLASS_ORDER directly). Gamma returned by encode_patches_omp()
    against this same D has one row per D column in the same order, so
    D[:, s:e] @ Gamma[s:e, :] reconstructs using only that class's atoms.

    Returns
    -------
    D : np.ndarray, shape (n_features, n_components)
    class_boundaries : dict[int, tuple[int, int]]
    class_order : list[str] or None
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    D = model.D_
    if D is None:
        raise ValueError(f"Loaded model at {path} has no fitted dictionary (D_ is None).")
    class_boundaries = getattr(model, "class_boundaries_", None)
    if not class_boundaries:
        raise ValueError(
            f"Loaded model at {path} ({type(model).__name__}) has no "
            "class_boundaries_ — per-class dictionary reconstruction needs "
            "a model trained with LCKSVD or IncrementalFrozenDictionary."
        )
    class_order = payload.get("class_order")
    logger.info(
        f"Loaded dictionary D of shape {D.shape} from {path} "
        f"with class_boundaries_={class_boundaries}"
    )
    return D, class_boundaries, class_order

def encode_patches_omp(X: np.ndarray, D: np.ndarray, n_nonzero_coefs: int = N_NONZERO_COEFS) -> np.ndarray:
    """Sparse-code X against D using Batch-OMP."""
    X_norm, _, _ = normalise_columns(X)
    omp = OMP(n_nonzero_coefs=n_nonzero_coefs, mode="batch", check_dict=True)
    Gamma = omp.encode(X_norm, D)
    logger.info(f"Encoded {X.shape[1]} patches -> Gamma shape {Gamma.shape}")
    return Gamma


# def encode_patches(
#     X: np.ndarray,
#     D: np.ndarray,
#     alpha: float = ALPHA,
#     max_iter: int = 500,
#     tol: float | None = 1e-8,
#     L0: float = 1.0,
#     eta: float = 2.0,
#     batch_size: int = 8192,
# ) -> np.ndarray:
#     """Sparse-code X against D via the reppi FISTA core solver.

#     Uses reppi.sparse.fista.core.fista_core directly (as FDDL's
#     solve_class_codes does) instead of the FISTA wrapper class: the
#     wrapper forces numpy arrays but its check-dict/soft-threshold
#     helpers are torch-only, so it raises TypeErrors on any input.
#     fista_core itself is backend-agnostic given torch-tensor closures.

#     Unlike FDDL's Eq. (7) sub-problem, this objective has no coupling
#     across columns of X (grad_f/prox_g are separable per sample), so it
#     is solved in column batches rather than as one whole-matrix FISTA
#     run -- a single run's live buffers (x_prev/y/x_k/grad, each
#     n_atoms x n_samples) would need tens of GB for this D/X size and
#     OOMs on GPU.
#     """
#     X_norm, _, _ = normalise_columns(X)
#     D_norm, _, _ = normalise_columns(D)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     D_t = torch.from_numpy(np.asarray(D_norm, dtype=np.float32)).to(device)
#     n_atoms = D_t.shape[1]
#     n_samples = X_norm.shape[1]
#     logger.info(f"Encoding on device: {device}")

#     Gamma = np.empty((n_atoms, n_samples), dtype=np.float32)
#     total_iters = 0

#     for start in range(0, n_samples, batch_size):
#         end = min(start + batch_size, n_samples)
#         X_t = torch.from_numpy(np.asarray(X_norm[:, start:end], dtype=np.float32)).to(device)

#         def grad_f(Z: torch.Tensor) -> torch.Tensor:
#             return 2.0 * (D_t.T @ (D_t @ Z - X_t))

#         def f(Z: torch.Tensor) -> float:
#             return float(torch.sum((D_t @ Z - X_t) ** 2))

#         def g(Z: torch.Tensor) -> float:
#             return float(alpha * torch.sum(torch.abs(Z)))

#         def prox_g(V: torch.Tensor, t: float) -> torch.Tensor:
#             return soft_threshold(V, alpha * t)

#         gamma0 = torch.zeros((n_atoms, end - start), dtype=torch.float32, device=device)

#         result = fista_core(
#             grad_f=grad_f,
#             prox_g=prox_g,
#             x0=gamma0,
#             f=f,
#             g=g,
#             mode="backtracking",
#             L0=L0,
#             eta=eta,
#             max_iter=max_iter,
#             tol=tol,
#         )
#         Gamma[:, start:end] = result.x.cpu().numpy()
#         total_iters += result.n_iter

#     logger.info(f"Encoded {X.shape[1]} patches -> Gamma shape {Gamma.shape} ({total_iters} total iters)")
#     return Gamma


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SPARSE_CODE_DIR.mkdir(parents=True, exist_ok=True)

    extract_unified(split=SPLIT)  # Ensure patches are extracted for the specified split
    # -- Load patches and labels ------------------------------------------------
    X, labels, scan_ids, _coords = load_unified_patch_matrix(split=SPLIT)

    # -- Load trained dictionary --------------------------------------------------
    D = load_dictionary(DICT_MODEL_PATH)

    # -- Sparse-code patches against the dictionary -------------------------------
    if (SPARSE_CODE_DIR / f"{SPLIT}_sparse_codes.npz_{MODEL}").exists():
        logger.info(f"Loading existing sparse codes from {SPARSE_CODE_DIR / f'{SPLIT}_sparse_codes.npz_{MODEL}'}")
        data = np.load(SPARSE_CODE_DIR / f"{SPLIT}_sparse_codes.npz_{MODEL}")
        Gamma = data["Gamma"]
    else:
        Gamma = encode_patches_omp(X, D)

        # -- Save sparse codes (dense) alongside labels for reuse ---------------------
        np.savez_compressed(SPARSE_CODE_DIR / f"{SPLIT}_sparse_codes.npz_{MODEL}", Gamma=Gamma, labels=labels)
        logger.info(f"Saved sparse codes -> {SPARSE_CODE_DIR / f'{SPLIT}_sparse_codes.npz_{MODEL}'}")


    # -- Inference and evaluate using SVM ------------------------------------------------------------------
    svm_clf = joblib.load(SVM_MODEL_PATH)
    logger.info(f"Inferencing from LinearSVC: {SVM_MODEL_PATH}")
    y_pred_svm = svm_clf.predict(Gamma.T)
    evaluate(y_pred_svm, labels)

    model_config = {
        "dict_model_path": str(DICT_MODEL_PATH),
        "svm_model_path": str(SVM_MODEL_PATH),
        "n_nonzero_coefs": N_NONZERO_COEFS,
        "alpha": ALPHA,
        "svm_params": svm_clf.get_params(),
    }
    save_classification_results(
        predictions=y_pred_svm,
        labels=labels,
        scan_ids=scan_ids,
        model_name=f"{MODEL}_svm",
        model_config=model_config,
        split=SPLIT,
    )



    # -- Inference and evaluate using Logistic Regression ---------------------------------------------------
    # logger.info("Inferencing from LogisticRegression...")
    # log_reg_clf = joblib.load(LOG_REG_MODEL_PATH)
    # y_pred_log_reg = log_reg_clf.predict(Gamma.T)
    # evaluate(y_pred_log_reg, labels)


if __name__ == "__main__":
    main()