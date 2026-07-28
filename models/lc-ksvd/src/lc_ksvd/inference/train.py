import logging
import pickle

import joblib
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from lc_ksvd.config import MODELS_DIR, SPARSE_CODE_DIR
from lc_ksvd.patch_extractor.patch_extraction import load_unified_patch_matrix
from lc_ksvd.metrics import normalise_columns  # adjust import path if different
from reppi import fista_core
from reppi.sparse.fista.utils import soft_threshold

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DICT_MODEL_PATH = MODELS_DIR / "unified_fddl.pkl"
SVM_MODEL_PATH = MODELS_DIR / "svm_model_fddl.pkl"

N_NONZERO_COEFS = 10
ALPHA = 0.1
SPLIT = "train"
MODEL = "fddl"

def load_dictionary(path=DICT_MODEL_PATH) -> np.ndarray:
    """Load the trained IncrementalFrozenDictionary payload and return D."""
    # with open(path, "rb") as f:
    #     payload = pickle.load(f)
    # model = payload["model"]
    model = np.load("/home/chest_ct/code/models/lc-ksvd/src/lc_ksvd/outputs/checkpoints/ksvd_checkpoint.npz", allow_pickle=True)
    D = model["D"]
    if D is None:
        raise ValueError(f"Loaded model at {path} has no fitted dictionary (D_ is None).")
    logger.info(f"Loaded dictionary D of shape {D.shape} from {path}")
    return D


def encode_patches(
    X: np.ndarray,
    D: np.ndarray,
    alpha: float = ALPHA,
    max_iter: int = 500,
    tol: float | None = 1e-8,
    L0: float = 1.0,
    eta: float = 2.0,
    batch_size: int = 8192,
) -> np.ndarray:
    """Sparse-code X against D via the reppi FISTA core solver.

    Uses reppi.sparse.fista.core.fista_core directly (as FDDL's
    solve_class_codes does) instead of the FISTA wrapper class: the
    wrapper forces numpy arrays but its check-dict/soft-threshold
    helpers are torch-only, so it raises TypeErrors on any input.
    fista_core itself is backend-agnostic given torch-tensor closures.

    Unlike FDDL's Eq. (7) sub-problem, this objective has no coupling
    across columns of X (grad_f/prox_g are separable per sample), so it
    is solved in column batches rather than as one whole-matrix FISTA
    run -- a single run's live buffers (x_prev/y/x_k/grad, each
    n_atoms x n_samples) would need tens of GB for this D/X size and
    OOMs on GPU.
    """
    X_norm, _, _ = normalise_columns(X)
    D_norm, _, _ = normalise_columns(D)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D_t = torch.from_numpy(np.asarray(D_norm, dtype=np.float32)).to(device)
    n_atoms = D_t.shape[1]
    n_samples = X_norm.shape[1]
    logger.info(f"Encoding on device: {device}")

    Gamma = np.empty((n_atoms, n_samples), dtype=np.float32)
    total_iters = 0

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        X_t = torch.from_numpy(np.asarray(X_norm[:, start:end], dtype=np.float32)).to(device)

        def grad_f(Z: torch.Tensor) -> torch.Tensor:
            return 2.0 * (D_t.T @ (D_t @ Z - X_t))

        def f(Z: torch.Tensor) -> float:
            return float(torch.sum((D_t @ Z - X_t) ** 2))

        def g(Z: torch.Tensor) -> float:
            return float(alpha * torch.sum(torch.abs(Z)))

        def prox_g(V: torch.Tensor, t: float) -> torch.Tensor:
            return soft_threshold(V, alpha * t)

        gamma0 = torch.zeros((n_atoms, end - start), dtype=torch.float32, device=device)

        result = fista_core(
            grad_f=grad_f,
            prox_g=prox_g,
            x0=gamma0,
            f=f,
            g=g,
            mode="backtracking",
            L0=L0,
            eta=eta,
            max_iter=max_iter,
            tol=tol,
        )
        Gamma[:, start:end] = result.x.cpu().numpy()
        total_iters += result.n_iter

    logger.info(f"Encoded {X.shape[1]} patches -> Gamma shape {Gamma.shape} ({total_iters} total iters)")
    return Gamma


def train_svm(gamma: np.ndarray, labels: np.ndarray):
    X = gamma.T  # (n_samples, n_features)

    pipeline = Pipeline([
        ("scaler", MaxAbsScaler()),
        ("svm", LinearSVC(dual=False, max_iter=10000))
    ])

    param_grid = {
        "svm__C": [0.1, 1.0, 10.0]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=2,
        verbose=2,
        pre_dispatch=1,
        refit=True
    )

    grid.fit(X, labels)

    print("\nBest parameters:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    return grid.best_estimator_


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SPARSE_CODE_DIR.mkdir(parents=True, exist_ok=True)

    # -- Load patches and labels ------------------------------------------------
    X, labels, _scan_ids, _coords = load_unified_patch_matrix(split=SPLIT)

    # -- Load trained dictionary --------------------------------------------------
    D = load_dictionary(DICT_MODEL_PATH)

    # -- Sparse-code patches against the dictionary -------------------------------
    if (SPARSE_CODE_DIR / f"{SPLIT}_sparse_codes_{MODEL}.npz").exists():
        logger.info(f"Loading existing sparse codes from {SPARSE_CODE_DIR / f'{SPLIT}_sparse_codes_{MODEL}.npz'}")
        data = np.load(SPARSE_CODE_DIR / f"{SPLIT}_sparse_codes_{MODEL}.npz")
        Gamma = data["Gamma"]
    else:
        Gamma = encode_patches(X, D)
        # -- Save sparse codes (dense) alongside labels for reuse ---------------------
        np.savez_compressed(SPARSE_CODE_DIR / f"{SPLIT}_sparse_codes_{MODEL}.npz", Gamma=Gamma, labels=labels)
        logger.info(f"Saved sparse codes -> {SPARSE_CODE_DIR / f'{SPLIT}_sparse_codes_{MODEL}.npz'}")

    # -- Train SVM ------------------------------------------------------------------
    logger.info("Training LinearSVC...")
    if(SVM_MODEL_PATH.exists()):
        logger.info(f"SVM model exists at: {SVM_MODEL_PATH}")
    else:
        svm_clf = train_svm(Gamma, labels)
        joblib.dump(svm_clf, SVM_MODEL_PATH)
        logger.info(f"Saved SVM model -> {SVM_MODEL_PATH}")

if __name__ == "__main__":
    main()
