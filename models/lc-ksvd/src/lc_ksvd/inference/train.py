import logging
import pickle

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from lc_ksvd.config import MODELS_DIR, SPARSE_CODE_DIR
from lc_ksvd.patch_extractor.patch_extraction import load_unified_patch_matrix
from lc_ksvd.metrics import normalise_columns  # adjust import path if different
from reppi import OMP
from reppi.sparse.fista.utils import soft_threshold

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

MODEL = "lcksvd"
DICT_MODEL_PATH = MODELS_DIR / f"unified_{MODEL}.pkl"
SVM_MODEL_PATH = MODELS_DIR / f"{MODEL}_svm_model.pkl"
GBM_MODEL_PATH = MODELS_DIR / f"{MODEL}_gbm_model.pkl"
XGB_MODEL_PATH = MODELS_DIR / f"{MODEL}_xgb_model.pkl"
LOGREG_MODEL_PATH = MODELS_DIR / f"{MODEL}_logreg_model.pkl"

N_NONZERO_COEFS = 10
ALPHA = 0.1
SPLIT = "train"
RANDOM_SEED = 42

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


def load_W(path=DICT_MODEL_PATH):
    """Load the trained payload and return the jointly-learned classifier W_."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    return getattr(model, "W_", None)

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

def train_logreg(gamma: np.ndarray, labels: np.ndarray):
    X = gamma.T
    pipeline = Pipeline([
        ("scaler", MaxAbsScaler()),
        ("logreg", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])
    param_grid = {"logreg__C": [0.1, 1.0, 10.0, 100.0]}
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1_macro", n_jobs=1, verbose=2)
    grid.fit(X, labels)
    print("\nBest parameters:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    return grid.best_estimator_


def train_gbm(gamma: np.ndarray, labels: np.ndarray):
    """Non-linear diagnostic classifier (HistGradientBoostingClassifier).

    Not expected to beat the linear/native classifiers above if LC-KSVD2's
    codes are already close to linearly separable (per ScSPM/LLC and the
    LC-KSVD literature) -- its main value is as a check: a large gap over
    LinearSVC/W_ signals the dictionary isn't discriminative enough yet,
    not that this should become the production classifier.
    """
    X = gamma.T
    clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_SEED)
    param_grid = {"max_depth": [3, 5, None], "learning_rate": [0.05, 0.1]}
    grid = GridSearchCV(clf, param_grid, cv=5, scoring="f1_macro", n_jobs=1, verbose=2)
    grid.fit(X, labels)
    print("\nBest parameters:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    return grid.best_estimator_


def train_xgb(gamma: np.ndarray, labels: np.ndarray):
    """Non-linear diagnostic classifier (XGBoost).

    Same diagnostic role as train_gbm (see its docstring) -- included
    alongside it rather than instead of it because XGBoost's level-wise
    tree growth is more conservative than LightGBM-style leaf-wise growth
    under class imbalance, and its explicit L1/L2 terms (reg_alpha/
    reg_lambda) give more regularisation control than
    HistGradientBoostingClassifier when a class (e.g. 2c/2d) has
    relatively few samples relative to normal.
    """
    X = gamma.T
    sample_weight = compute_sample_weight("balanced", labels)
    clf = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_SEED,
    )
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "reg_lambda": [1.0, 5.0],
    }
    grid = GridSearchCV(clf, param_grid, cv=5, scoring="f1_macro", n_jobs=1, verbose=2)
    grid.fit(X, labels, sample_weight=sample_weight)
    print("\nBest parameters:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    return grid.best_estimator_


def train_svm(gamma: np.ndarray, labels: np.ndarray):
    X = gamma.T  # (n_samples, n_features)

    pipeline = Pipeline([
        ("scaler", MaxAbsScaler()),
        ("svm", LinearSVC(dual=False, max_iter=10000))
    ])

    param_grid = {
        "svm__C": [10.0, 100.0]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=1,
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
        Gamma = encode_patches_omp(X, D)
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

    # -- Sanity-check LC-KSVD2's own jointly-learned classifier (W_) -----------------
    W = load_W(DICT_MODEL_PATH)
    if W is not None:
        y_pred_w = np.argmax(W @ Gamma, axis=0)
        train_acc_w = float(np.mean(y_pred_w == labels))
        logger.info(f"Native LC-KSVD2 classifier (W_) train accuracy: {train_acc_w:.4f}")
    else:
        logger.info("Loaded model has no W_ (not lcksvd2) -- skipping native-classifier check.")

    # -- Train non-linear diagnostic classifier (HistGradientBoostingClassifier) ----
    logger.info("Training HistGradientBoostingClassifier...")
    if(GBM_MODEL_PATH.exists()):
        logger.info(f"GBM model exists at: {GBM_MODEL_PATH}")
    else:
        gbm_clf = train_gbm(Gamma, labels)
        joblib.dump(gbm_clf, GBM_MODEL_PATH)
        logger.info(f"Saved GBM model -> {GBM_MODEL_PATH}")

    # -- Train non-linear diagnostic classifier (XGBoost) ----------------------------
    logger.info("Training XGBoost...")
    if(XGB_MODEL_PATH.exists()):
        logger.info(f"XGB model exists at: {XGB_MODEL_PATH}")
    else:
        xgb_clf = train_xgb(Gamma, labels)
        joblib.dump(xgb_clf, XGB_MODEL_PATH)
        logger.info(f"Saved XGB model -> {XGB_MODEL_PATH}")

    # -- Train Logistic Regression ----------------------------------------------------
    logger.info("Training LogisticRegression...")
    if(LOGREG_MODEL_PATH.exists()):
        logger.info(f"LogReg model exists at: {LOGREG_MODEL_PATH}")
    else:
        logreg_clf = train_logreg(Gamma, labels)
        joblib.dump(logreg_clf, LOGREG_MODEL_PATH)
        logger.info(f"Saved LogReg model -> {LOGREG_MODEL_PATH}")

if __name__ == "__main__":
    main()
