# Critical Issues Found in AUROC Calculations

## 1. ❌ **BOOTSTRAP RESAMPLING NOT APPLIED** (bootstrap_values.py)
**File:** `models/ct-clip/scripts/bootstrap_values.py` (Lines 106-109)

### Issue:
```python
indices = np.random.choice(range(len(labels)), size=len(labels), replace=True)
#sampled_labels = labels[indices]           # ← COMMENTED OUT!
#sampled_predicted = predicted[indices]      # ← COMMENTED OUT!
sampled_labels = labels                       # ← USING ORIGINAL DATA
sampled_predicted = predicted                 # ← USING ORIGINAL DATA
```

### Problem:
- Random indices are generated but **never used**
- Bootstrap iterations use identical data (same labels & predictions every iteration)
- Creates 1000 identical AUROC values instead of bootstrapped distribution
- **All bootstrap confidence intervals are meaningless**

### Impact:
- Confidence intervals are artificially tight
- No actual estimate of variance/uncertainty
- Statistical analysis is invalid

---

## 2. ❌ **INDEX MISMATCH IN evaluate_external()** (eval.py)
**File:** `models/ct-clip/scripts/eval.py` (Lines 229-243)

### Issue:
```python
for i in range(num_classes):  # num_classes = 18
    if i != 13 and i!= 4:     # Skips indices 13 and 4
        
        if i ==1 or i==4:     # But this checks for i==4, which was skipped!
            label = y_true[:,counter]  # Uses 'counter' for indexing
            l1 = y_pred[:, 1]
            l2 = y_pred[:, 4]
            prob = np.maximum(l1, l2)  # Takes max of indices 1 and 4
        else:
            prob = y_pred[:,i]
            label = y_true[:,counter]  # Inconsistent indexing
        
        counter = counter + 1
```

### Problems:
1. **Index skipping contradiction**: Loop skips i=13 and i=4, but condition checks for i==4
2. **Inconsistent indexing**: 
   - `y_pred` indexed by raw `i` (can be 0-17)
   - `y_true` indexed by `counter` (which gets incremented selectively)
   - Creates misalignment between predictions and ground truth
3. **Special case logic unclear**: Merging predictions from indices 1 and 4 suggests data structure mismatch
4. **Counter logic broken**: Counter only increments when `i != 13 and i!= 4`, but the inner condition can never be true for i==4

### Impact:
- Wrong predictions paired with wrong ground truth labels
- AUROC values don't represent actual model performance for those classes
- Results are fundamentally unreliable

---

## 3. ⚠️ **MISSING INPUT VALIDATION** (eval.py)
**File:** `models/ct-clip/scripts/eval.py` (Lines 59-100)

### Issue:
The `plot_roc()` function doesn't validate input data:

```python
def plot_roc(y_pred, y_true, roc_name, plot_dir, plot=True):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)  # No validation
    roc_auc = auc(fpr, tpr)
```

### Potential Problems:
- No check if y_true has both 0s and 1s (binary classification requirement)
- No check for NaN values in predictions
- No check if data shapes match
- No check if predictions are in [0, 1] range
- Single class or constant predictions will cause `sklearn.metrics.auc()` to return unexpected values

### Impact:
- Silent failures or incorrect AUROC values
- Difficult to debug data quality issues
- Especially problematic if any class is imbalanced or has all same labels

---

## 4. ⚠️ **LACK OF ERROR HANDLING FOR IMBALANCED CLASSES** 

### Issue:
When a class has all 0s or all 1s in the dataset, ROC-AUC cannot be computed.

### Example:
- If a rare class appears in only 1 sample (or 0 samples) in evaluation set
- `roc_curve()` may behave unexpectedly
- `auc()` could return NaN or 0.5 depending on the data

### Suggested Fix:
Check if ground truth has both classes before computing AUROC:

```python
if len(np.unique(y_true)) < 2:
    return np.nan
```

---

## 5. ⚠️ **DATA SHAPE ASSUMPTIONS NOT VERIFIED**

### Issue in evaluate_internal():
```python
num_classes = y_pred.shape[-1]  # Assumes last dim is num_classes
```

If input arrays have unexpected shapes, indexing could fail silently or produce wrong results.

---

## Summary Table

| Issue | Location | Severity | Impact |
|-------|----------|----------|--------|
| Bootstrap not applied | bootstrap_values.py:106-109 | 🔴 CRITICAL | All confidence intervals invalid |
| Index mismatch | eval.py:229-243 | 🔴 CRITICAL | Wrong pred/label pairs for evaluation |
| No input validation | eval.py:59-100 | 🟠 HIGH | Silent failures possible |
| No imbalance handling | eval.py & bootstrap_values.py | 🟠 HIGH | NaN AUROC for rare classes |
| Shape assumptions | eval.py:165 | 🟡 MEDIUM | Edge cases could fail |

---

## Recommended Fixes

### Fix 1: Restore Bootstrap Resampling
Replace lines 106-109 in bootstrap_values.py:
```python
indices = np.random.choice(range(len(labels)), size=len(labels), replace=True)
sampled_labels = labels[indices]      # Apply resampling
sampled_predicted = predicted[indices]  # Apply resampling
```

### Fix 2: Fix evaluate_external() Index Logic
Clarify the intent: are you evaluating 16 classes (skipping 13 & 4) or 18 classes?

```python
# Option A: Evaluate all 18 classes
for i in range(num_classes):
    y_true_i = y_true[:, i]
    y_pred_i = y_pred[:, i]
    # Calculate AUROC for class i

# Option B: Evaluate 16 classes only  
class_indices = [i for i in range(num_classes) if i not in [4, 13]]
for i in class_indices:
    y_true_i = y_true[:, i]
    y_pred_i = y_pred[:, i]
    # Calculate AUROC for class i
```

### Fix 3: Add Input Validation
```python
def plot_roc(y_pred, y_true, roc_name, plot_dir, plot=True):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    # Validation
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}")
    
    if len(np.unique(y_true)) < 2:
        return None, None, None, np.nan  # Cannot compute ROC-AUC
    
    if np.any(np.isnan(y_pred)) or np.any(np.isnan(y_true)):
        raise ValueError("NaN values found in input data")
    
    # Rest of function...
```

---

## Questions to Verify

1. Is the bootstrap resampling being used elsewhere? Or was it intentionally disabled?
2. What is the purpose of merging predictions from indices 1 and 4 in evaluate_external()?
3. Should evaluate_external() evaluate all 18 classes or just 16?
4. Are the AUROC values you're seeing unusually high, low, or constant? This would indicate which issue is causing problems.
