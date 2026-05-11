"""
Continual Learning metrics: Forgetting, Backward Transfer.
Also contains array utilities used in evaluation.
"""
import numpy as np


def forgetting(results: list[list[float]]) -> float:
    """
    Compute average forgetting after learning the last task.

    Args:
        results: results[t][i] = accuracy on task i after learning t tasks.
                 Outer list indexed by sequential task (0..T-1).
    Returns:
        Mean forgetting across tasks 0..T-2.
    """
    n_tasks = len(results)
    # Pad shorter rows with 0.0
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    best_per_task = np.max(np_res, axis=0)
    fgt = [best_per_task[i] - results[-1][i] for i in range(n_tasks - 1)]
    return float(np.mean(fgt))


def backward_transfer(results: list[list[float]]) -> float:
    """
    Compute average Backward Transfer (BWT).
    BWT > 0 means positive backward transfer (rare).
    BWT < 0 means forgetting.

    Args:
        results: Same format as forgetting().
    Returns:
        Mean BWT across tasks 0..T-2.
    """
    n_tasks = len(results)
    bwt = [results[-1][i] - results[i][i] for i in range(n_tasks - 1)]
    return float(np.mean(bwt))


def pad_numpy_arrays(arrays: list[np.ndarray],
                     pad_value: float = 0.0) -> np.ndarray:
    """
    Pad a list of arrays with varying shapes to the same shape and stack.
    Used when per-task probability arrays have different number of columns.
    """
    max_dim = max(a.ndim for a in arrays)
    arrays = [a.reshape((1,) * (max_dim - a.ndim) + a.shape) for a in arrays]
    max_shape = np.max([a.shape for a in arrays], axis=0)
    padded = []
    for a in arrays:
        pw = [(0, max_shape[i] - a.shape[i]) for i in range(max_dim)]
        padded.append(np.pad(a, pw, constant_values=pad_value))
    return np.stack(padded)