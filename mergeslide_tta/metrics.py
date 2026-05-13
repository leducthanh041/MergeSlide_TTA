import numpy as np


def forgetting(results: list[list[float]]) -> float:
    """
    Tính average forgetting sau khi học xong task cuối.

    results là triangular list of lists:
        results[0] = [bacc_t0, bacc_t1]              # sau merge task 1
        results[1] = [bacc_t0, bacc_t1, bacc_t2]     # sau merge task 2
        ...
        results[T-2] = [bacc_t0, ..., bacc_tT-1]     # sau merge task T-1

    Số row = NUM_TASKS - 1 = 5
    Row dài nhất = NUM_TASKS = 6  ← đây là max_len cần pad đến
    """
    n_rows  = len(results)                        # = NUM_TASKS - 1 = 5
    max_len = max(len(r) for r in results)        # = NUM_TASKS = 6  ← FIX

    # Pad tất cả row về max_len trước khi tạo numpy array
    for i in range(n_rows):
        results[i] += [0.0] * (max_len - len(results[i]))

    np_res        = np.array(results)             # shape (5, 6) — OK
    best_per_task = np.max(np_res, axis=0)        # peak BAcc mỗi task

    fgt = [best_per_task[i] - results[-1][i] for i in range(n_rows)]
    return float(np.mean(fgt))


def backward_transfer(results: list[list[float]]) -> float:
    """
    Tính average Backward Transfer (BWT).

    results[i][i] = BAcc task i ngay sau khi học task i+1
                    (diagonal của triangular matrix, offset 1).
    results[-1][i] = BAcc task i sau khi học hết tất cả task.
    """
    n_rows = len(results)   # = NUM_TASKS - 1
    bwt = [results[-1][i] - results[i][i] for i in range(n_rows)]
    return float(np.mean(bwt))


def pad_numpy_arrays(arrays: list[np.ndarray],
                     pad_value: float = 0.0) -> np.ndarray:
    """
    Pad list of arrays có shape khác nhau về cùng shape rồi stack.
    Dùng khi per-task probability arrays có số cột khác nhau
    (RCC có 3 class, các task khác có 2 class).
    """
    max_dim   = max(a.ndim for a in arrays)
    arrays    = [a.reshape((1,) * (max_dim - a.ndim) + a.shape) for a in arrays]
    max_shape = np.max([a.shape for a in arrays], axis=0)
    padded    = []
    for a in arrays:
        pw = [(0, max_shape[i] - a.shape[i]) for i in range(max_dim)]
        padded.append(np.pad(a, pw, constant_values=pad_value))
    return np.stack(padded)