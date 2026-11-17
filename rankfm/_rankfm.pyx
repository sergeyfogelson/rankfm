#!python
#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

# -----------------------
# [C/Python] dependencies
# -----------------------

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport log, exp, pow

cimport cython
cimport rankfm.mt19937ar as mt

import numpy as np

# --------------------
# [C] helper functions
# --------------------

cdef inline int lsearch(int item, const int *items, int n) noexcept nogil:
    """linear search for a given item in a sorted array of items"""

    cdef int i
    for i in range(n):
        if item == items[i]:
            return 1
    return 0


cdef inline int bsearch(int item, const int *items, int n) noexcept nogil:
    """binary search for a given item in a sorted array of items"""

    cdef int lo = 0
    cdef int hi = n - 1
    cdef int md

    while lo <= hi:
        md = lo + ((hi - lo) >> 1)
        if items[md] == item:
            return 1
        elif items[md] < item:
            lo = md + 1
        else:
            hi = md - 1
    return 0


cdef inline int item_observed(int item, const int *items, int n) noexcept nogil:
    """check if item is in sorted array using linear search for small n and binary search otherwise"""

    if n <= 16:
        return lsearch(item, items, n)
    return bsearch(item, items, n)


cdef inline void shuffle_int_array(int *arr, int n) noexcept nogil:
    """in-place Fisher-Yates shuffle using Mersenne Twister RNG without holding the GIL"""

    cdef int i, j, tmp
    for i in range(n - 1, 0, -1):
        j = <int>(mt.genrand_int32() % <unsigned long>(i + 1))
        tmp = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp


cdef inline float safe_sigmoid_neg(float x) noexcept nogil:
    """numerically stable 1 / (1 + exp(x))"""

    if x > 20.0:
        return exp(-x)
    elif x < -20.0:
        return 1.0
    return 1.0 / (exp(x) + 1.0)


cdef inline float safe_log_sigmoid(float x) noexcept nogil:
    """numerically stable log(1 / (1 + exp(-x)))"""

    if x > 20.0:
        return -exp(-x)
    elif x < -20.0:
        return x
    return -log(1.0 + exp(-x))


cdef inline float dot_product(int F, const float *a, const float *b) noexcept nogil:
    """compute inner product between two F-dimensional contiguous vectors"""

    cdef int f
    cdef float res = 0.0
    for f in range(F):
        res += a[f] * b[f]
    return res


cdef inline void compute_user_embedding_csr(
    int F,
    int u,
    const float *v_u_ptr,
    const float *v_uf_ptr,
    const int *uf_indptr,
    const int *uf_indices,
    const float *uf_data,
    int use_v_u,
    int u_has_uf,
    float *out_u_emb
) noexcept nogil:
    """compute effective user embedding using CSR sparse user features and contiguous SIMD-friendly loops"""

    cdef int f, idx, p
    cdef float val
    cdef const float *v_u_row = &v_u_ptr[u * F]
    cdef const float *v_uf_row

    if use_v_u:
        for f in range(F):
            out_u_emb[f] = v_u_row[f]
    else:
        for f in range(F):
            out_u_emb[f] = 0.0

    if u_has_uf:
        for idx in range(uf_indptr[u], uf_indptr[u + 1]):
            p = uf_indices[idx]
            val = uf_data[idx]
            v_uf_row = &v_uf_ptr[p * F]
            for f in range(F):
                out_u_emb[f] += val * v_uf_row[f]


cdef inline float compute_item_embedding_and_bias_csr(
    int F,
    int item,
    const float *w_i_ptr,
    const float *w_if_ptr,
    const float *v_i_ptr,
    const float *v_if_ptr,
    const int *if_indptr,
    const int *if_indices,
    const float *if_data,
    int x_if_any,
    float *out_i_emb
) noexcept nogil:
    """compute item scalar bias and effective item embedding using CSR sparse item features"""

    cdef int f, idx, q
    cdef float val
    cdef float bias = w_i_ptr[item]
    cdef const float *v_i_row = &v_i_ptr[item * F]
    cdef const float *v_if_row

    for f in range(F):
        out_i_emb[f] = v_i_row[f]

    if x_if_any:
        for idx in range(if_indptr[item], if_indptr[item + 1]):
            q = if_indices[idx]
            val = if_data[idx]
            bias += val * w_if_ptr[q]
            v_if_row = &v_if_ptr[q * F]
            for f in range(F):
                out_i_emb[f] += val * v_if_row[f]

    return bias


cdef inline void heap_sift_down(float *hs, int *hi, int size, int pos) noexcept nogil:
    """sift-down on a min-heap of (score, item_idx) pairs"""

    cdef int child, right
    cdef float s_val = hs[pos]
    cdef int i_val = hi[pos]

    while True:
        child = (pos << 1) + 1
        if child >= size:
            break
        right = child + 1
        if right < size and hs[right] < hs[child]:
            child = right
        if s_val <= hs[child]:
            break
        hs[pos] = hs[child]
        hi[pos] = hi[child]
        pos = child

    hs[pos] = s_val
    hi[pos] = i_val


cdef inline void heap_sift_up(float *hs, int *hi, int pos) noexcept nogil:
    """sift-up on a min-heap of (score, item_idx) pairs"""

    cdef int parent
    cdef float s_val = hs[pos]
    cdef int i_val = hi[pos]

    while pos > 0:
        parent = (pos - 1) >> 1
        if hs[parent] <= s_val:
            break
        hs[pos] = hs[parent]
        hi[pos] = hi[parent]
        pos = parent

    hs[pos] = s_val
    hi[pos] = i_val


# -------------------------
# [Python] helper functions
# -------------------------

def dense_to_csr(float[:, ::1] mat):
    """convert a 2D float32 feature matrix into (indptr, indices, data) CSR arrays"""

    arr = np.asarray(mat)
    n_rows = arr.shape[0]
    nz_rows, nz_cols = np.nonzero(arr)
    if len(nz_rows) == 0:
        indptr = np.zeros(n_rows + 1, dtype=np.int32)
        indices = np.zeros(1, dtype=np.int32)
        data = np.zeros(1, dtype=np.float32)
        return indptr, indices, data, 0

    counts = np.bincount(nz_rows, minlength=n_rows)
    indptr = np.empty(n_rows + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, dtype=np.int32, out=indptr[1:])
    indices = np.ascontiguousarray(nz_cols, dtype=np.int32)
    data = np.ascontiguousarray(arr[nz_rows, nz_cols], dtype=np.float32)
    return indptr, indices, data, 1


def assert_finite(w_i, w_if, v_u, v_i, v_uf, v_if):
    """assert all model weights are finite"""

    assert np.isfinite(np.sum(w_i)), "item weights [w_i] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(w_if)), "item feature weights [w_if] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_u)), "user factors [v_u] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_i)), "item factors [v_i] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_uf)), "user-feature factors [v_uf] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_if)), "item-feature factors [v_if] are not finite - try decreasing feature/sample_weight magnitudes"


def reg_penalty(alpha, beta, w_i, w_if, v_u, v_i, v_uf, v_if):
    """calculate the total regularization penalty for all model weights"""

    penalty = 0.0
    penalty += np.sum(alpha * np.square(w_i))
    penalty += np.sum(alpha * np.square(v_u))
    penalty += np.sum(alpha * np.square(v_i))
    penalty += np.sum(beta * np.square(w_if))
    penalty += np.sum(beta * np.square(v_uf))
    penalty += np.sum(beta * np.square(v_if))
    return penalty


# --------------------------------
# [RankFM] core modeling functions
# --------------------------------

def _fit(
    int[:, ::1] interactions,
    float[::1] sample_weight,
    int[::1] ui_indptr,
    int[::1] ui_indices,
    float[:, ::1] x_uf,
    float[:, ::1] x_if,
    float[::1] w_i,
    float[::1] w_if,
    float[:, ::1] v_u,
    float[:, ::1] v_i,
    float[:, ::1] v_uf,
    float[:, ::1] v_if,
    float alpha,
    float beta,
    float learning_rate,
    str learning_schedule,
    float learning_exponent,
    int max_samples,
    int epochs,
    bint verbose,
    float user_dropout=0.0,
    int seed=1492
):
    """learn model weights via nogil BPR/WARP SGD with CSR sparse feature traversal and contiguous SIMD updates"""

    cdef float MARGIN = 1.0
    cdef int N = interactions.shape[0]
    cdef int I = v_i.shape[0]
    cdef int F = v_u.shape[1]

    if N == 0 or I <= 1:
        return

    cdef int schedule_mode
    if learning_schedule == 'constant':
        schedule_mode = 0
    elif learning_schedule == 'invscaling':
        schedule_mode = 1
    else:
        raise ValueError('unknown [learning_schedule]')

    # Pre-extract sparse CSR structures for x_uf and x_if once prior to training
    uf_indptr_np, uf_indices_np, uf_data_np, x_uf_any_py = dense_to_csr(x_uf)
    if_indptr_np, if_indices_np, if_data_np, x_if_any_py = dense_to_csr(x_if)

    cdef int[::1] uf_indptr = uf_indptr_np
    cdef int[::1] uf_indices = uf_indices_np
    cdef float[::1] uf_data = uf_data_np
    cdef int[::1] if_indptr = if_indptr_np
    cdef int[::1] if_indices = if_indices_np
    cdef float[::1] if_data = if_data_np
    cdef int x_uf_any = x_uf_any_py
    cdef int x_if_any = x_if_any_py

    shuffle_index = np.arange(N, dtype=np.int32)
    cdef int[::1] shuffle_index_mv = shuffle_index

    # Raw C pointers for zero-overhead nogil access
    cdef const int *intx_ptr = &interactions[0, 0]
    cdef const float *sw_ptr = &sample_weight[0]
    cdef const int *ui_indptr_ptr = &ui_indptr[0]
    cdef const int *ui_indices_ptr = &ui_indices[0] if ui_indices.shape[0] > 0 else NULL

    cdef const int *uf_indptr_ptr = &uf_indptr[0]
    cdef const int *uf_indices_ptr = &uf_indices[0]
    cdef const float *uf_data_ptr = &uf_data[0]

    cdef const int *if_indptr_ptr = &if_indptr[0]
    cdef const int *if_indices_ptr = &if_indices[0]
    cdef const float *if_data_ptr = &if_data[0]

    cdef float *w_i_ptr = &w_i[0]
    cdef float *w_if_ptr = &w_if[0]
    cdef float *v_u_ptr = &v_u[0, 0]
    cdef float *v_i_ptr = &v_i[0, 0]
    cdef float *v_uf_ptr = &v_uf[0, 0]
    cdef float *v_if_ptr = &v_if[0, 0]
    cdef int *shuf_ptr = &shuffle_index_mv[0]

    cdef float *u_emb = <float*>malloc(F * sizeof(float))
    cdef float *i_emb = <float*>malloc(F * sizeof(float))
    cdef float *cand_emb = <float*>malloc(F * sizeof(float))
    cdef float *j_emb = <float*>malloc(F * sizeof(float))
    cdef float *diff_ij = <float*>malloc(F * sizeof(float))

    cdef int epoch, r, row, u, i, j, f, p, q, sampled, min_index, n_seen_u, u_has_uf, use_v_u
    cdef int idx, idx_i, end_i, idx_j, end_j
    cdef float eta, log_likelihood, sw, ut_ui, ut_uj, bias_i, bias_cand
    cdef float pairwise_utility, min_pairwise_utility, multiplier, d_outer
    cdef float denom_log_I = log(I)
    cdef float d_reg_a = 2.0 * alpha
    cdef float d_reg_b = 2.0 * beta
    cdef float scale_step, reg_a_keep, reg_b_keep, step_uf, step_if, diff_x_if, u_f
    cdef float *v_u_row
    cdef float *v_i_row
    cdef float *v_j_row
    cdef float *v_uf_row
    cdef float *v_if_row
    cdef const int *u_seen_ptr

    mt.init_genrand(seed)

    try:
        for epoch in range(epochs):
            if schedule_mode == 0:
                eta = learning_rate
            else:
                eta = learning_rate / pow(epoch + 1, learning_exponent)

            reg_a_keep = 1.0 - eta * d_reg_a
            reg_b_keep = 1.0 - eta * d_reg_b
            log_likelihood = 0.0

            with nogil:
                shuffle_int_array(shuf_ptr, N)

                for r in range(N):
                    row = shuf_ptr[r]
                    u = intx_ptr[row * 2]
                    i = intx_ptr[row * 2 + 1]
                    sw = sw_ptr[row]

                    n_seen_u = ui_indptr_ptr[u + 1] - ui_indptr_ptr[u]
                    if n_seen_u >= I:
                        continue
                    u_seen_ptr = &ui_indices_ptr[ui_indptr_ptr[u]] if ui_indices_ptr != NULL else NULL

                    u_has_uf = 1 if (x_uf_any and uf_indptr_ptr[u + 1] > uf_indptr_ptr[u]) else 0
                    if u_has_uf and user_dropout > 0.0 and mt.genrand_real2() < user_dropout:
                        use_v_u = 0
                    else:
                        use_v_u = 1

                    compute_user_embedding_csr(
                        F, u, v_u_ptr, v_uf_ptr,
                        uf_indptr_ptr, uf_indices_ptr, uf_data_ptr,
                        use_v_u, u_has_uf, u_emb
                    )
                    bias_i = compute_item_embedding_and_bias_csr(
                        F, i, w_i_ptr, w_if_ptr, v_i_ptr, v_if_ptr,
                        if_indptr_ptr, if_indices_ptr, if_data_ptr,
                        x_if_any, i_emb
                    )
                    ut_ui = bias_i + dot_product(F, u_emb, i_emb)

                    min_index = -1
                    min_pairwise_utility = 1e6

                    for sampled in range(1, max_samples + 1):
                        while True:
                            j = <int>(mt.genrand_int32() % <unsigned long>I)
                            if u_seen_ptr == NULL or not item_observed(j, u_seen_ptr, n_seen_u):
                                break

                        bias_cand = compute_item_embedding_and_bias_csr(
                            F, j, w_i_ptr, w_if_ptr, v_i_ptr, v_if_ptr,
                            if_indptr_ptr, if_indices_ptr, if_data_ptr,
                            x_if_any, cand_emb
                        )
                        ut_uj = bias_cand + dot_product(F, u_emb, cand_emb)
                        pairwise_utility = ut_ui - ut_uj

                        if min_index == -1 or pairwise_utility < min_pairwise_utility:
                            min_index = j
                            min_pairwise_utility = pairwise_utility
                            memcpy(j_emb, cand_emb, F * sizeof(float))

                        if pairwise_utility < MARGIN:
                            break

                    j = min_index
                    pairwise_utility = min_pairwise_utility
                    multiplier = log((I - 1) / sampled) / denom_log_I
                    if verbose:
                        log_likelihood += safe_log_sigmoid(pairwise_utility)

                    d_outer = safe_sigmoid_neg(pairwise_utility)
                    scale_step = eta * sw * multiplier * d_outer

                    # Update item scalar biases
                    w_i_ptr[i] = w_i_ptr[i] * reg_a_keep + scale_step
                    w_i_ptr[j] = w_i_ptr[j] * reg_a_keep - scale_step

                    # Contiguous SIMD-friendly factor updates
                    for f in range(F):
                        diff_ij[f] = i_emb[f] - j_emb[f]

                    if use_v_u:
                        v_u_row = &v_u_ptr[u * F]
                        for f in range(F):
                            v_u_row[f] = v_u_row[f] * reg_a_keep + scale_step * diff_ij[f]

                    v_i_row = &v_i_ptr[i * F]
                    v_j_row = &v_i_ptr[j * F]
                    for f in range(F):
                        u_f = u_emb[f]
                        v_i_row[f] = v_i_row[f] * reg_a_keep + scale_step * u_f
                        v_j_row[f] = v_j_row[f] * reg_a_keep - scale_step * u_f

                    # Sparse CSR user-feature factor updates (outer loop over non-zero p, inner loop contiguous over f)
                    if u_has_uf:
                        for idx in range(uf_indptr_ptr[u], uf_indptr_ptr[u + 1]):
                            p = uf_indices_ptr[idx]
                            step_uf = scale_step * uf_data_ptr[idx]
                            v_uf_row = &v_uf_ptr[p * F]
                            for f in range(F):
                                v_uf_row[f] = v_uf_row[f] * reg_b_keep + step_uf * diff_ij[f]

                    # Sparse 2-pointer merge over non-zero item features of (i, j) for w_if and v_if updates
                    if x_if_any:
                        idx_i = if_indptr_ptr[i]
                        end_i = if_indptr_ptr[i + 1]
                        idx_j = if_indptr_ptr[j]
                        end_j = if_indptr_ptr[j + 1]

                        while idx_i < end_i or idx_j < end_j:
                            if idx_j >= end_j or (idx_i < end_i and if_indices_ptr[idx_i] < if_indices_ptr[idx_j]):
                                q = if_indices_ptr[idx_i]
                                diff_x_if = if_data_ptr[idx_i]
                                idx_i += 1
                            elif idx_i >= end_i or if_indices_ptr[idx_j] < if_indices_ptr[idx_i]:
                                q = if_indices_ptr[idx_j]
                                diff_x_if = -if_data_ptr[idx_j]
                                idx_j += 1
                            else:
                                q = if_indices_ptr[idx_i]
                                diff_x_if = if_data_ptr[idx_i] - if_data_ptr[idx_j]
                                idx_i += 1
                                idx_j += 1

                            if diff_x_if != 0.0:
                                step_if = scale_step * diff_x_if
                                w_if_ptr[q] = w_if_ptr[q] * reg_b_keep + step_if
                                v_if_row = &v_if_ptr[q * F]
                                for f in range(F):
                                    v_if_row[f] = v_if_row[f] * reg_b_keep + step_if * u_emb[f]

            assert_finite(w_i, w_if, v_u, v_i, v_uf, v_if)

            if verbose:
                penalty = reg_penalty(alpha, beta, w_i, w_if, v_u, v_i, v_uf, v_if)
                log_likelihood = round(log_likelihood - penalty, 2)
                print("\ntraining epoch:", epoch)
                print("log likelihood:", log_likelihood)

    finally:
        free(u_emb)
        free(i_emb)
        free(cand_emb)
        free(j_emb)
        free(diff_ij)


def _fold_in(
    int[:, ::1] interactions,
    float[::1] sample_weight,
    int[::1] ui_indptr,
    int[::1] ui_indices,
    float[:, ::1] u_feat_emb,
    float[::1] item_bias,
    float[:, ::1] v_i_eff,
    float[:, ::1] v_u,
    float alpha,
    float learning_rate,
    str learning_schedule,
    float learning_exponent,
    int max_samples,
    int epochs,
    int seed=1492
):
    """learn or refine user latent factors [v_u] in nogil C while holding all item and feature weights fixed"""

    cdef float MARGIN = 1.0
    cdef int N = interactions.shape[0]
    cdef int I = v_i_eff.shape[0]
    cdef int F = v_u.shape[1]

    if N == 0 or I <= 1 or epochs <= 0:
        return

    cdef int schedule_mode
    if learning_schedule == 'constant':
        schedule_mode = 0
    elif learning_schedule == 'invscaling':
        schedule_mode = 1
    else:
        raise ValueError('unknown [learning_schedule]')

    shuffle_index = np.arange(N, dtype=np.int32)
    cdef int[::1] shuffle_index_mv = shuffle_index

    cdef const int *intx_ptr = &interactions[0, 0]
    cdef const float *sw_ptr = &sample_weight[0]
    cdef const int *ui_indptr_ptr = &ui_indptr[0]
    cdef const int *ui_indices_ptr = &ui_indices[0] if ui_indices.shape[0] > 0 else NULL
    cdef const float *u_feat_ptr = &u_feat_emb[0, 0]
    cdef const float *item_bias_ptr = &item_bias[0]
    cdef const float *v_i_eff_ptr = &v_i_eff[0, 0]
    cdef float *v_u_ptr = &v_u[0, 0]
    cdef int *shuf_ptr = &shuffle_index_mv[0]

    cdef float *u_emb = <float*>malloc(F * sizeof(float))

    cdef int epoch, r, row, u, i, j, f, sampled, min_index, n_seen_u
    cdef float eta, reg_a_keep, scale_step, sw, ut_ui, ut_uj, pairwise_utility, min_pairwise_utility, multiplier, d_outer
    cdef float denom_log_I = log(I)
    cdef float d_reg_a = 2.0 * alpha
    cdef float *v_u_row
    cdef const float *u_feat_row
    cdef const float *v_i_row
    cdef const float *v_j_row
    cdef const int *u_seen_ptr

    mt.init_genrand(seed)

    try:
        with nogil:
            for epoch in range(epochs):
                if schedule_mode == 0:
                    eta = learning_rate
                else:
                    eta = learning_rate / pow(epoch + 1, learning_exponent)

                reg_a_keep = 1.0 - eta * d_reg_a
                shuffle_int_array(shuf_ptr, N)

                for r in range(N):
                    row = shuf_ptr[r]
                    u = intx_ptr[row * 2]
                    i = intx_ptr[row * 2 + 1]
                    sw = sw_ptr[row]

                    n_seen_u = ui_indptr_ptr[u + 1] - ui_indptr_ptr[u]
                    if n_seen_u >= I:
                        continue
                    u_seen_ptr = &ui_indices_ptr[ui_indptr_ptr[u]] if ui_indices_ptr != NULL else NULL

                    v_u_row = &v_u_ptr[u * F]
                    u_feat_row = &u_feat_ptr[u * F]
                    for f in range(F):
                        u_emb[f] = v_u_row[f] + u_feat_row[f]

                    v_i_row = &v_i_eff_ptr[i * F]
                    ut_ui = item_bias_ptr[i] + dot_product(F, u_emb, v_i_row)

                    min_index = -1
                    min_pairwise_utility = 1e6

                    for sampled in range(1, max_samples + 1):
                        while True:
                            j = <int>(mt.genrand_int32() % <unsigned long>I)
                            if u_seen_ptr == NULL or not item_observed(j, u_seen_ptr, n_seen_u):
                                break

                        ut_uj = item_bias_ptr[j] + dot_product(F, u_emb, &v_i_eff_ptr[j * F])
                        pairwise_utility = ut_ui - ut_uj

                        if min_index == -1 or pairwise_utility < min_pairwise_utility:
                            min_index = j
                            min_pairwise_utility = pairwise_utility

                        if pairwise_utility < MARGIN:
                            break

                    j = min_index
                    pairwise_utility = min_pairwise_utility
                    multiplier = log((I - 1) / sampled) / denom_log_I
                    d_outer = safe_sigmoid_neg(pairwise_utility)
                    scale_step = eta * sw * multiplier * d_outer
                    v_j_row = &v_i_eff_ptr[j * F]

                    for f in range(F):
                        v_u_row[f] = v_u_row[f] * reg_a_keep + scale_step * (v_i_row[f] - v_j_row[f])
    finally:
        free(u_emb)


def _predict(
    int[::1] u_idx,
    int[::1] i_idx,
    float[:, ::1] u_emb,
    float[::1] item_bias,
    float[:, ::1] v_i_eff
):
    """score (user, item) pairs in nogil C using integer indices (-1 indicates missing)"""

    cdef int N = u_idx.shape[0]
    cdef int F = u_emb.shape[1]
    cdef int row, u, i

    scores = np.full(N, np.nan, dtype=np.float32)
    if N == 0 or u_emb.shape[0] == 0 or item_bias.shape[0] == 0:
        return scores

    cdef float[::1] scores_mv = scores
    cdef const int *u_ptr = &u_idx[0]
    cdef const int *i_ptr = &i_idx[0]
    cdef const float *u_emb_ptr = &u_emb[0, 0]
    cdef const float *item_bias_ptr = &item_bias[0]
    cdef const float *v_i_eff_ptr = &v_i_eff[0, 0]
    cdef float *scores_ptr = &scores_mv[0]

    with nogil:
        for row in range(N):
            u = u_ptr[row]
            i = i_ptr[row]
            if u >= 0 and i >= 0:
                scores_ptr[row] = item_bias_ptr[i] + dot_product(F, &u_emb_ptr[u * F], &v_i_eff_ptr[i * F])

    return scores


def _select_topk_chunk(
    float[:, ::1] scores_chunk,
    float[::1] item_bias,
    int[::1] chunk_u_idx,
    int[::1] ui_indptr,
    int[::1] ui_indices,
    int n_items,
    bint filter_previous,
    int[:, ::1] out_items
):
    """select sorted top-K item indices per row from a BLAS score chunk using a nogil size-K min-heap"""

    cdef int B = scores_chunk.shape[0]
    cdef int I = scores_chunk.shape[1]
    cdef int K = n_items
    if B == 0 or K <= 0:
        return

    cdef const float *scores_ptr = &scores_chunk[0, 0]
    cdef const float *bias_ptr = &item_bias[0]
    cdef const int *u_ptr = &chunk_u_idx[0]
    cdef const int *indptr_ptr = &ui_indptr[0] if filter_previous else NULL
    cdef const int *indices_ptr = &ui_indices[0] if (filter_previous and ui_indices.shape[0] > 0) else NULL
    cdef int *out_ptr = &out_items[0, 0]

    cdef float *heap_s = <float*>malloc(K * sizeof(float))
    cdef int *heap_i = <int*>malloc(K * sizeof(int))
    cdef int *seen_tag = <int*>malloc(I * sizeof(int)) if filter_previous else NULL

    cdef int b, u, i, idx, h_size, k_idx
    cdef float s
    cdef const float *row_scores
    cdef int *row_out

    try:
        with nogil:
            if filter_previous:
                for i in range(I):
                    seen_tag[i] = -1

            for b in range(B):
                u = u_ptr[b]
                row_scores = &scores_ptr[<Py_ssize_t>b * I]
                row_out = &out_ptr[<Py_ssize_t>b * K]

                if filter_previous and indices_ptr != NULL:
                    for idx in range(indptr_ptr[u], indptr_ptr[u + 1]):
                        seen_tag[indices_ptr[idx]] = b

                h_size = 0
                for i in range(I):
                    if filter_previous and seen_tag[i] == b:
                        continue
                    s = row_scores[i] + bias_ptr[i]
                    if h_size < K:
                        heap_s[h_size] = s
                        heap_i[h_size] = i
                        heap_sift_up(heap_s, heap_i, h_size)
                        h_size += 1
                    elif s > heap_s[0]:
                        heap_s[0] = s
                        heap_i[0] = i
                        heap_sift_down(heap_s, heap_i, K, 0)

                for k_idx in range(h_size, K):
                    row_out[k_idx] = -1

                for k_idx in range(h_size - 1, -1, -1):
                    row_out[k_idx] = heap_i[0]
                    h_size -= 1
                    if h_size > 0:
                        heap_s[0] = heap_s[h_size]
                        heap_i[0] = heap_i[h_size]
                        heap_sift_down(heap_s, heap_i, h_size, 0)
    finally:
        free(heap_s)
        free(heap_i)
        if seen_tag != NULL:
            free(seen_tag)
