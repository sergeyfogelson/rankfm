#!python
#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

# -----------------------
# [C/Python] dependencies
# -----------------------

from libc.math cimport exp, log, pow
from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libc.stdlib cimport calloc, free, malloc
from libc.string cimport memcpy, memset

import warnings

import numpy as np

cdef extern from "fast_ops.h":
    ctypedef struct rfm_rng_t:
        uint64_t s[4]

    ctypedef void (*rfm_worker_fn)(int tid, int n_threads, void *ctx) noexcept nogil
    ctypedef struct rfm_pool_t:
        pass

    void *rfm_aligned_malloc(size_t nbytes) noexcept nogil
    void rfm_aligned_free(void *ptr) noexcept nogil
    int64_t rfm_atomic_fetch_add_i64(int64_t *ptr, int64_t val) noexcept nogil
    int rfm_atomic_cas_i64(int64_t *ptr, int64_t expected, int64_t desired) noexcept nogil

    void rfm_rng_seed(rfm_rng_t *rng, uint64_t seed, uint64_t stream) noexcept nogil
    uint32_t rfm_rng_bounded(rfm_rng_t *rng, uint32_t bound) noexcept nogil
    float rfm_rng_float01(rfm_rng_t *rng) noexcept nogil

    ctypedef struct rfm_perm_t:
        uint64_t n
    void rfm_perm_init(rfm_perm_t *perm, uint64_t n, uint64_t seed) noexcept nogil
    uint64_t rfm_perm_at(const rfm_perm_t *perm, uint64_t x) noexcept nogil

    float rfm_dot(int F, const float *a, const float *b) noexcept nogil
    void rfm_vec_zero(int F, float *dst) noexcept nogil
    void rfm_vec_copy(int F, const float *src, float *dst) noexcept nogil
    void rfm_vec_add(int F, const float *a, const float *b, float *out) noexcept nogil
    void rfm_vec_sub(int F, const float *a, const float *b, float *out) noexcept nogil
    void rfm_vec_axpy(int F, float alpha, const float *x, float *y) noexcept nogil
    void rfm_vec_reg_axpy(int F, float keep, float alpha, const float *x, float *y) noexcept nogil
    void rfm_vec_scale(int F, float keep, float *y) noexcept nogil
    void rfm_dual_item_update(
        int F, float keep, float step, const float *u_emb,
        float *vi, float *vj, float *veff_i, float *veff_j, int update_veff
    ) noexcept nogil

    rfm_pool_t *rfm_pool_create(int n_threads) noexcept nogil
    void rfm_pool_run(rfm_pool_t *pool, rfm_worker_fn fn, void *ctx) noexcept nogil
    void rfm_pool_destroy(rfm_pool_t *pool) noexcept nogil


# --------------------
# [C] helper functions
# --------------------

cdef inline int lsearch(int32_t item, const int32_t *items, int64_t n) noexcept nogil:
    """linear search for a given item in a sorted array of items"""

    cdef int64_t i
    for i in range(n):
        if item == items[i]:
            return 1
    return 0


cdef inline int bsearch(int32_t item, const int32_t *items, int64_t n) noexcept nogil:
    """binary search for a given item in a sorted array of items"""

    cdef int64_t lo = 0
    cdef int64_t hi = n - 1
    cdef int64_t md

    while lo <= hi:
        md = lo + ((hi - lo) >> 1)
        if items[md] == item:
            return 1
        elif items[md] < item:
            lo = md + 1
        else:
            hi = md - 1
    return 0


cdef inline int item_observed(int32_t item, const int32_t *items, int64_t n) noexcept nogil:
    """check if item is in sorted array using linear search for small n and binary search otherwise"""

    if n <= 16:
        return lsearch(item, items, n)
    return bsearch(item, items, n)


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


cdef inline void compute_user_embedding_csr(
    int F,
    int32_t u,
    const float *v_u_ptr,
    const float *v_uf_ptr,
    const int64_t *uf_indptr,
    const int32_t *uf_indices,
    const float *uf_data,
    int use_v_u,
    int u_has_uf,
    float *out_u_emb
) noexcept nogil:
    """compute effective user embedding using CSR sparse user features and 64-bit safe offsets"""

    cdef int64_t idx
    cdef int32_t p
    cdef float val
    cdef const float *v_u_row = &v_u_ptr[<int64_t>u * <int64_t>F]

    if use_v_u:
        rfm_vec_copy(F, v_u_row, out_u_emb)
    else:
        rfm_vec_zero(F, out_u_emb)

    if u_has_uf:
        for idx in range(uf_indptr[u], uf_indptr[u + 1]):
            p = uf_indices[idx]
            val = uf_data[idx]
            rfm_vec_axpy(F, val, &v_uf_ptr[<int64_t>p * <int64_t>F], out_u_emb)


cdef inline float compute_item_embedding_csr(
    int F,
    int32_t m,
    const float *w_i_ptr,
    const float *w_if_ptr,
    const float *v_i_ptr,
    const float *v_if_ptr,
    const int64_t *if_indptr,
    const int32_t *if_indices,
    const float *if_data,
    float *out_emb
) noexcept nogil:
    """write item [m]'s effective embedding (v_i + x_if @ v_if) to [out_emb] from the current weights and return its effective bias"""

    cdef int64_t idx
    cdef int32_t q
    cdef float val
    cdef float bias = w_i_ptr[m]

    rfm_vec_copy(F, &v_i_ptr[<int64_t>m * <int64_t>F], out_emb)
    for idx in range(if_indptr[m], if_indptr[m + 1]):
        q = if_indices[idx]
        val = if_data[idx]
        bias += val * w_if_ptr[q]
        rfm_vec_axpy(F, val, &v_if_ptr[<int64_t>q * <int64_t>F], out_emb)
    return bias


cdef inline float sparse_sq_distance(
    const int64_t *indptr,
    const int32_t *indices,
    const float *data,
    int32_t a,
    int32_t b
) noexcept nogil:
    """squared euclidean distance between CSR rows [a] and [b] via a two-pointer merge over their sorted columns"""

    cdef int64_t ia = indptr[a], ea = indptr[a + 1], ib = indptr[b], eb = indptr[b + 1]
    cdef float d, total = 0.0

    while ia < ea and ib < eb:
        if indices[ia] < indices[ib]:
            total += data[ia] * data[ia]
            ia += 1
        elif indices[ib] < indices[ia]:
            total += data[ib] * data[ib]
            ib += 1
        else:
            d = data[ia] - data[ib]
            total += d * d
            ia += 1
            ib += 1
    while ia < ea:
        total += data[ia] * data[ia]
        ia += 1
    while ib < eb:
        total += data[ib] * data[ib]
        ib += 1
    return total


cdef inline void heap_sift_down(float *hs, int32_t *hi, int size, int pos) noexcept nogil:
    """sift-down on a min-heap of (score, item_idx) pairs"""

    cdef int child, right
    cdef float s_val = hs[pos]
    cdef int32_t i_val = hi[pos]

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


cdef inline void heap_sift_up(float *hs, int32_t *hi, int pos) noexcept nogil:
    """sift-up on a min-heap of (score, item_idx) pairs"""

    cdef int parent
    cdef float s_val = hs[pos]
    cdef int32_t i_val = hi[pos]

    while pos > 0:
        parent = (pos - 1) >> 1
        if hs[parent] <= s_val:
            break
        hs[pos] = hs[parent]
        hi[pos] = hi[parent]
        pos = parent

    hs[pos] = s_val
    hi[pos] = i_val


# -------------------------------------------
# [C] Parallel Worker Contexts & Entry Points
# -------------------------------------------

cdef struct DenseCsrCtx:
    const float *mat_ptr
    int64_t *indptr_ptr
    int32_t *indices_ptr
    float *data_ptr
    int64_t n_rows
    int64_t n_cols


cdef void _dense_count_worker(int tid, int n_threads, void *raw_ctx) noexcept nogil:
    cdef DenseCsrCtx *ctx = <DenseCsrCtx *>raw_ctx
    cdef int64_t r_start = (ctx.n_rows * tid) // n_threads
    cdef int64_t r_end = (ctx.n_rows * (tid + 1)) // n_threads
    cdef int64_t r, c, cnt
    cdef const float *row_ptr

    for r in range(r_start, r_end):
        row_ptr = &ctx.mat_ptr[r * ctx.n_cols]
        cnt = 0
        for c in range(ctx.n_cols):
            if row_ptr[c] != 0.0:
                cnt += 1
        ctx.indptr_ptr[r + 1] = cnt


cdef void _dense_fill_worker(int tid, int n_threads, void *raw_ctx) noexcept nogil:
    cdef DenseCsrCtx *ctx = <DenseCsrCtx *>raw_ctx
    cdef int64_t r_start = (ctx.n_rows * tid) // n_threads
    cdef int64_t r_end = (ctx.n_rows * (tid + 1)) // n_threads
    cdef int64_t r, c, pos
    cdef float val
    cdef const float *row_ptr

    for r in range(r_start, r_end):
        row_ptr = &ctx.mat_ptr[r * ctx.n_cols]
        pos = ctx.indptr_ptr[r]
        for c in range(ctx.n_cols):
            val = row_ptr[c]
            if val != 0.0:
                ctx.indices_ptr[pos] = <int32_t>c
                ctx.data_ptr[pos] = val
                pos += 1


cdef struct CsrSpMMCtx:
    const int64_t *indptr_ptr
    const int32_t *indices_ptr
    const float *data_ptr
    const float *rhs_ptr
    const float *base_ptr
    float *out_ptr
    int64_t n_rows
    int F


cdef void _csr_spmm_worker(int tid, int n_threads, void *raw_ctx) noexcept nogil:
    cdef CsrSpMMCtx *ctx = <CsrSpMMCtx *>raw_ctx
    cdef int64_t r_start = (ctx.n_rows * tid) // n_threads
    cdef int64_t r_end = (ctx.n_rows * (tid + 1)) // n_threads
    cdef int F = ctx.F
    cdef int64_t r, idx
    cdef int32_t col
    cdef float val
    cdef float *out_row

    for r in range(r_start, r_end):
        out_row = &ctx.out_ptr[r * <int64_t>F]
        if ctx.base_ptr != NULL:
            rfm_vec_copy(F, &ctx.base_ptr[r * <int64_t>F], out_row)
        else:
            rfm_vec_zero(F, out_row)
        for idx in range(ctx.indptr_ptr[r], ctx.indptr_ptr[r + 1]):
            col = ctx.indices_ptr[idx]
            val = ctx.data_ptr[idx]
            rfm_vec_axpy(F, val, &ctx.rhs_ptr[<int64_t>col * <int64_t>F], out_row)


cdef struct CsrSpMVCtx:
    const int64_t *indptr_ptr
    const int32_t *indices_ptr
    const float *data_ptr
    const float *vec_ptr
    const float *base_ptr
    float *out_ptr
    int64_t n_rows


cdef void _csr_spmv_worker(int tid, int n_threads, void *raw_ctx) noexcept nogil:
    cdef CsrSpMVCtx *ctx = <CsrSpMVCtx *>raw_ctx
    cdef int64_t r_start = (ctx.n_rows * tid) // n_threads
    cdef int64_t r_end = (ctx.n_rows * (tid + 1)) // n_threads
    cdef int64_t r, idx
    cdef float acc

    for r in range(r_start, r_end):
        acc = ctx.base_ptr[r] if ctx.base_ptr != NULL else 0.0
        for idx in range(ctx.indptr_ptr[r], ctx.indptr_ptr[r + 1]):
            acc += ctx.data_ptr[idx] * ctx.vec_ptr[ctx.indices_ptr[idx]]
        ctx.out_ptr[r] = acc


cdef struct RefreshEffItemsCtx:
    const float *w_i_ptr
    const float *w_if_ptr
    const float *v_i_ptr
    const float *v_if_ptr
    const int64_t *if_indptr_ptr
    const int32_t *if_indices_ptr
    const float *if_data_ptr
    float *b_eff_ptr
    float *v_eff_ptr
    int32_t I
    int F


cdef void _refresh_eff_items_worker(int tid, int n_threads, void *raw_ctx) noexcept nogil:
    cdef RefreshEffItemsCtx *ctx = <RefreshEffItemsCtx *>raw_ctx
    cdef int32_t m_start = <int32_t>((<int64_t>ctx.I * tid) // n_threads)
    cdef int32_t m_end = <int32_t>((<int64_t>ctx.I * (tid + 1)) // n_threads)
    cdef int F = ctx.F
    cdef int32_t m, q
    cdef int64_t idx
    cdef float val, bias
    cdef float *v_eff_row

    for m in range(m_start, m_end):
        bias = ctx.w_i_ptr[m]
        v_eff_row = &ctx.v_eff_ptr[<int64_t>m * <int64_t>F]
        rfm_vec_copy(F, &ctx.v_i_ptr[<int64_t>m * <int64_t>F], v_eff_row)
        for idx in range(ctx.if_indptr_ptr[m], ctx.if_indptr_ptr[m + 1]):
            q = ctx.if_indices_ptr[idx]
            val = ctx.if_data_ptr[idx]
            bias += val * ctx.w_if_ptr[q]
            rfm_vec_axpy(F, val, &ctx.v_if_ptr[<int64_t>q * <int64_t>F], v_eff_row)
        ctx.b_eff_ptr[m] = bias


# Batched updates for densely used feature rows ("hot" rows).
#
# A feature column that is non-zero on a large share of rows has its weight row written by every thread on
# nearly every step, so the cores spend their time taking those cache lines from one another and training
# stops scaling. In batched mode a thread does not write a hot row per step. It adds the step's update to a
# private accumulator and, every [batch_steps] of its own steps, applies the batch to the shared row once.
# Between applications the shared hot rows are only read, so they stay in every core's cache. All other
# weights keep their per-step lock-free updates.
#
# A batch is a sum of updates that were computed without seeing each other, across all threads at once. Such
# a sum overshoots once learning_rate * threads * batch_steps passes a limit (measured at roughly 32), so the
# batch is multiplied by [scale] = min(1, HOT_MAX_RATE / (learning_rate * threads * batch_steps)). The L2
# decay accumulated with the batch is scaled by the same factor, keeping the regularization strength intact.
#
# Per-thread accumulator layout: [uf vectors | if vectors | if scalars | uf decay | if decay]
cdef struct HotRows:
    const int32_t *uf_slot      # feature column -> hot slot, or -1
    const int32_t *uf_rows      # hot slot -> feature column
    const int32_t *if_slot
    const int32_t *if_rows
    float *acc_base             # per-thread accumulators
    size_t acc_stride
    int32_t n_uf
    int32_t n_if
    int batch_steps
    float scale


cdef inline void hot_rows_apply(
    HotRows *hot, int F, float *v_uf, float *v_if, float *w_if,
    float *acc_uf, float *acc_if, float *acc_w_if, float *dec_uf, float *dec_if
) noexcept nogil:
    """apply and clear this thread's accumulated hot-row updates"""

    cdef int32_t h
    cdef float scale = hot.scale
    cdef float keep

    for h in range(hot.n_uf):
        if dec_uf[h] > 0.0:
            keep = 1.0 - scale * dec_uf[h]
            if keep < 0.0:
                keep = 0.0
            rfm_vec_reg_axpy(F, keep, scale, &acc_uf[<int64_t>h * F], &v_uf[<int64_t>hot.uf_rows[h] * <int64_t>F])
            rfm_vec_zero(F, &acc_uf[<int64_t>h * F])
            dec_uf[h] = 0.0
    for h in range(hot.n_if):
        if dec_if[h] > 0.0:
            keep = 1.0 - scale * dec_if[h]
            if keep < 0.0:
                keep = 0.0
            rfm_vec_reg_axpy(F, keep, scale, &acc_if[<int64_t>h * F], &v_if[<int64_t>hot.if_rows[h] * <int64_t>F])
            rfm_vec_zero(F, &acc_if[<int64_t>h * F])
            w_if[hot.if_rows[h]] = w_if[hot.if_rows[h]] * keep + scale * acc_w_if[h]
            acc_w_if[h] = 0.0
            dec_if[h] = 0.0


cdef struct FitBlockCtx:
    HotRows hot
    const int32_t *u_ptr
    const int32_t *i_ptr
    const float *sw_ptr
    rfm_perm_t perm
    const int64_t *ui_indptr_ptr
    const int32_t *ui_indices_ptr
    const int64_t *uf_indptr_ptr
    const int32_t *uf_indices_ptr
    const float *uf_data_ptr
    const int64_t *if_indptr_ptr
    const int32_t *if_indices_ptr
    const float *if_data_ptr
    float *w_i_ptr
    float *w_if_ptr
    float *v_u_ptr
    float *v_i_ptr
    float *v_uf_ptr
    float *v_if_ptr
    float *b_eff_ptr
    float *v_eff_ptr
    const float *mult_lut_ptr
    const float *uf_step_scale_ptr
    float *scratch_base
    size_t scratch_stride
    double *ll_slots
    int64_t *last_step_uf
    int64_t *last_step_if
    int64_t block_start
    int64_t block_end
    int64_t cursor
    int64_t grain
    int32_t I
    int F
    int max_samples
    int x_uf_any
    int x_if_any
    int verbose
    int uniform_reg
    int normalize_steps
    float eta
    float reg_a_keep
    float reg_b_keep
    float log_reg_uf_lazy
    float log_reg_if_lazy
    float user_dropout
    uint64_t base_seed
    uint64_t stream_epoch_base


cdef void _fit_block_worker(int tid, int n_threads, void *raw_ctx) noexcept nogil:
    cdef FitBlockCtx *ctx = <FitBlockCtx *>raw_ctx
    cdef int64_t block_end = ctx.block_end
    if ctx.block_start >= block_end:
        return

    cdef int F = ctx.F
    cdef int32_t I = ctx.I
    cdef uint32_t uI = <uint32_t>I
    cdef int max_samples = ctx.max_samples
    cdef int x_uf_any = ctx.x_uf_any
    cdef int x_if_any = ctx.x_if_any
    cdef int uniform_reg = ctx.uniform_reg
    cdef float eta = ctx.eta
    cdef float reg_a_keep = ctx.reg_a_keep
    cdef float reg_b_keep = ctx.reg_b_keep
    cdef float log_reg_uf_lazy = ctx.log_reg_uf_lazy
    cdef float log_reg_if_lazy = ctx.log_reg_if_lazy
    cdef float user_dropout = ctx.user_dropout
    cdef int64_t grain = ctx.grain

    cdef float *my_scratch = &ctx.scratch_base[<size_t>tid * ctx.scratch_stride]
    cdef float *u_emb = my_scratch
    cdef float *i_emb = &my_scratch[F]
    cdef float *j_emb = &my_scratch[2 * F]
    cdef float *diff_ij = &my_scratch[3 * F]

    cdef HotRows *hot = &ctx.hot
    cdef int use_hot = 1 if (hot.n_uf > 0 or hot.n_if > 0) else 0
    cdef const int32_t *hot_uf_slot = hot.uf_slot if hot.n_uf > 0 else NULL
    cdef const int32_t *hot_if_slot = hot.if_slot if hot.n_if > 0 else NULL
    cdef int since_apply = 0
    cdef int32_t h
    cdef float *acc_uf = NULL
    cdef float *acc_if = NULL
    cdef float *acc_w_if = NULL
    cdef float *dec_uf = NULL
    cdef float *dec_if = NULL
    if use_hot:
        acc_uf = &hot.acc_base[<size_t>tid * hot.acc_stride]
        acc_if = &acc_uf[<int64_t>hot.n_uf * F]
        acc_w_if = &acc_if[<int64_t>hot.n_if * F]
        dec_uf = &acc_w_if[hot.n_if]
        dec_if = &dec_uf[hot.n_uf]
        memset(acc_uf, 0, hot.acc_stride * sizeof(float))

    cdef rfm_rng_t rng
    cdef int64_t c_start, c_end, r, row, n_seen_u, idx, idx_i, end_i, idx_j, end_j, last_step
    cdef int32_t u, i, j, p, q, min_index
    cdef int sampled, u_has_uf, use_v_u
    cdef float sw, ut_ui, ut_uj, pairwise_utility, min_pairwise_utility
    cdef float d_outer, scale_step, step_u, step_i, old_wi, new_wi, old_wj, new_wj, diff_x_if, step_if, lazy_keep
    cdef float norm_u, norm_i, keep_u_a, keep_u_b, keep_i_a, keep_i_b, bias_i, bias_j
    cdef int normalize_steps = ctx.normalize_steps
    cdef const int32_t *u_seen_ptr
    cdef double local_ll = 0.0

    while True:
        if n_threads <= 1:
            c_start = ctx.block_start
            c_end = block_end
        else:
            c_start = rfm_atomic_fetch_add_i64(&ctx.cursor, grain)
            if c_start >= block_end:
                break
            c_end = c_start + grain
            if c_end > block_end:
                c_end = block_end

        rfm_rng_seed(&rng, ctx.base_seed, ctx.stream_epoch_base + <uint64_t>c_start)

        for r in range(c_start, c_end):
            row = <int64_t>rfm_perm_at(&ctx.perm, <uint64_t>r)
            u = ctx.u_ptr[row]
            i = ctx.i_ptr[row]
            sw = ctx.sw_ptr[row] if ctx.sw_ptr != NULL else 1.0

            n_seen_u = ctx.ui_indptr_ptr[u + 1] - ctx.ui_indptr_ptr[u]
            if n_seen_u >= I:
                continue
            u_seen_ptr = &ctx.ui_indices_ptr[ctx.ui_indptr_ptr[u]] if ctx.ui_indices_ptr != NULL else NULL

            u_has_uf = 1 if (x_uf_any and ctx.uf_indptr_ptr[u + 1] > ctx.uf_indptr_ptr[u]) else 0
            if u_has_uf and user_dropout > 0.0 and rfm_rng_float01(&rng) < user_dropout:
                use_v_u = 0
            else:
                use_v_u = 1

            # Lazy catch-up on user features before reading u_emb when uniform_reg is enabled.
            # Each feature remembers the step it was last decayed at. A thread decays it only after winning a
            # compare-and-swap that moves that marker forward, so with several threads every interval of steps is
            # applied exactly once, and a thread working on an earlier chunk than the marker simply skips.
            if u_has_uf and uniform_reg:
                for idx in range(ctx.uf_indptr_ptr[u], ctx.uf_indptr_ptr[u + 1]):
                    p = ctx.uf_indices_ptr[idx]
                    last_step = ctx.last_step_uf[p]
                    if r > last_step and rfm_atomic_cas_i64(&ctx.last_step_uf[p], last_step, r):
                        lazy_keep = <float>exp(<double>log_reg_uf_lazy * <double>(r - last_step))
                        rfm_vec_scale(F, lazy_keep, &ctx.v_uf_ptr[<int64_t>p * <int64_t>F])

            compute_user_embedding_csr(
                F, u, ctx.v_u_ptr, ctx.v_uf_ptr,
                ctx.uf_indptr_ptr, ctx.uf_indices_ptr, ctx.uf_data_ptr,
                use_v_u, u_has_uf, u_emb
            )

            # The materialized (b_eff, V_eff) cache makes scoring a candidate O(F), but with item features it is only
            # rebuilt every [sync_interval] steps, so it lags behind the feature weights. That lag is harmless when
            # searching for a negative, yet a gradient computed from lagging scores keeps pushing a pair that is
            # already ranked correctly, and training runs away. So the positive item here, and the chosen negative
            # below, are recomputed from the current weights (and written back, refreshing those two cache rows).
            if x_if_any:
                bias_i = compute_item_embedding_csr(
                    F, i, ctx.w_i_ptr, ctx.w_if_ptr, ctx.v_i_ptr, ctx.v_if_ptr,
                    ctx.if_indptr_ptr, ctx.if_indices_ptr, ctx.if_data_ptr, i_emb
                )
                ctx.b_eff_ptr[i] = bias_i
                rfm_vec_copy(F, i_emb, &ctx.v_eff_ptr[<int64_t>i * <int64_t>F])
            else:
                bias_i = ctx.b_eff_ptr[i]
                rfm_vec_copy(F, &ctx.v_eff_ptr[<int64_t>i * <int64_t>F], i_emb)
            ut_ui = bias_i + rfm_dot(F, u_emb, i_emb)

            min_index = -1
            min_pairwise_utility = 1e6

            for sampled in range(1, max_samples + 1):
                while True:
                    j = <int32_t>rfm_rng_bounded(&rng, uI)
                    if u_seen_ptr == NULL or not item_observed(j, u_seen_ptr, n_seen_u):
                        break

                ut_uj = ctx.b_eff_ptr[j] + rfm_dot(F, u_emb, &ctx.v_eff_ptr[<int64_t>j * <int64_t>F])
                pairwise_utility = ut_ui - ut_uj

                if min_index == -1 or pairwise_utility < min_pairwise_utility:
                    min_index = j
                    min_pairwise_utility = pairwise_utility

                if pairwise_utility < 1.0:
                    break

            j = min_index
            pairwise_utility = min_pairwise_utility
            if x_if_any:
                bias_j = compute_item_embedding_csr(
                    F, j, ctx.w_i_ptr, ctx.w_if_ptr, ctx.v_i_ptr, ctx.v_if_ptr,
                    ctx.if_indptr_ptr, ctx.if_indices_ptr, ctx.if_data_ptr, j_emb
                )
                ctx.b_eff_ptr[j] = bias_j
                rfm_vec_copy(F, j_emb, &ctx.v_eff_ptr[<int64_t>j * <int64_t>F])
                pairwise_utility = ut_ui - (bias_j + rfm_dot(F, u_emb, j_emb))
            else:
                rfm_vec_copy(F, &ctx.v_eff_ptr[<int64_t>j * <int64_t>F], j_emb)

            if ctx.verbose:
                local_ll += <double>safe_log_sigmoid(pairwise_utility)

            d_outer = safe_sigmoid_neg(pairwise_utility)
            scale_step = eta * sw * ctx.mult_lut_ptr[sampled] * d_outer

            # One step moves the pairwise utility by about step * (|u|^2 * (2 + |x_i - x_j|^2) + |v_i - v_j|^2 * (1 + |x_u|^2)):
            # every active feature adds to the effective learning rate. Dividing each side's step by its feature mass
            # keeps the movement at the feature-free level, step * (2 * |u|^2 + |v_i - v_j|^2), whatever the row density.
            #
            # The scale multiplies the whole update, gradient and L2 decay alike: it is a per-sample learning rate.
            # Scaling the gradient alone would leave the decay at full strength and so multiply the effective
            # regularization on exactly the rows with the most features.
            norm_u = 1.0
            norm_i = 1.0
            if normalize_steps:
                if u_has_uf:
                    norm_u = ctx.uf_step_scale_ptr[u]
                if x_if_any:
                    norm_i = 2.0 / (2.0 + sparse_sq_distance(ctx.if_indptr_ptr, ctx.if_indices_ptr, ctx.if_data_ptr, i, j))
            step_u = scale_step * norm_u
            step_i = scale_step * norm_i
            keep_u_a = 1.0 - norm_u * (1.0 - reg_a_keep)
            keep_u_b = 1.0 - norm_u * (1.0 - reg_b_keep)
            keep_i_a = 1.0 - norm_i * (1.0 - reg_a_keep)
            keep_i_b = 1.0 - norm_i * (1.0 - reg_b_keep)

            # Update item scalar biases (+ immediate b_eff delta when x_if_any)
            old_wi = ctx.w_i_ptr[i]
            new_wi = old_wi * keep_i_a + step_i
            ctx.w_i_ptr[i] = new_wi

            old_wj = ctx.w_i_ptr[j]
            new_wj = old_wj * keep_i_a - step_i
            ctx.w_i_ptr[j] = new_wj

            if x_if_any:
                ctx.b_eff_ptr[i] += (new_wi - old_wi)
                ctx.b_eff_ptr[j] += (new_wj - old_wj)

            # Contiguous SIMD factor updates
            rfm_vec_sub(F, i_emb, j_emb, diff_ij)

            if use_v_u:
                rfm_vec_reg_axpy(F, keep_u_a, step_u, diff_ij, &ctx.v_u_ptr[<int64_t>u * <int64_t>F])

            rfm_dual_item_update(
                F,
                keep_i_a,
                step_i,
                u_emb,
                &ctx.v_i_ptr[<int64_t>i * <int64_t>F],
                &ctx.v_i_ptr[<int64_t>j * <int64_t>F],
                &ctx.v_eff_ptr[<int64_t>i * <int64_t>F],
                &ctx.v_eff_ptr[<int64_t>j * <int64_t>F],
                x_if_any
            )

            # Sparse CSR user-feature factor updates
            if u_has_uf:
                if uniform_reg:
                    for idx in range(ctx.uf_indptr_ptr[u], ctx.uf_indptr_ptr[u + 1]):
                        p = ctx.uf_indices_ptr[idx]
                        rfm_vec_axpy(F, step_u * ctx.uf_data_ptr[idx], diff_ij, &ctx.v_uf_ptr[<int64_t>p * <int64_t>F])
                else:
                    for idx in range(ctx.uf_indptr_ptr[u], ctx.uf_indptr_ptr[u + 1]):
                        p = ctx.uf_indices_ptr[idx]
                        if hot_uf_slot != NULL and hot_uf_slot[p] >= 0:
                            h = hot_uf_slot[p]
                            rfm_vec_axpy(F, step_u * ctx.uf_data_ptr[idx], diff_ij, &acc_uf[<int64_t>h * F])
                            dec_uf[h] += 1.0 - keep_u_b
                        else:
                            rfm_vec_reg_axpy(
                                F, keep_u_b, step_u * ctx.uf_data_ptr[idx],
                                diff_ij, &ctx.v_uf_ptr[<int64_t>p * <int64_t>F]
                            )

            # Sparse 2-pointer merge over non-zero item features of (i, j)
            if x_if_any:
                idx_i = ctx.if_indptr_ptr[i]
                end_i = ctx.if_indptr_ptr[i + 1]
                idx_j = ctx.if_indptr_ptr[j]
                end_j = ctx.if_indptr_ptr[j + 1]

                while idx_i < end_i or idx_j < end_j:
                    if idx_j >= end_j or (idx_i < end_i and ctx.if_indices_ptr[idx_i] < ctx.if_indices_ptr[idx_j]):
                        q = ctx.if_indices_ptr[idx_i]
                        diff_x_if = ctx.if_data_ptr[idx_i]
                        idx_i += 1
                    elif idx_i >= end_i or ctx.if_indices_ptr[idx_j] < ctx.if_indices_ptr[idx_i]:
                        q = ctx.if_indices_ptr[idx_j]
                        diff_x_if = -ctx.if_data_ptr[idx_j]
                        idx_j += 1
                    else:
                        q = ctx.if_indices_ptr[idx_i]
                        diff_x_if = ctx.if_data_ptr[idx_i] - ctx.if_data_ptr[idx_j]
                        idx_i += 1
                        idx_j += 1

                    if uniform_reg:
                        last_step = ctx.last_step_if[q]
                        if r + 1 > last_step and rfm_atomic_cas_i64(&ctx.last_step_if[q], last_step, r + 1):
                            lazy_keep = <float>exp(<double>log_reg_if_lazy * <double>(r + 1 - last_step))
                            ctx.w_if_ptr[q] *= lazy_keep
                            rfm_vec_scale(F, lazy_keep, &ctx.v_if_ptr[<int64_t>q * <int64_t>F])
                        if diff_x_if != 0.0:
                            step_if = step_i * diff_x_if
                            ctx.w_if_ptr[q] += step_if
                            rfm_vec_axpy(F, step_if, u_emb, &ctx.v_if_ptr[<int64_t>q * <int64_t>F])
                    elif hot_if_slot != NULL and hot_if_slot[q] >= 0:
                        h = hot_if_slot[q]
                        dec_if[h] += 1.0 - keep_i_b
                        if diff_x_if != 0.0:
                            step_if = step_i * diff_x_if
                            acc_w_if[h] += step_if
                            rfm_vec_axpy(F, step_if, u_emb, &acc_if[<int64_t>h * F])
                    else:
                        if diff_x_if != 0.0:
                            step_if = step_i * diff_x_if
                            ctx.w_if_ptr[q] = ctx.w_if_ptr[q] * keep_i_b + step_if
                            rfm_vec_reg_axpy(F, keep_i_b, step_if, u_emb, &ctx.v_if_ptr[<int64_t>q * <int64_t>F])
                        else:
                            ctx.w_if_ptr[q] *= keep_i_b
                            rfm_vec_scale(F, keep_i_b, &ctx.v_if_ptr[<int64_t>q * <int64_t>F])

            if use_hot:
                since_apply += 1
                if since_apply >= hot.batch_steps:
                    hot_rows_apply(hot, F, ctx.v_uf_ptr, ctx.v_if_ptr, ctx.w_if_ptr, acc_uf, acc_if, acc_w_if, dec_uf, dec_if)
                    since_apply = 0

        # never carry a partial batch out of the chunk: the block-level cache refresh must see every update
        if use_hot and since_apply > 0:
            hot_rows_apply(hot, F, ctx.v_uf_ptr, ctx.v_if_ptr, ctx.w_if_ptr, acc_uf, acc_if, acc_w_if, dec_uf, dec_if)
            since_apply = 0

        if n_threads <= 1:
            break

    if ctx.verbose:
        ctx.ll_slots[tid * 8] += local_ll


cdef struct FoldInCtx:
    const int32_t *u_ptr
    const int32_t *i_ptr
    const float *sw_ptr
    rfm_perm_t perm
    const int64_t *ui_indptr_ptr
    const int32_t *ui_indices_ptr
    const float *u_feat_ptr
    const float *item_bias_ptr
    const float *v_i_eff_ptr
    float *v_u_ptr
    const float *mult_lut_ptr
    float *scratch_base
    size_t scratch_stride
    int64_t N
    int64_t cursor
    int64_t grain
    int32_t I
    int F
    int max_samples
    float eta
    float reg_a_keep
    uint64_t base_seed
    uint64_t stream_epoch_base


cdef void _fold_in_worker(int tid, int n_threads, void *raw_ctx) noexcept nogil:
    cdef FoldInCtx *ctx = <FoldInCtx *>raw_ctx
    cdef int64_t N = ctx.N
    if N <= 0:
        return

    cdef int F = ctx.F
    cdef int32_t I = ctx.I
    cdef uint32_t uI = <uint32_t>I
    cdef int max_samples = ctx.max_samples
    cdef float eta = ctx.eta
    cdef float reg_a_keep = ctx.reg_a_keep
    cdef int64_t grain = ctx.grain

    cdef float *u_emb = &ctx.scratch_base[<size_t>tid * ctx.scratch_stride]
    cdef float *diff_ij = &u_emb[F]

    cdef rfm_rng_t rng
    cdef int64_t c_start, c_end, r, row, n_seen_u
    cdef int32_t u, i, j, min_index
    cdef int sampled
    cdef float sw, ut_ui, ut_uj, pairwise_utility, min_pairwise_utility, d_outer, scale_step
    cdef float *v_u_row
    cdef const float *u_feat_row
    cdef const float *v_i_row
    cdef const float *v_j_row
    cdef const int32_t *u_seen_ptr

    while True:
        if n_threads <= 1:
            c_start = 0
            c_end = N
        else:
            c_start = rfm_atomic_fetch_add_i64(&ctx.cursor, grain)
            if c_start >= N:
                break
            c_end = c_start + grain
            if c_end > N:
                c_end = N

        rfm_rng_seed(&rng, ctx.base_seed, ctx.stream_epoch_base + <uint64_t>c_start)

        for r in range(c_start, c_end):
            row = <int64_t>rfm_perm_at(&ctx.perm, <uint64_t>r)
            u = ctx.u_ptr[row]
            i = ctx.i_ptr[row]
            sw = ctx.sw_ptr[row] if ctx.sw_ptr != NULL else 1.0

            n_seen_u = ctx.ui_indptr_ptr[u + 1] - ctx.ui_indptr_ptr[u]
            if n_seen_u >= I:
                continue
            u_seen_ptr = &ctx.ui_indices_ptr[ctx.ui_indptr_ptr[u]] if ctx.ui_indices_ptr != NULL else NULL

            v_u_row = &ctx.v_u_ptr[<int64_t>u * <int64_t>F]
            u_feat_row = &ctx.u_feat_ptr[<int64_t>u * <int64_t>F]
            rfm_vec_add(F, v_u_row, u_feat_row, u_emb)

            v_i_row = &ctx.v_i_eff_ptr[<int64_t>i * <int64_t>F]
            ut_ui = ctx.item_bias_ptr[i] + rfm_dot(F, u_emb, v_i_row)

            min_index = -1
            min_pairwise_utility = 1e6

            for sampled in range(1, max_samples + 1):
                while True:
                    j = <int32_t>rfm_rng_bounded(&rng, uI)
                    if u_seen_ptr == NULL or not item_observed(j, u_seen_ptr, n_seen_u):
                        break

                ut_uj = ctx.item_bias_ptr[j] + rfm_dot(F, u_emb, &ctx.v_i_eff_ptr[<int64_t>j * <int64_t>F])
                pairwise_utility = ut_ui - ut_uj

                if min_index == -1 or pairwise_utility < min_pairwise_utility:
                    min_index = j
                    min_pairwise_utility = pairwise_utility

                if pairwise_utility < 1.0:
                    break

            j = min_index
            pairwise_utility = min_pairwise_utility
            d_outer = safe_sigmoid_neg(pairwise_utility)
            scale_step = eta * sw * ctx.mult_lut_ptr[sampled] * d_outer
            v_j_row = &ctx.v_i_eff_ptr[<int64_t>j * <int64_t>F]

            rfm_vec_sub(F, v_i_row, v_j_row, diff_ij)
            rfm_vec_reg_axpy(F, reg_a_keep, scale_step, diff_ij, v_u_row)

        if n_threads <= 1:
            break


cdef struct PredictCtx:
    const int32_t *u_ptr
    const int32_t *i_ptr
    const float *u_emb_ptr
    const float *item_bias_ptr
    const float *v_i_eff_ptr
    float *scores_ptr
    int64_t N
    int F


cdef void _predict_worker(int tid, int n_threads, void *raw_ctx) noexcept nogil:
    cdef PredictCtx *ctx = <PredictCtx *>raw_ctx
    cdef int64_t r_start = (ctx.N * tid) // n_threads
    cdef int64_t r_end = (ctx.N * (tid + 1)) // n_threads
    cdef int F = ctx.F
    cdef int64_t row
    cdef int32_t u, i

    for row in range(r_start, r_end):
        u = ctx.u_ptr[row]
        i = ctx.i_ptr[row]
        if u >= 0 and i >= 0:
            ctx.scores_ptr[row] = ctx.item_bias_ptr[i] + rfm_dot(
                F,
                &ctx.u_emb_ptr[<int64_t>u * <int64_t>F],
                &ctx.v_i_eff_ptr[<int64_t>i * <int64_t>F]
            )


cdef struct TopKChunkCtx:
    const float *scores_ptr
    const float *bias_ptr
    const int32_t *u_ptr
    const int64_t *indptr_ptr
    const int32_t *indices_ptr
    int32_t *out_ptr
    float *heap_s_base
    int32_t *heap_i_base
    int64_t *seen_base
    size_t heap_stride
    size_t seen_stride
    int64_t B
    int64_t cursor
    int64_t grain
    int32_t I
    int32_t K
    int filter_previous


cdef void _select_topk_worker(int tid, int n_threads, void *raw_ctx) noexcept nogil:
    cdef TopKChunkCtx *ctx = <TopKChunkCtx *>raw_ctx
    cdef int64_t B = ctx.B
    if B <= 0:
        return

    cdef int32_t I = ctx.I
    cdef int32_t K = ctx.K
    cdef int filter_previous = ctx.filter_previous
    cdef int64_t grain = ctx.grain

    # per-thread scratch comes from the caller, which has already verified the allocation succeeded
    cdef float *heap_s = &ctx.heap_s_base[<size_t>tid * ctx.heap_stride]
    cdef int32_t *heap_i = &ctx.heap_i_base[<size_t>tid * ctx.heap_stride]
    cdef int64_t *seen_tag = &ctx.seen_base[<size_t>tid * ctx.seen_stride] if filter_previous else NULL

    cdef int64_t c_start, c_end, b, idx
    cdef int32_t u, i, h_size, k_idx
    cdef float s
    cdef const float *row_scores
    cdef int32_t *row_out

    if filter_previous:
        for i in range(I):
            seen_tag[i] = -1

    while True:
        if n_threads <= 1:
            c_start = 0
            c_end = B
        else:
            c_start = rfm_atomic_fetch_add_i64(&ctx.cursor, grain)
            if c_start >= B:
                break
            c_end = c_start + grain
            if c_end > B:
                c_end = B

        for b in range(c_start, c_end):
            u = ctx.u_ptr[b]
            row_scores = &ctx.scores_ptr[b * <int64_t>I]
            row_out = &ctx.out_ptr[b * <int64_t>K]

            if filter_previous and ctx.indices_ptr != NULL:
                for idx in range(ctx.indptr_ptr[u], ctx.indptr_ptr[u + 1]):
                    seen_tag[ctx.indices_ptr[idx]] = b

            h_size = 0
            for i in range(I):
                if filter_previous and seen_tag[i] == b:
                    continue
                s = row_scores[i] + ctx.bias_ptr[i]
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

        if n_threads <= 1:
            break


# -------------------------
# [Python] helper functions
# -------------------------

def _locate_pairs_in_csr(
    const int64_t[::1] indptr,
    const int32_t[::1] indices,
    const int32_t[::1] rows,
    const int32_t[::1] cols
):
    """for each (row, col) pair return where [col] sits, or would be inserted, in that row of a sorted CSR

    returns (positions int64[n] into [indices], present bool[n]). each lookup is a binary search inside one row,
    so the cost depends on the number of pairs and not on the size of the matrix.
    """

    cdef int64_t n = rows.shape[0]
    positions = np.empty(n, dtype=np.int64)
    present = np.zeros(n, dtype=np.bool_)
    if n == 0:
        return positions, present

    cdef int64_t[::1] positions_mv = positions
    cdef unsigned char[::1] present_mv = present.view(np.uint8)
    cdef int64_t k, lo, hi, md
    cdef int32_t col

    with nogil:
        for k in range(n):
            lo = indptr[rows[k]]
            hi = indptr[rows[k] + 1]
            col = cols[k]
            while lo < hi:
                md = lo + ((hi - lo) >> 1)
                if indices[md] < col:
                    lo = md + 1
                else:
                    hi = md
            positions_mv[k] = lo
            if lo < indptr[rows[k] + 1] and indices[lo] == col:
                present_mv[k] = 1

    return positions, present


def _permutation_sample(uint64_t n, uint64_t seed, positions):
    """evaluate the per-epoch shuffle permutation of [0, n) at the given positions (exposed for tests)"""

    cdef rfm_perm_t perm
    rfm_perm_init(&perm, n, seed)
    pos = np.ascontiguousarray(positions, dtype=np.uint64)
    out = np.empty(len(pos), dtype=np.uint64)
    cdef uint64_t[::1] pos_mv = pos
    cdef uint64_t[::1] out_mv = out
    cdef int64_t k
    for k in range(pos_mv.shape[0]):
        out_mv[k] = rfm_perm_at(&perm, pos_mv[k])
    return out


def dense_to_csr(float[:, ::1] mat, int n_threads=1):
    """convert a 2D float32 feature matrix into (int64 indptr, int32 indices, float32 data, has_any) via 2-pass C scan"""

    cdef int64_t n_rows = mat.shape[0]
    cdef int64_t n_cols = mat.shape[1]

    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    if n_rows == 0 or n_cols == 0:
        return indptr, np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.float32), 0

    cdef int64_t[::1] indptr_mv = indptr
    cdef DenseCsrCtx ctx
    ctx.mat_ptr = &mat[0, 0]
    ctx.indptr_ptr = &indptr_mv[0]
    ctx.indices_ptr = NULL
    ctx.data_ptr = NULL
    ctx.n_rows = n_rows
    ctx.n_cols = n_cols

    cdef int eff_threads = n_threads if (n_threads > 1 and n_rows * n_cols >= 65536) else 1
    cdef rfm_pool_t *pool = NULL
    cdef int64_t r, nnz

    with nogil:
        pool = rfm_pool_create(eff_threads)
        rfm_pool_run(pool, _dense_count_worker, &ctx)
        for r in range(n_rows):
            ctx.indptr_ptr[r + 1] += ctx.indptr_ptr[r]
        nnz = ctx.indptr_ptr[n_rows]

    if nnz == 0:
        with nogil:
            rfm_pool_destroy(pool)
        return indptr, np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.float32), 0

    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float32)
    cdef int32_t[::1] indices_mv = indices
    cdef float[::1] data_mv = data

    ctx.indices_ptr = &indices_mv[0]
    ctx.data_ptr = &data_mv[0]

    with nogil:
        rfm_pool_run(pool, _dense_fill_worker, &ctx)
        rfm_pool_destroy(pool)

    return indptr, indices, data, 1


def _csr_dense_matmul(
    indptr_arr,
    int32_t[::1] indices,
    float[::1] data,
    float[:, ::1] rhs,
    base=None,
    int n_threads=1
):
    """compute [base + CSR @ rhs] in parallel nogil C without materializing a dense feature matrix"""

    indptr_i64 = np.ascontiguousarray(indptr_arr, dtype=np.int64)
    cdef int64_t[::1] indptr_mv = indptr_i64
    cdef int64_t n_rows = indptr_mv.shape[0] - 1
    cdef int F = rhs.shape[1]

    out = np.empty((n_rows, F), dtype=np.float32)
    if n_rows <= 0 or F <= 0:
        return out

    cdef float[:, ::1] out_mv = out
    cdef float[:, ::1] base_mv
    cdef int has_base = base is not None
    if has_base:
        base_mv = np.ascontiguousarray(base, dtype=np.float32)

    cdef CsrSpMMCtx ctx
    ctx.indptr_ptr = &indptr_mv[0]
    ctx.indices_ptr = &indices[0] if indices.shape[0] > 0 else NULL
    ctx.data_ptr = &data[0] if data.shape[0] > 0 else NULL
    ctx.rhs_ptr = &rhs[0, 0] if rhs.shape[0] > 0 else NULL
    ctx.base_ptr = &base_mv[0, 0] if has_base else NULL
    ctx.out_ptr = &out_mv[0, 0]
    ctx.n_rows = n_rows
    ctx.F = F

    cdef int eff_threads = n_threads if (n_threads > 1 and n_rows * F >= 16384) else 1
    cdef rfm_pool_t *pool = NULL

    with nogil:
        pool = rfm_pool_create(eff_threads)
        rfm_pool_run(pool, _csr_spmm_worker, &ctx)
        rfm_pool_destroy(pool)

    return out


def _csr_matvec(
    indptr_arr,
    int32_t[::1] indices,
    float[::1] data,
    float[::1] vec,
    base=None,
    int n_threads=1
):
    """compute [base + CSR @ vec] in parallel nogil C"""

    indptr_i64 = np.ascontiguousarray(indptr_arr, dtype=np.int64)
    cdef int64_t[::1] indptr_mv = indptr_i64
    cdef int64_t n_rows = indptr_mv.shape[0] - 1

    out = np.empty(n_rows, dtype=np.float32)
    if n_rows <= 0:
        return out

    cdef float[::1] out_mv = out
    cdef float[::1] base_mv
    cdef int has_base = base is not None
    if has_base:
        base_mv = np.ascontiguousarray(base, dtype=np.float32)

    cdef CsrSpMVCtx ctx
    ctx.indptr_ptr = &indptr_mv[0]
    ctx.indices_ptr = &indices[0] if indices.shape[0] > 0 else NULL
    ctx.data_ptr = &data[0] if data.shape[0] > 0 else NULL
    ctx.vec_ptr = &vec[0] if vec.shape[0] > 0 else NULL
    ctx.base_ptr = &base_mv[0] if has_base else NULL
    ctx.out_ptr = &out_mv[0]
    ctx.n_rows = n_rows

    cdef int eff_threads = n_threads if (n_threads > 1 and n_rows >= 4096) else 1
    cdef rfm_pool_t *pool = NULL

    with nogil:
        pool = rfm_pool_create(eff_threads)
        rfm_pool_run(pool, _csr_spmv_worker, &ctx)
        rfm_pool_destroy(pool)

    return out


# a feature column is "hot" when it is non-zero on at least this share of rows; at most this many are batched,
# densest first, and each thread applies its batch after this many of its own steps. HOT_MAX_RATE caps
# learning_rate * threads * batch_steps, at half the level where summed batches were measured to overshoot.
HOT_MIN_DENSITY = 1.0 / 16.0
HOT_MAX_ROWS = 64
HOT_BATCH_STEPS = 64
HOT_MAX_RATE = 16.0


def _select_hot_rows(indices, int64_t n_cols, int64_t n_rows):
    """pick the feature columns dense enough that concurrent threads would keep colliding on their weight rows

    returns (slot_of_column int32[n_cols] with -1 for ordinary columns, hot_columns int32[n_hot])
    """

    slots = np.full(max(n_cols, 1), -1, dtype=np.int32)
    idx = np.asarray(indices)
    if n_cols <= 0 or n_rows <= 0 or len(idx) == 0:
        return slots, np.zeros(0, dtype=np.int32)

    counts = np.bincount(idx, minlength=n_cols)[:n_cols]
    hot = np.flatnonzero(counts >= max(2.0, n_rows * HOT_MIN_DENSITY))
    if len(hot) > HOT_MAX_ROWS:
        hot = np.sort(hot[np.argsort(-counts[hot], kind='stable')[:HOT_MAX_ROWS]])
    hot = np.ascontiguousarray(hot, dtype=np.int32)
    slots[hot] = np.arange(len(hot), dtype=np.int32)
    return slots, hot


# healthy factors stay within a few units of zero; past this magnitude training is running away
LARGE_WEIGHT_THRESHOLD = 1.0e3


def check_weights(w_i, w_if, v_u, v_i, v_uf, v_if):
    """raise AssertionError if any model weight is NaN or infinite, else return (name, magnitude) of the largest one

    the error is raised explicitly rather than with [assert], which `python -O` removes. finiteness is read off each
    array's max and min (NaN and infinity both propagate through them), so every array is scanned twice and never copied.
    """

    worst_label, worst_value = "", 0.0
    for label, weights in (
        ("item weights [w_i]", w_i),
        ("item feature weights [w_if]", w_if),
        ("user factors [v_u]", v_u),
        ("item factors [v_i]", v_i),
        ("user-feature factors [v_uf]", v_uf),
        ("item-feature factors [v_if]", v_if),
    ):
        if weights.size == 0:
            continue
        value = max(float(np.max(weights)), -float(np.min(weights)))
        if not np.isfinite(value):
            raise AssertionError("{} are not finite - try decreasing feature/sample_weight magnitudes".format(label))
        if value > worst_value:
            worst_label, worst_value = label, value
    return worst_label, worst_value


def reg_penalty(alpha, beta, w_i, w_if, v_u, v_i, v_uf, v_if):
    """calculate the total regularization penalty for all model weights without allocating temporary full-size arrays"""

    penalty = 0.0
    penalty += float(alpha) * float(np.einsum('i,i->', w_i, w_i, dtype=np.float64))
    penalty += float(alpha) * float(np.einsum('ij,ij->', v_u, v_u, dtype=np.float64))
    penalty += float(alpha) * float(np.einsum('ij,ij->', v_i, v_i, dtype=np.float64))
    penalty += float(beta) * float(np.einsum('i,i->', w_if, w_if, dtype=np.float64))
    penalty += float(beta) * float(np.einsum('ij,ij->', v_uf, v_uf, dtype=np.float64))
    penalty += float(beta) * float(np.einsum('ij,ij->', v_if, v_if, dtype=np.float64))
    return penalty


# --------------------------------
# [RankFM] core modeling functions
# --------------------------------

def _fit_csr(
    const int32_t[::1] u_idx,
    const int32_t[::1] i_idx,
    sample_weight,
    ui_indptr_arr,
    int32_t[::1] ui_indices,
    uf_indptr_arr,
    int32_t[::1] uf_indices,
    float[::1] uf_data,
    int x_uf_any,
    if_indptr_arr,
    int32_t[::1] if_indices,
    float[::1] if_data,
    int x_if_any,
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
    int seed=1492,
    int n_threads=1,
    bint uniform_feature_reg=False,
    bint normalize_steps=True,
    bint batched_dense=False
):
    """learn model weights via multi-threaded nogil BPR/WARP SGD with materialized effective item cache and native CSR features

    interactions arrive as parallel arrays: [u_idx] and [i_idx] hold the user and item index of each one, and
    [sample_weight] is either None (every interaction weighs 1) or a float32 array of the same length
    """

    cdef int64_t N = u_idx.shape[0]
    if i_idx.shape[0] != N:
        raise ValueError("[u_idx] and [i_idx] must have the same length")
    cdef const float[::1] sw_mv
    cdef bint has_weights = sample_weight is not None
    if has_weights:
        sw_mv = sample_weight
        if sw_mv.shape[0] != N:
            raise ValueError("[sample_weight] must have one entry per interaction")
    cdef int32_t I = <int32_t>v_i.shape[0]
    cdef int F = <int>v_u.shape[1]
    cdef int64_t P_uf = v_uf.shape[0]
    cdef int64_t P_if = v_if.shape[0]

    if N == 0 or I <= 1 or epochs <= 0:
        return

    # the WARP multiplier log((I - 1) / sampled) turns negative once sampled > I - 1, which would flip the
    # gradient into ascent on catalogs smaller than [max_samples]
    if max_samples > I - 1:
        max_samples = I - 1

    cdef int schedule_mode
    if learning_schedule == 'constant':
        schedule_mode = 0
    elif learning_schedule == 'invscaling':
        schedule_mode = 1
    else:
        raise ValueError('unknown [learning_schedule]')

    ui_indptr_i64 = np.ascontiguousarray(ui_indptr_arr, dtype=np.int64)
    uf_indptr_i64 = np.ascontiguousarray(uf_indptr_arr, dtype=np.int64)
    if_indptr_i64 = np.ascontiguousarray(if_indptr_arr, dtype=np.int64)

    cdef int64_t[::1] ui_indptr = ui_indptr_i64
    cdef int64_t[::1] uf_indptr = uf_indptr_i64
    cdef int64_t[::1] if_indptr = if_indptr_i64

    cdef int eff_threads = n_threads if (n_threads > 1 and N >= 1024) else 1
    if eff_threads > 1 and N // eff_threads < 256:
        eff_threads = max(1, <int>(N // 256))

    # hot rows for batched dense updates: chosen here, before any C allocation, since NumPy may raise. the
    # choice does not depend on the thread count, so a model trains the same way however many threads it gets.
    cdef int32_t n_hot_uf = 0
    cdef int32_t n_hot_if = 0
    cdef int32_t[::1] hot_uf_slot_mv, hot_uf_rows_mv, hot_if_slot_mv, hot_if_rows_mv
    cdef float *hot_acc_base = NULL
    cdef size_t hot_acc_stride = 0
    if batched_dense and not uniform_feature_reg:
        if x_uf_any:
            hot_uf_slot_arr, hot_uf_rows_arr = _select_hot_rows(uf_indices, P_uf, uf_indptr.shape[0] - 1)
            n_hot_uf = <int32_t>len(hot_uf_rows_arr)
            if n_hot_uf > 0:
                hot_uf_slot_mv = hot_uf_slot_arr
                hot_uf_rows_mv = hot_uf_rows_arr
        if x_if_any:
            hot_if_slot_arr, hot_if_rows_arr = _select_hot_rows(if_indices, P_if, I)
            n_hot_if = <int32_t>len(hot_if_rows_arr)
            if n_hot_if > 0:
                hot_if_slot_mv = hot_if_slot_arr
                hot_if_rows_mv = hot_if_rows_arr

    # per-user step scale 1 / (1 + |x_u|^2), computed once (see the step normalization note in the worker)
    cdef float[::1] uf_step_scale_mv
    cdef bint scale_user_steps = normalize_steps and x_uf_any
    if scale_user_steps:
        uf_row_ids = np.repeat(np.arange(uf_indptr.shape[0] - 1, dtype=np.int64), np.diff(uf_indptr_i64))
        uf_values = np.asarray(uf_data, dtype=np.float64)
        uf_sq_norm = np.bincount(uf_row_ids, weights=uf_values * uf_values, minlength=uf_indptr.shape[0] - 1)
        uf_step_scale_mv = np.ascontiguousarray(1.0 / (1.0 + uf_sq_norm), dtype=np.float32)

    # Pad per-thread scratch stride (4 * F floats) to 64-byte cache lines (16 floats)
    cdef size_t raw_floats = <size_t>(4 * F)
    cdef size_t scratch_stride = (raw_floats + 15u) & ~(<size_t>15u)

    cdef float *mult_lut = <float *>rfm_aligned_malloc(<size_t>(max_samples + 1) * sizeof(float))
    cdef float *scratch_base = <float *>rfm_aligned_malloc(<size_t>eff_threads * scratch_stride * sizeof(float))
    cdef double *ll_slots = <double *>calloc(<size_t>(eff_threads * 8), sizeof(double))
    cdef float *b_eff_alloc = NULL
    cdef float *v_eff_alloc = NULL
    cdef int64_t *last_step_uf = NULL
    cdef int64_t *last_step_if = NULL
    cdef rfm_pool_t *pool = NULL

    if mult_lut == NULL or scratch_base == NULL or ll_slots == NULL:
        rfm_aligned_free(mult_lut)
        rfm_aligned_free(scratch_base)
        free(ll_slots)
        raise MemoryError("failed to allocate RankFM training scratch buffers")

    if x_if_any:
        b_eff_alloc = <float *>rfm_aligned_malloc(<size_t>I * sizeof(float))
        v_eff_alloc = <float *>rfm_aligned_malloc(<size_t>I * <size_t>F * sizeof(float))
        if b_eff_alloc == NULL or v_eff_alloc == NULL:
            rfm_aligned_free(mult_lut)
            rfm_aligned_free(scratch_base)
            free(ll_slots)
            rfm_aligned_free(b_eff_alloc)
            rfm_aligned_free(v_eff_alloc)
            raise MemoryError("failed to allocate RankFM effective item cache")

    if n_hot_uf > 0 or n_hot_if > 0:
        hot_acc_stride = <size_t>(<int64_t>n_hot_uf * (F + 1) + <int64_t>n_hot_if * (F + 2))
        hot_acc_stride = (hot_acc_stride + 15u) & ~(<size_t>15u)
        hot_acc_base = <float *>rfm_aligned_malloc(<size_t>eff_threads * hot_acc_stride * sizeof(float))
        if hot_acc_base == NULL:
            rfm_aligned_free(mult_lut)
            rfm_aligned_free(scratch_base)
            free(ll_slots)
            rfm_aligned_free(b_eff_alloc)
            rfm_aligned_free(v_eff_alloc)
            raise MemoryError("failed to allocate RankFM batched-update buffers")

    if uniform_feature_reg:
        if x_uf_any and P_uf > 0:
            last_step_uf = <int64_t *>calloc(<size_t>P_uf, sizeof(int64_t))
        if x_if_any and P_if > 0:
            last_step_if = <int64_t *>calloc(<size_t>P_if, sizeof(int64_t))

    cdef int s_idx
    cdef double denom_log_I = log(<double>I)
    mult_lut[0] = 1.0
    for s_idx in range(1, max_samples + 1):
        mult_lut[s_idx] = <float>(log(<double>(I - 1) / <double>s_idx) / denom_log_I)

    # Sync interval for refreshing (b_eff, V_eff) when item_features are present:
    # amortizes O(nnz_if * F) refresh to < 1% of WARP work while bounding staleness <= 32,768 steps
    cdef int64_t sync_interval
    if not x_if_any:
        sync_interval = N
    elif N <= 2048:
        sync_interval = max(<int64_t>32, N // 4)
    else:
        sync_interval = max(<int64_t>2048, min(N, min(<int64_t>32768, max(<int64_t>4096, if_indptr[I] // 4))))

    cdef RefreshEffItemsCtx ref_ctx
    ref_ctx.w_i_ptr = &w_i[0]
    ref_ctx.w_if_ptr = &w_if[0]
    ref_ctx.v_i_ptr = &v_i[0, 0]
    ref_ctx.v_if_ptr = &v_if[0, 0]
    ref_ctx.if_indptr_ptr = &if_indptr[0]
    ref_ctx.if_indices_ptr = &if_indices[0] if if_indices.shape[0] > 0 else NULL
    ref_ctx.if_data_ptr = &if_data[0] if if_data.shape[0] > 0 else NULL
    ref_ctx.b_eff_ptr = b_eff_alloc
    ref_ctx.v_eff_ptr = v_eff_alloc
    ref_ctx.I = I
    ref_ctx.F = F

    cdef FitBlockCtx ctx
    ctx.u_ptr = &u_idx[0]
    ctx.i_ptr = &i_idx[0]
    ctx.sw_ptr = &sw_mv[0] if has_weights else NULL
    ctx.ui_indptr_ptr = &ui_indptr[0]
    ctx.ui_indices_ptr = &ui_indices[0] if ui_indices.shape[0] > 0 else NULL
    ctx.uf_indptr_ptr = &uf_indptr[0]
    ctx.uf_indices_ptr = &uf_indices[0] if uf_indices.shape[0] > 0 else NULL
    ctx.uf_data_ptr = &uf_data[0] if uf_data.shape[0] > 0 else NULL
    ctx.if_indptr_ptr = &if_indptr[0]
    ctx.if_indices_ptr = &if_indices[0] if if_indices.shape[0] > 0 else NULL
    ctx.if_data_ptr = &if_data[0] if if_data.shape[0] > 0 else NULL
    ctx.w_i_ptr = &w_i[0]
    ctx.w_if_ptr = &w_if[0]
    ctx.v_u_ptr = &v_u[0, 0]
    ctx.v_i_ptr = &v_i[0, 0]
    ctx.v_uf_ptr = &v_uf[0, 0]
    ctx.v_if_ptr = &v_if[0, 0]
    ctx.b_eff_ptr = b_eff_alloc if x_if_any else &w_i[0]
    ctx.v_eff_ptr = v_eff_alloc if x_if_any else &v_i[0, 0]
    ctx.mult_lut_ptr = mult_lut
    ctx.uf_step_scale_ptr = &uf_step_scale_mv[0] if scale_user_steps else NULL
    ctx.normalize_steps = 1 if normalize_steps else 0
    ctx.scratch_base = scratch_base
    ctx.scratch_stride = scratch_stride
    ctx.ll_slots = ll_slots
    ctx.last_step_uf = last_step_uf
    ctx.last_step_if = last_step_if
    ctx.hot.uf_slot = &hot_uf_slot_mv[0] if n_hot_uf > 0 else NULL
    ctx.hot.uf_rows = &hot_uf_rows_mv[0] if n_hot_uf > 0 else NULL
    ctx.hot.if_slot = &hot_if_slot_mv[0] if n_hot_if > 0 else NULL
    ctx.hot.if_rows = &hot_if_rows_mv[0] if n_hot_if > 0 else NULL
    ctx.hot.acc_base = hot_acc_base
    ctx.hot.acc_stride = hot_acc_stride
    ctx.hot.n_uf = n_hot_uf
    ctx.hot.n_if = n_hot_if
    ctx.hot.batch_steps = HOT_BATCH_STEPS
    ctx.hot.scale = 1.0
    cdef double hot_max_rate = HOT_MAX_RATE
    cdef double weight_scale = 1.0
    if has_weights and (n_hot_uf > 0 or n_hot_if > 0):
        weight_scale = max(1.0, float(np.mean(sample_weight, dtype=np.float64)))
    ctx.I = I
    ctx.F = F
    ctx.max_samples = max_samples
    ctx.x_uf_any = x_uf_any
    ctx.x_if_any = x_if_any
    ctx.verbose = 1 if verbose else 0
    ctx.uniform_reg = 1 if uniform_feature_reg else 0
    ctx.user_dropout = user_dropout
    ctx.base_seed = <uint64_t>(<uint32_t>seed)

    cdef int epoch, t_i
    cdef bint warned_large = False
    cdef int64_t b_start, b_end, p_idx, q_idx, rem_steps
    cdef int64_t n_u_rows = max(<int64_t>1, <int64_t>v_u.shape[0])
    cdef double uf_density = (<double>uf_indptr[n_u_rows]) / max(1.0, <double>n_u_rows * <double>max(<int64_t>1, P_uf)) if x_uf_any else 0.0
    cdef double if_density = (2.0 * <double>if_indptr[I]) / max(1.0, <double>I * <double>max(<int64_t>1, P_if)) if x_if_any else 0.0
    cdef float eta, d_reg_a = 2.0 * alpha, d_reg_b = 2.0 * beta, lazy_tail
    cdef double log_likelihood

    try:
        pool = rfm_pool_create(eff_threads)

        for epoch in range(epochs):
            if schedule_mode == 0:
                eta = learning_rate
            else:
                eta = learning_rate / <float>pow(<double>(epoch + 1), <double>learning_exponent)

            ctx.eta = eta
            ctx.reg_a_keep = 1.0 - eta * d_reg_a
            ctx.reg_b_keep = 1.0 - eta * d_reg_b
            ctx.hot.scale = <float>min(1.0, hot_max_rate / (<double>eta * weight_scale * <double>eff_threads * <double>ctx.hot.batch_steps))
            if ctx.uniform_reg:
                ctx.log_reg_uf_lazy = <float>log(max(1e-6, 1.0 - (<double>eta * <double>d_reg_b) * max(uf_density, 1.0 / <double>N)))
                ctx.log_reg_if_lazy = <float>log(max(1e-6, 1.0 - (<double>eta * <double>d_reg_b) * max(if_density, 1.0 / <double>N)))
                if last_step_uf != NULL:
                    memset(last_step_uf, 0, <size_t>P_uf * sizeof(int64_t))
                if last_step_if != NULL:
                    memset(last_step_if, 0, <size_t>P_if * sizeof(int64_t))
            else:
                ctx.log_reg_uf_lazy = 0.0
                ctx.log_reg_if_lazy = 0.0

            if verbose:
                memset(ll_slots, 0, <size_t>(eff_threads * 8) * sizeof(double))

            with nogil:
                rfm_perm_init(&ctx.perm, <uint64_t>N, ctx.base_seed ^ ((<uint64_t>(epoch + 1)) * 0x9E3779B97F4A7C15ULL))

                b_start = 0
                while b_start < N:
                    b_end = b_start + sync_interval
                    if b_end > N:
                        b_end = N

                    if x_if_any:
                        rfm_pool_run(pool, _refresh_eff_items_worker, &ref_ctx)

                    ctx.block_start = b_start
                    ctx.block_end = b_end
                    ctx.cursor = b_start
                    ctx.grain = max(<int64_t>64, min(<int64_t>256, (b_end - b_start) // max(<int64_t>1, <int64_t>(eff_threads * 4))))
                    # one RNG stream per (epoch, chunk start): the epoch sits above bit 40 so epochs can never share a stream
                    ctx.stream_epoch_base = (<uint64_t>(epoch + 1)) << 40
                    rfm_pool_run(pool, _fit_block_worker, &ctx)

                    b_start = b_end

                if ctx.uniform_reg:
                    if last_step_uf != NULL:
                        for p_idx in range(P_uf):
                            rem_steps = N - last_step_uf[p_idx]
                            if rem_steps > 0:
                                lazy_tail = <float>exp(<double>ctx.log_reg_uf_lazy * <double>rem_steps)
                                rfm_vec_scale(F, lazy_tail, &ctx.v_uf_ptr[p_idx * <int64_t>F])
                    if last_step_if != NULL:
                        for q_idx in range(P_if):
                            rem_steps = N - last_step_if[q_idx]
                            if rem_steps > 0:
                                lazy_tail = <float>exp(<double>ctx.log_reg_if_lazy * <double>rem_steps)
                                ctx.w_if_ptr[q_idx] *= lazy_tail
                                rfm_vec_scale(F, lazy_tail, &ctx.v_if_ptr[q_idx * <int64_t>F])

            # weights can also explode by many orders of magnitude while staying finite, hence the magnitude warning
            large_label, large_value = check_weights(w_i, w_if, v_u, v_i, v_uf, v_if)
            if not warned_large:
                if large_value > LARGE_WEIGHT_THRESHOLD:
                    warned_large = True
                    warnings.warn(
                        "training appears to be diverging: {} reached magnitude {:.3g} after epoch {}. "
                        "lower [learning_rate], scale down feature/sample_weight values, or raise [alpha]/[beta]".format(
                            large_label, large_value, epoch + 1
                        ),
                        RuntimeWarning
                    )

            if verbose:
                log_likelihood = 0.0
                for t_i in range(eff_threads):
                    log_likelihood += ll_slots[t_i * 8]
                penalty = reg_penalty(alpha, beta, w_i, w_if, v_u, v_i, v_uf, v_if)
                log_likelihood = round(log_likelihood - penalty, 2)
                print("\ntraining epoch:", epoch)
                print("log likelihood:", log_likelihood)

    finally:
        with nogil:
            rfm_pool_destroy(pool)
        rfm_aligned_free(mult_lut)
        rfm_aligned_free(scratch_base)
        free(ll_slots)
        rfm_aligned_free(b_eff_alloc)
        rfm_aligned_free(v_eff_alloc)
        free(last_step_uf)
        free(last_step_if)
        rfm_aligned_free(hot_acc_base)


def _fit(
    int32_t[:, ::1] interactions,
    float[::1] sample_weight,
    ui_indptr,
    int32_t[::1] ui_indices,
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
    int seed=1492,
    int n_threads=1,
    bint uniform_feature_reg=False,
    bint normalize_steps=True
):
    """backwards-compatible dense-feature wrapper delegating to _fit_csr"""

    uf_indptr, uf_indices, uf_data, x_uf_any = dense_to_csr(x_uf, n_threads)
    if_indptr, if_indices, if_data, x_if_any = dense_to_csr(x_if, n_threads)
    _fit_csr(
        np.ascontiguousarray(interactions[:, 0]),
        np.ascontiguousarray(interactions[:, 1]),
        sample_weight,
        ui_indptr,
        ui_indices,
        uf_indptr,
        uf_indices,
        uf_data,
        x_uf_any,
        if_indptr,
        if_indices,
        if_data,
        x_if_any,
        w_i,
        w_if,
        v_u,
        v_i,
        v_uf,
        v_if,
        alpha,
        beta,
        learning_rate,
        learning_schedule,
        learning_exponent,
        max_samples,
        epochs,
        verbose,
        user_dropout,
        seed,
        n_threads,
        uniform_feature_reg,
        normalize_steps
    )


def _fold_in(
    const int32_t[::1] u_idx,
    const int32_t[::1] i_idx,
    sample_weight,
    ui_indptr_arr,
    int32_t[::1] ui_indices,
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
    int seed=1492,
    int n_threads=1
):
    """learn or refine user latent factors [v_u] in multi-threaded nogil C while holding item/feature weights fixed"""

    cdef int64_t N = u_idx.shape[0]
    if i_idx.shape[0] != N:
        raise ValueError("[u_idx] and [i_idx] must have the same length")
    cdef const float[::1] sw_mv
    cdef bint has_weights = sample_weight is not None
    if has_weights:
        sw_mv = sample_weight
        if sw_mv.shape[0] != N:
            raise ValueError("[sample_weight] must have one entry per interaction")
    cdef int32_t I = <int32_t>v_i_eff.shape[0]
    cdef int F = <int>v_u.shape[1]

    if N == 0 or I <= 1 or epochs <= 0:
        return

    # the WARP multiplier log((I - 1) / sampled) turns negative once sampled > I - 1, which would flip the
    # gradient into ascent on catalogs smaller than [max_samples]
    if max_samples > I - 1:
        max_samples = I - 1

    cdef int schedule_mode
    if learning_schedule == 'constant':
        schedule_mode = 0
    elif learning_schedule == 'invscaling':
        schedule_mode = 1
    else:
        raise ValueError('unknown [learning_schedule]')

    ui_indptr_i64 = np.ascontiguousarray(ui_indptr_arr, dtype=np.int64)
    cdef int64_t[::1] ui_indptr = ui_indptr_i64

    cdef int eff_threads = n_threads if (n_threads > 1 and N >= 1024) else 1
    if eff_threads > 1 and N // eff_threads < 256:
        eff_threads = max(1, <int>(N // 256))

    cdef size_t scratch_stride = (<size_t>(2 * F) + 15u) & ~(<size_t>15u)
    cdef float *mult_lut = <float *>rfm_aligned_malloc(<size_t>(max_samples + 1) * sizeof(float))
    cdef float *scratch_base = <float *>rfm_aligned_malloc(<size_t>eff_threads * scratch_stride * sizeof(float))
    cdef rfm_pool_t *pool = NULL

    if mult_lut == NULL or scratch_base == NULL:
        rfm_aligned_free(mult_lut)
        rfm_aligned_free(scratch_base)
        raise MemoryError("failed to allocate RankFM fold-in scratch buffers")

    cdef int s_idx
    cdef double denom_log_I = log(<double>I)
    mult_lut[0] = 1.0
    for s_idx in range(1, max_samples + 1):
        mult_lut[s_idx] = <float>(log(<double>(I - 1) / <double>s_idx) / denom_log_I)

    cdef FoldInCtx ctx
    ctx.u_ptr = &u_idx[0]
    ctx.i_ptr = &i_idx[0]
    ctx.sw_ptr = &sw_mv[0] if has_weights else NULL
    ctx.ui_indptr_ptr = &ui_indptr[0]
    ctx.ui_indices_ptr = &ui_indices[0] if ui_indices.shape[0] > 0 else NULL
    ctx.u_feat_ptr = &u_feat_emb[0, 0]
    ctx.item_bias_ptr = &item_bias[0]
    ctx.v_i_eff_ptr = &v_i_eff[0, 0]
    ctx.v_u_ptr = &v_u[0, 0]
    ctx.mult_lut_ptr = mult_lut
    ctx.scratch_base = scratch_base
    ctx.scratch_stride = scratch_stride
    ctx.N = N
    ctx.grain = max(<int64_t>64, min(<int64_t>256, N // max(<int64_t>1, <int64_t>(eff_threads * 4))))
    ctx.I = I
    ctx.F = F
    ctx.max_samples = max_samples
    ctx.base_seed = <uint64_t>(<uint32_t>seed)

    cdef int epoch
    cdef float eta, d_reg_a = 2.0 * alpha

    try:
        with nogil:
            pool = rfm_pool_create(eff_threads)
            for epoch in range(epochs):
                if schedule_mode == 0:
                    eta = learning_rate
                else:
                    eta = learning_rate / <float>pow(<double>(epoch + 1), <double>learning_exponent)

                ctx.eta = eta
                ctx.reg_a_keep = 1.0 - eta * d_reg_a
                ctx.cursor = 0
                ctx.stream_epoch_base = ((<uint64_t>(epoch + 1)) << 40) | (1ULL << 39)

                rfm_perm_init(&ctx.perm, <uint64_t>N, (ctx.base_seed + 0x5A5A5A5AULL) ^ ((<uint64_t>(epoch + 1)) * 0x9E3779B97F4A7C15ULL))
                rfm_pool_run(pool, _fold_in_worker, &ctx)
    finally:
        with nogil:
            rfm_pool_destroy(pool)
        rfm_aligned_free(mult_lut)
        rfm_aligned_free(scratch_base)


def _predict(
    int32_t[::1] u_idx,
    int32_t[::1] i_idx,
    float[:, ::1] u_emb,
    float[::1] item_bias,
    float[:, ::1] v_i_eff,
    int n_threads=1
):
    """score (user, item) pairs in multi-threaded nogil C using integer indices (-1 indicates missing)"""

    cdef int64_t N = u_idx.shape[0]
    cdef int F = <int>u_emb.shape[1]

    scores = np.full(N, np.nan, dtype=np.float32)
    if N == 0 or u_emb.shape[0] == 0 or item_bias.shape[0] == 0:
        return scores

    cdef float[::1] scores_mv = scores
    cdef PredictCtx ctx
    ctx.u_ptr = &u_idx[0]
    ctx.i_ptr = &i_idx[0]
    ctx.u_emb_ptr = &u_emb[0, 0]
    ctx.item_bias_ptr = &item_bias[0]
    ctx.v_i_eff_ptr = &v_i_eff[0, 0]
    ctx.scores_ptr = &scores_mv[0]
    ctx.N = N
    ctx.F = F

    cdef int eff_threads = n_threads if (n_threads > 1 and N >= 8192) else 1
    cdef rfm_pool_t *pool = NULL

    with nogil:
        pool = rfm_pool_create(eff_threads)
        rfm_pool_run(pool, _predict_worker, &ctx)
        rfm_pool_destroy(pool)

    return scores


def _select_topk_chunk(
    float[:, ::1] scores_chunk,
    float[::1] item_bias,
    int32_t[::1] chunk_u_idx,
    ui_indptr_arr,
    int32_t[::1] ui_indices,
    int n_items,
    bint filter_previous,
    int32_t[:, ::1] out_items,
    int n_threads=1
):
    """select sorted top-K item indices per row from a BLAS score chunk using parallel nogil size-K min-heaps"""

    cdef int64_t B = scores_chunk.shape[0]
    cdef int32_t I = <int32_t>scores_chunk.shape[1]
    cdef int32_t K = <int32_t>n_items
    if B == 0 or K <= 0:
        return

    cdef int64_t[::1] ui_indptr
    if filter_previous:
        ui_indptr_i64 = np.ascontiguousarray(ui_indptr_arr, dtype=np.int64)
        ui_indptr = ui_indptr_i64

    cdef int eff_threads = n_threads if (n_threads > 1 and B >= 32) else 1
    if eff_threads > B:
        eff_threads = <int>B

    cdef TopKChunkCtx ctx
    ctx.scores_ptr = &scores_chunk[0, 0]
    ctx.bias_ptr = &item_bias[0]
    ctx.u_ptr = &chunk_u_idx[0]
    ctx.indptr_ptr = &ui_indptr[0] if filter_previous else NULL
    ctx.indices_ptr = &ui_indices[0] if (filter_previous and ui_indices.shape[0] > 0) else NULL
    ctx.out_ptr = &out_items[0, 0]
    ctx.B = B
    ctx.cursor = 0
    ctx.grain = max(<int64_t>8, min(<int64_t>64, B // max(<int64_t>1, <int64_t>(eff_threads * 4))))
    ctx.I = I
    ctx.K = K
    ctx.filter_previous = 1 if filter_previous else 0

    # heap slots padded to whole cache lines (16 four-byte entries) so neighbouring threads never share one
    ctx.heap_stride = (<size_t>K + 15u) & ~(<size_t>15u)
    ctx.seen_stride = (<size_t>I + 7u) & ~(<size_t>7u)
    ctx.heap_s_base = <float *>rfm_aligned_malloc(<size_t>eff_threads * ctx.heap_stride * sizeof(float))
    ctx.heap_i_base = <int32_t *>rfm_aligned_malloc(<size_t>eff_threads * ctx.heap_stride * sizeof(int32_t))
    ctx.seen_base = <int64_t *>rfm_aligned_malloc(<size_t>eff_threads * ctx.seen_stride * sizeof(int64_t)) if filter_previous else NULL

    cdef rfm_pool_t *pool = NULL
    try:
        if ctx.heap_s_base == NULL or ctx.heap_i_base == NULL or (filter_previous and ctx.seen_base == NULL):
            raise MemoryError("failed to allocate RankFM top-K selection buffers")
        with nogil:
            pool = rfm_pool_create(eff_threads)
            rfm_pool_run(pool, _select_topk_worker, &ctx)
            rfm_pool_destroy(pool)
    finally:
        rfm_aligned_free(ctx.heap_s_base)
        rfm_aligned_free(ctx.heap_i_base)
        rfm_aligned_free(ctx.seen_base)
