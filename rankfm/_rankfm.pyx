#!python
#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

# -----------------------
# [C/Python] dependencies
# -----------------------

from libc.math cimport exp, log, pow, sqrt
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
    int32_t rfm_alias_draw(rfm_rng_t *rng, uint32_t n, const float *prob, const int32_t *alias) noexcept nogil

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
    void rfm_vec_mul(int F, const float *a, const float *b, float *out) noexcept nogil
    void rfm_vec_mul_axpy(int F, float alpha, const float *gate, const float *x, float *y) noexcept nogil
    void rfm_vec_reg_mul_axpy(int F, float keep, float alpha, const float *gate, const float *x, float *y) noexcept nogil
    float rfm_norm2(int F, const float *x) noexcept nogil
    float rfm_normalize_copy(int F, const float *src, float *dst, float eps) noexcept nogil
    float rfm_normalize_inplace(int F, float *x, float eps) noexcept nogil
    void rfm_cosine_grad_project(int F, float eff_inv_norm, const float *x_hat, const float *g_hat, float *out_gx) noexcept nogil
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

    cdef int eff_threads = n_threads if (n_threads > 1 and N >= 4096 and I >= 128) else 1
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

    cdef int eff_threads = n_threads if (n_threads > 1 and N >= 4096 and I >= 128) else 1
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


# =====================================================================
# RankFM v0.5.0 Neural-FM, Contrastive LogQ & Multi-Interest Extensions
# =====================================================================

cdef inline float safe_sigmoid_pos(float x) noexcept nogil:
    """numerically stable 1 / (1 + exp(-x))"""
    if x > 20.0:
        return 1.0
    elif x < -20.0:
        return exp(x)
    return 1.0 / (1.0 + exp(-x))


cdef inline float clamp_f32(float x, float lo, float hi) noexcept nogil:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


cdef struct RefreshItemsV050Ctx:
    const float *w_i_ptr
    const float *w_if_ptr
    const float *v_i_ptr
    const float *v_if_ptr
    const int64_t *if_indptr_ptr
    const int32_t *if_indices_ptr
    const float *if_data_ptr
    const int32_t *if_field_ptr
    const int32_t *if_is_dyn_ptr
    const float *gamma_if_ptr
    const float *item_log_freq_ptr
    const float *a_gate_ptr
    const float *b_gate_ptr
    float *b_eff_ptr
    float *v_raw_eff_ptr
    float *v_eff_ptr
    float *inv_norm_eff_ptr
    float *g_item_ptr
    float *s_item_ptr
    float freq_gate_static_drop
    float out_scale
    int32_t I
    int F
    int x_if_any
    int use_field_gates
    int use_freq_gate
    int use_cosine


cdef void _refresh_eff_items_v050_worker(int tid, int n_threads, void *raw_ctx) noexcept nogil:
    cdef RefreshItemsV050Ctx *ctx = <RefreshItemsV050Ctx *>raw_ctx
    cdef int32_t m_start = <int32_t>((<int64_t>ctx.I * tid) // n_threads)
    cdef int32_t m_end = <int32_t>((<int64_t>ctx.I * (tid + 1)) // n_threads)
    if m_start >= m_end:
        return

    cdef int F = ctx.F
    cdef float *s_vec = <float *>rfm_aligned_malloc(<size_t>(2 * F) * sizeof(float))
    if s_vec == NULL:
        return
    cdef float *d_vec = &s_vec[F]

    cdef int32_t m, q, fld, is_dyn
    cdef int64_t idx
    cdef int f
    cdef float val, bias, log_n, g_mf, inv_n
    cdef const float *vi_row
    cdef float *v_raw_row
    cdef float *v_out_row
    cdef float *g_row
    cdef float *s_row

    for m in range(m_start, m_end):
        bias = ctx.w_i_ptr[m]
        vi_row = &ctx.v_i_ptr[<int64_t>m * <int64_t>F]
        v_raw_row = &ctx.v_raw_eff_ptr[<int64_t>m * <int64_t>F]
        v_out_row = &ctx.v_eff_ptr[<int64_t>m * <int64_t>F]

        if ctx.x_if_any:
            rfm_vec_zero(F, s_vec)
            rfm_vec_zero(F, d_vec)
            for idx in range(ctx.if_indptr_ptr[m], ctx.if_indptr_ptr[m + 1]):
                q = ctx.if_indices_ptr[idx]
                val = ctx.if_data_ptr[idx]
                bias += val * ctx.w_if_ptr[q]
                is_dyn = ctx.if_is_dyn_ptr[q] if (ctx.use_freq_gate and ctx.if_is_dyn_ptr != NULL) else 0
                if ctx.use_field_gates and ctx.if_field_ptr != NULL and ctx.gamma_if_ptr != NULL:
                    fld = ctx.if_field_ptr[q]
                    if is_dyn:
                        rfm_vec_mul_axpy(F, val, &ctx.gamma_if_ptr[<int64_t>fld * <int64_t>F], &ctx.v_if_ptr[<int64_t>q * <int64_t>F], d_vec)
                    else:
                        rfm_vec_mul_axpy(F, val, &ctx.gamma_if_ptr[<int64_t>fld * <int64_t>F], &ctx.v_if_ptr[<int64_t>q * <int64_t>F], s_vec)
                else:
                    if is_dyn:
                        rfm_vec_axpy(F, val, &ctx.v_if_ptr[<int64_t>q * <int64_t>F], d_vec)
                    else:
                        rfm_vec_axpy(F, val, &ctx.v_if_ptr[<int64_t>q * <int64_t>F], s_vec)

            if ctx.use_freq_gate and ctx.a_gate_ptr != NULL and ctx.b_gate_ptr != NULL and ctx.item_log_freq_ptr != NULL:
                log_n = ctx.item_log_freq_ptr[m]
                g_row = &ctx.g_item_ptr[<int64_t>m * <int64_t>F] if ctx.g_item_ptr != NULL else NULL
                s_row = &ctx.s_item_ptr[<int64_t>m * <int64_t>F] if ctx.s_item_ptr != NULL else NULL
                for f in range(F):
                    g_mf = safe_sigmoid_pos(ctx.a_gate_ptr[f] * log_n + ctx.b_gate_ptr[f])
                    if g_row != NULL:
                        g_row[f] = g_mf
                    if s_row != NULL:
                        s_row[f] = s_vec[f]
                    v_raw_row[f] = g_mf * vi_row[f] + (1.0 - ctx.freq_gate_static_drop * g_mf) * s_vec[f] + d_vec[f]
            else:
                for f in range(F):
                    v_raw_row[f] = vi_row[f] + s_vec[f] + d_vec[f]
        else:
            if ctx.use_freq_gate and ctx.a_gate_ptr != NULL and ctx.b_gate_ptr != NULL and ctx.item_log_freq_ptr != NULL:
                log_n = ctx.item_log_freq_ptr[m]
                g_row = &ctx.g_item_ptr[<int64_t>m * <int64_t>F] if ctx.g_item_ptr != NULL else NULL
                if ctx.s_item_ptr != NULL:
                    rfm_vec_zero(F, &ctx.s_item_ptr[<int64_t>m * <int64_t>F])
                for f in range(F):
                    g_mf = safe_sigmoid_pos(ctx.a_gate_ptr[f] * log_n + ctx.b_gate_ptr[f])
                    if g_row != NULL:
                        g_row[f] = g_mf
                    v_raw_row[f] = g_mf * vi_row[f]
            else:
                rfm_vec_copy(F, vi_row, v_raw_row)

        ctx.b_eff_ptr[m] = bias

        if ctx.use_cosine:
            inv_n = rfm_normalize_copy(F, v_raw_row, v_out_row, 1e-6)
            if ctx.inv_norm_eff_ptr != NULL:
                ctx.inv_norm_eff_ptr[m] = inv_n
            if ctx.out_scale != 1.0:
                rfm_vec_scale(F, ctx.out_scale, v_out_row)
        else:
            rfm_vec_copy(F, v_raw_row, v_out_row)
            if ctx.inv_norm_eff_ptr != NULL:
                ctx.inv_norm_eff_ptr[m] = 1.0
            if ctx.out_scale != 1.0:
                rfm_vec_scale(F, ctx.out_scale, v_out_row)

    rfm_aligned_free(s_vec)


cdef inline void _encode_single_user_v050(
    int F,
    int M,
    int32_t u_v_idx,
    int32_t u_uf_idx,
    int32_t u_uh_idx,
    int32_t seq_item,
    int32_t exclude_item,
    int use_v_u,
    int u_has_uf,
    int use_history,
    int use_seq_anchor,
    int use_field_gates,
    int use_event_gates,
    int use_cosine,
    float history_exponent,
    float inv_route_temp,
    const float *v_u_ptr,
    const float *v_uf_ptr,
    const int64_t *uf_indptr,
    const int32_t *uf_indices,
    const float *uf_data,
    const int32_t *uf_field,
    const float *gamma_uf_ptr,
    const int64_t *uh_indptr,
    const int32_t *uh_indices,
    const float *uh_weights,
    const int32_t *uh_kinds,
    const float *p_hist_ptr,
    const float *g_event_ptr,
    const float *p_seq_ptr,
    const float *g_seq_ptr,
    const float *c_proto_ptr,
    float *u_base_buf,
    float *u_hist_buf,
    float *head_w_buf,
    float *route_buf,
    float *u_raw_out,
    float *u_out,
    float *u_inv_norm_out
) noexcept nogil:
    """encode 1 user into M x F representations with leave-target-out exclusion and optional cosine normalization"""

    cdef int64_t idx, h_start, h_end, pos_local
    cdef int32_t p, h, k_h, fld
    cdef int m
    cdef float val, w_h, w_total, norm_scale, max_logit, denom, lg, wm

    rfm_vec_zero(F, u_base_buf)

    if u_has_uf and u_uf_idx >= 0 and uf_indptr != NULL and uf_indices != NULL and uf_data != NULL:
        for idx in range(uf_indptr[u_uf_idx], uf_indptr[u_uf_idx + 1]):
            p = uf_indices[idx]
            val = uf_data[idx]
            if use_field_gates and uf_field != NULL and gamma_uf_ptr != NULL:
                fld = uf_field[p]
                rfm_vec_mul_axpy(F, val, &gamma_uf_ptr[<int64_t>fld * <int64_t>F], &v_uf_ptr[<int64_t>p * <int64_t>F], u_base_buf)
            else:
                rfm_vec_axpy(F, val, &v_uf_ptr[<int64_t>p * <int64_t>F], u_base_buf)

    if use_seq_anchor and seq_item >= 0 and seq_item != exclude_item and p_seq_ptr != NULL:
        if g_seq_ptr != NULL:
            rfm_vec_mul_axpy(F, 1.0, g_seq_ptr, &p_seq_ptr[<int64_t>seq_item * <int64_t>F], u_base_buf)
        else:
            rfm_vec_axpy(F, 1.0, &p_seq_ptr[<int64_t>seq_item * <int64_t>F], u_base_buf)

    for m in range(M):
        if use_v_u and u_v_idx >= 0 and v_u_ptr != NULL:
            rfm_vec_add(
                F,
                u_base_buf,
                &v_u_ptr[(<int64_t>u_v_idx * <int64_t>M + <int64_t>m) * <int64_t>F],
                &u_raw_out[<int64_t>m * <int64_t>F]
            )
        else:
            rfm_vec_copy(F, u_base_buf, &u_raw_out[<int64_t>m * <int64_t>F])

    if use_history and u_uh_idx >= 0 and uh_indptr != NULL and uh_indices != NULL and p_hist_ptr != NULL:
        h_start = uh_indptr[u_uh_idx]
        h_end = uh_indptr[u_uh_idx + 1]
        if h_end > h_start:
            if M == 1:
                rfm_vec_zero(F, u_hist_buf)
                w_total = 0.0
                for idx in range(h_start, h_end):
                    h = uh_indices[idx]
                    if h == exclude_item or h < 0:
                        continue
                    w_h = uh_weights[idx] if uh_weights != NULL else 1.0
                    if w_h <= 0.0:
                        continue
                    w_total += w_h
                    if use_event_gates and g_event_ptr != NULL:
                        k_h = uh_kinds[idx] if uh_kinds != NULL else 0
                        rfm_vec_mul_axpy(F, w_h, &g_event_ptr[<int64_t>k_h * <int64_t>F], &p_hist_ptr[<int64_t>h * <int64_t>F], u_hist_buf)
                    else:
                        rfm_vec_axpy(F, w_h, &p_hist_ptr[<int64_t>h * <int64_t>F], u_hist_buf)
                head_w_buf[0] = w_total
                if w_total > 1e-8:
                    norm_scale = 1.0 / <float>pow(<double>w_total, <double>history_exponent)
                    rfm_vec_axpy(F, norm_scale, u_hist_buf, u_raw_out)
            else:
                memset(u_hist_buf, 0, <size_t>(M * F) * sizeof(float))
                for m in range(M):
                    head_w_buf[m] = 0.0
                w_total = 0.0
                pos_local = 0
                for idx in range(h_start, h_end):
                    h = uh_indices[idx]
                    if h == exclude_item or h < 0:
                        if route_buf != NULL:
                            for m in range(M):
                                route_buf[pos_local * <int64_t>M + m] = 0.0
                        pos_local += 1
                        continue
                    w_h = uh_weights[idx] if uh_weights != NULL else 1.0
                    if w_h <= 0.0:
                        if route_buf != NULL:
                            for m in range(M):
                                route_buf[pos_local * <int64_t>M + m] = 0.0
                        pos_local += 1
                        continue
                    w_total += w_h
                    max_logit = -1e30
                    for m in range(M):
                        lg = rfm_dot(F, &p_hist_ptr[<int64_t>h * <int64_t>F], &c_proto_ptr[<int64_t>m * <int64_t>F]) * inv_route_temp
                        route_buf[pos_local * <int64_t>M + m] = lg
                        if lg > max_logit:
                            max_logit = lg
                    denom = 0.0
                    for m in range(M):
                        lg = <float>exp(<double>(route_buf[pos_local * <int64_t>M + m] - max_logit))
                        route_buf[pos_local * <int64_t>M + m] = lg
                        denom += lg
                    denom = 1.0 / max(denom, <float>1e-12)
                    k_h = uh_kinds[idx] if (use_event_gates and uh_kinds != NULL) else 0
                    for m in range(M):
                        wm = route_buf[pos_local * <int64_t>M + m] * denom
                        route_buf[pos_local * <int64_t>M + m] = wm
                        wm *= w_h
                        head_w_buf[m] += wm
                        if use_event_gates and g_event_ptr != NULL:
                            rfm_vec_mul_axpy(
                                F, wm,
                                &g_event_ptr[<int64_t>k_h * <int64_t>F],
                                &p_hist_ptr[<int64_t>h * <int64_t>F],
                                &u_hist_buf[<int64_t>m * <int64_t>F]
                            )
                        else:
                            rfm_vec_axpy(
                                F, wm,
                                &p_hist_ptr[<int64_t>h * <int64_t>F],
                                &u_hist_buf[<int64_t>m * <int64_t>F]
                            )
                    pos_local += 1

                if w_total > 1e-8:
                    for m in range(M):
                        if head_w_buf[m] > 1e-8:
                            norm_scale = 1.0 / <float>pow(<double>head_w_buf[m], <double>history_exponent)
                            rfm_vec_axpy(F, norm_scale, &u_hist_buf[<int64_t>m * <int64_t>F], &u_raw_out[<int64_t>m * <int64_t>F])

    if use_cosine:
        for m in range(M):
            u_inv_norm_out[m] = rfm_normalize_copy(
                F,
                &u_raw_out[<int64_t>m * <int64_t>F],
                &u_out[<int64_t>m * <int64_t>F],
                1e-6
            )
    else:
        memcpy(u_out, u_raw_out, <size_t>(M * F) * sizeof(float))
        for m in range(M):
            u_inv_norm_out[m] = 1.0


cdef struct EncodeUsersV050Ctx:
    const int32_t *u_v_map_ptr
    const int32_t *u_uf_map_ptr
    const int32_t *u_uh_map_ptr
    const int32_t *last_items_ptr
    const float *v_u_ptr
    const float *v_uf_ptr
    const int64_t *uf_indptr_ptr
    const int32_t *uf_indices_ptr
    const float *uf_data_ptr
    const int32_t *uf_field_ptr
    const float *gamma_uf_ptr
    const int64_t *uh_indptr_ptr
    const int32_t *uh_indices_ptr
    const float *uh_weights_ptr
    const int32_t *uh_kinds_ptr
    const float *p_hist_ptr
    const float *g_event_ptr
    const float *p_seq_ptr
    const float *g_seq_ptr
    const float *c_proto_ptr
    float *out_emb_ptr
    int64_t n_users
    int F
    int M
    int max_hist_len
    int use_v_u
    int x_uf_any
    int use_history
    int use_seq_anchor
    int use_field_gates
    int use_event_gates
    int use_cosine
    float history_exponent
    float inv_route_temp


cdef void _encode_users_v050_worker(int tid, int n_threads, void *raw_ctx) noexcept nogil:
    cdef EncodeUsersV050Ctx *ctx = <EncodeUsersV050Ctx *>raw_ctx
    cdef int64_t r_start = (ctx.n_users * tid) // n_threads
    cdef int64_t r_end = (ctx.n_users * (tid + 1)) // n_threads
    if r_start >= r_end:
        return

    cdef int F = ctx.F
    cdef int M = ctx.M
    cdef int max_h = max(1, ctx.max_hist_len)
    cdef size_t buf_floats = <size_t>((1 + 2 * M) * F + 2 * M + max_h * M)
    cdef float *buf = <float *>rfm_aligned_malloc(buf_floats * sizeof(float))
    if buf == NULL:
        return

    cdef float *u_base_buf = buf
    cdef float *u_hist_buf = &u_base_buf[F]
    cdef float *u_raw_buf = &u_hist_buf[M * F]
    cdef float *head_w_buf = &u_raw_buf[M * F]
    cdef float *u_inv_buf = &head_w_buf[M]
    cdef float *route_buf = &u_inv_buf[M]

    cdef int64_t r
    cdef int32_t u_v, u_uf, u_uh, seq_it
    cdef int has_uf_row

    for r in range(r_start, r_end):
        u_v = ctx.u_v_map_ptr[r] if ctx.u_v_map_ptr != NULL else <int32_t>r
        u_uf = ctx.u_uf_map_ptr[r] if ctx.u_uf_map_ptr != NULL else <int32_t>r
        u_uh = ctx.u_uh_map_ptr[r] if ctx.u_uh_map_ptr != NULL else <int32_t>r
        seq_it = ctx.last_items_ptr[r] if ctx.last_items_ptr != NULL else -1
        has_uf_row = 1 if (ctx.x_uf_any and u_uf >= 0 and ctx.uf_indptr_ptr[u_uf + 1] > ctx.uf_indptr_ptr[u_uf]) else 0

        _encode_single_user_v050(
            F, M,
            u_v, u_uf, u_uh, seq_it, -1,
            ctx.use_v_u if u_v >= 0 else 0,
            has_uf_row,
            ctx.use_history,
            ctx.use_seq_anchor,
            ctx.use_field_gates,
            ctx.use_event_gates,
            ctx.use_cosine,
            ctx.history_exponent,
            ctx.inv_route_temp,
            ctx.v_u_ptr,
            ctx.v_uf_ptr,
            ctx.uf_indptr_ptr,
            ctx.uf_indices_ptr,
            ctx.uf_data_ptr,
            ctx.uf_field_ptr,
            ctx.gamma_uf_ptr,
            ctx.uh_indptr_ptr,
            ctx.uh_indices_ptr,
            ctx.uh_weights_ptr,
            ctx.uh_kinds_ptr,
            ctx.p_hist_ptr,
            ctx.g_event_ptr,
            ctx.p_seq_ptr,
            ctx.g_seq_ptr,
            ctx.c_proto_ptr,
            u_base_buf,
            u_hist_buf,
            head_w_buf,
            route_buf,
            u_raw_buf,
            &ctx.out_emb_ptr[r * <int64_t>(M * F)],
            u_inv_buf
        )

    rfm_aligned_free(buf)


cdef struct FitV050Ctx:
    const int32_t *u_ptr
    const int32_t *i_ptr
    const float *sw_ptr
    const float *margin_sig_ptr
    const int32_t *prev_items_ptr
    rfm_perm_t perm
    const int64_t *ui_indptr_ptr
    const int32_t *ui_indices_ptr
    const int64_t *uf_indptr_ptr
    const int32_t *uf_indices_ptr
    const float *uf_data_ptr
    const int32_t *uf_field_ptr
    const int64_t *if_indptr_ptr
    const int32_t *if_indices_ptr
    const float *if_data_ptr
    const int32_t *if_field_ptr
    const int32_t *if_is_dyn_ptr
    const int64_t *uh_indptr_ptr
    const int32_t *uh_indices_ptr
    const float *uh_weights_ptr
    const int32_t *uh_kinds_ptr
    const float *item_log_freq_ptr
    const float *alias_prob_ptr
    const int32_t *alias_idx_ptr
    const float *log_q_ptr
    const float *mult_lut_ptr
    float *w_i_ptr
    float *w_if_ptr
    float *v_u_ptr
    float *v_i_ptr
    float *v_uf_ptr
    float *v_if_ptr
    float *p_hist_ptr
    float *p_seq_ptr
    float *g_event_ptr
    float *g_seq_ptr
    float *gamma_uf_ptr
    float *gamma_if_ptr
    float *a_gate_ptr
    float *b_gate_ptr
    float *c_proto_ptr
    float *b_eff_ptr
    float *v_raw_eff_ptr
    float *v_eff_ptr
    float *inv_norm_eff_ptr
    float *g_item_ptr
    float *s_item_ptr
    float *scratch_base
    size_t scratch_stride
    int32_t *neg_id_base
    size_t neg_id_stride
    float *gate_grad_base
    size_t gate_grad_stride
    double *ll_slots
    int64_t block_start
    int64_t block_end
    int64_t cursor
    int64_t grain
    int32_t I
    int F
    int M
    int K_event
    int G_uf
    int G_if
    int max_samples
    int max_hist_len
    int loss_mode
    int use_cosine
    int use_alias
    int x_uf_any
    int x_if_any
    int use_history
    int use_seq_anchor
    int use_field_gates
    int use_event_gates
    int use_freq_gate
    int verbose
    float eta
    float reg_a_keep
    float reg_b_keep
    float user_dropout
    float inv_temp
    float temperature
    float logq_weight
    float margin
    float weight_margin_scale
    float neg_weight
    float history_exponent
    float inv_route_temp
    float head_agg_temp
    float freq_gate_static_drop
    uint64_t base_seed
    uint64_t stream_epoch_base


cdef inline void _update_single_item_v050(
    FitV050Ctx *ctx,
    int32_t m,
    float bias_step,
    const float *grad_v_raw,
    float reg_weight,
    float *d_gamma_if,
    float *d_a_gate,
    float *d_b_gate,
    float *tmp_vec
) noexcept nogil:
    """apply regularized update to item m (w_i, v_i, optional sparse X_if, and thread-local gate accumulators)"""

    cdef int F = ctx.F
    cdef float reg_a_keep = 1.0 - reg_weight * (1.0 - ctx.reg_a_keep)
    cdef float reg_b_keep = 1.0 - reg_weight * (1.0 - ctx.reg_b_keep)
    cdef float old_w = ctx.w_i_ptr[m]
    cdef float new_w = clamp_f32(old_w * reg_a_keep + bias_step, -25.0, 25.0)
    ctx.w_i_ptr[m] = new_w
    ctx.b_eff_ptr[m] += (new_w - old_w)

    cdef float *vi_row = &ctx.v_i_ptr[<int64_t>m * <int64_t>F]
    cdef float *v_raw_row = &ctx.v_raw_eff_ptr[<int64_t>m * <int64_t>F]
    cdef float *v_eff_row = &ctx.v_eff_ptr[<int64_t>m * <int64_t>F]
    cdef const float *g_row = &ctx.g_item_ptr[<int64_t>m * <int64_t>F] if (ctx.use_freq_gate and ctx.g_item_ptr != NULL) else NULL
    cdef const float *s_row = &ctx.s_item_ptr[<int64_t>m * <int64_t>F] if (ctx.use_freq_gate and ctx.s_item_ptr != NULL) else NULL
    cdef int f, fld, is_dyn
    cdef int32_t q
    cdef int64_t idx
    cdef float old_vf, new_vf, g_mf, dg_dz, log_n, x_val
    cdef float *vif_row
    cdef const float *gam_row

    if g_row != NULL:
        log_n = ctx.item_log_freq_ptr[m]
        for f in range(F):
            g_mf = g_row[f]
            dg_dz = grad_v_raw[f] * g_mf * (1.0 - g_mf) * (vi_row[f] - ctx.freq_gate_static_drop * (s_row[f] if s_row != NULL else 0.0))
            d_a_gate[f] += dg_dz * log_n
            d_b_gate[f] += dg_dz

            old_vf = vi_row[f]
            new_vf = old_vf * reg_a_keep + g_mf * grad_v_raw[f]
            vi_row[f] = new_vf
            v_raw_row[f] += g_mf * (new_vf - old_vf)
    else:
        for f in range(F):
            old_vf = vi_row[f]
            new_vf = old_vf * reg_a_keep + grad_v_raw[f]
            vi_row[f] = new_vf
            v_raw_row[f] += (new_vf - old_vf)

    if ctx.use_cosine:
        ctx.inv_norm_eff_ptr[m] = rfm_normalize_copy(F, v_raw_row, v_eff_row, 1e-6)
    else:
        rfm_vec_copy(F, v_raw_row, v_eff_row)

    if ctx.x_if_any:
        for idx in range(ctx.if_indptr_ptr[m], ctx.if_indptr_ptr[m + 1]):
            q = ctx.if_indices_ptr[idx]
            x_val = ctx.if_data_ptr[idx]
            if x_val == 0.0:
                continue
            ctx.w_if_ptr[q] = clamp_f32(ctx.w_if_ptr[q] * reg_b_keep + bias_step * x_val, -15.0, 15.0)
            vif_row = &ctx.v_if_ptr[<int64_t>q * <int64_t>F]
            is_dyn = ctx.if_is_dyn_ptr[q] if (ctx.use_freq_gate and ctx.if_is_dyn_ptr != NULL) else 0

            if g_row != NULL and not is_dyn:
                for f in range(F):
                    tmp_vec[f] = x_val * (1.0 - ctx.freq_gate_static_drop * g_row[f]) * grad_v_raw[f]
            else:
                for f in range(F):
                    tmp_vec[f] = x_val * grad_v_raw[f]

            if ctx.use_field_gates and ctx.if_field_ptr != NULL and ctx.gamma_if_ptr != NULL:
                fld = ctx.if_field_ptr[q]
                gam_row = &ctx.gamma_if_ptr[<int64_t>fld * <int64_t>F]
                for f in range(F):
                    d_gamma_if[<int64_t>fld * <int64_t>F + f] += tmp_vec[f] * vif_row[f]
                    vif_row[f] = vif_row[f] * reg_b_keep + gam_row[f] * tmp_vec[f]
            else:
                rfm_vec_reg_axpy(F, reg_b_keep, 1.0, tmp_vec, &ctx.v_if_ptr[<int64_t>q * <int64_t>F])


cdef void _fit_v050_worker(int tid, int n_threads, void *raw_ctx) noexcept nogil:
    cdef FitV050Ctx *ctx = <FitV050Ctx *>raw_ctx
    cdef int64_t block_end = ctx.block_end
    if ctx.block_start >= block_end:
        return

    cdef int F = ctx.F
    cdef int M = ctx.M
    cdef int32_t I = ctx.I
    cdef uint32_t uI = <uint32_t>I
    cdef int max_samples = ctx.max_samples
    cdef int max_h = max(1, ctx.max_hist_len)
    cdef int loss_mode = ctx.loss_mode
    cdef int use_cosine = ctx.use_cosine
    cdef int use_alias = ctx.use_alias
    cdef int x_uf_any = ctx.x_uf_any
    cdef int x_if_any = ctx.x_if_any
    cdef int use_history = ctx.use_history
    cdef int use_seq_anchor = ctx.use_seq_anchor
    cdef int use_field_gates = ctx.use_field_gates
    cdef int use_event_gates = ctx.use_event_gates
    cdef float eta = ctx.eta
    cdef float reg_a_keep = ctx.reg_a_keep
    cdef float reg_b_keep = ctx.reg_b_keep
    cdef float user_dropout = ctx.user_dropout
    cdef float inv_temp = ctx.inv_temp
    cdef float logq_weight = ctx.logq_weight
    cdef int64_t grain = ctx.grain

    # Thread-local scratch layout
    cdef float *sc = &ctx.scratch_base[<size_t>tid * ctx.scratch_stride]
    cdef float *u_base_buf = sc
    cdef float *u_hist_buf = &u_base_buf[F]
    cdef float *u_raw_buf = &u_hist_buf[M * F]
    cdef float *u_out_buf = &u_raw_buf[M * F]
    cdef float *i_emb_buf = &u_out_buf[M * F]
    cdef float *neg_cent_buf = &i_emb_buf[F]
    cdef float *g_hat_buf = &neg_cent_buf[F]
    cdef float *grad_u_raw = &g_hat_buf[F]
    cdef float *grad_i_raw = &grad_u_raw[F]
    cdef float *tmp_vec = &grad_i_raw[F]
    cdef float *head_w_buf = &tmp_vec[F]
    cdef float *u_inv_buf = &head_w_buf[M]
    cdef float *head_scores = &u_inv_buf[M]
    cdef float *head_pi = &head_scores[M]
    cdef float *z_neg_buf = &head_pi[M]
    cdef float *route_buf = &z_neg_buf[max_samples]

    cdef int32_t *neg_ids = &ctx.neg_id_base[<size_t>tid * ctx.neg_id_stride]

    # Thread-local gate gradient slices
    cdef float *gg = &ctx.gate_grad_base[<size_t>tid * ctx.gate_grad_stride]
    memset(gg, 0, ctx.gate_grad_stride * sizeof(float))
    cdef float *d_g_event = gg
    cdef float *d_g_seq = &d_g_event[ctx.K_event * F]
    cdef float *d_gamma_uf = &d_g_seq[F]
    cdef float *d_gamma_if = &d_gamma_uf[ctx.G_uf * F]
    cdef float *d_a_gate = &d_gamma_if[ctx.G_if * F]
    cdef float *d_b_gate = &d_a_gate[F]
    cdef float *d_c_proto = &d_b_gate[F]

    cdef rfm_rng_t rng
    cdef int64_t c_start, c_end, r, row, n_seen_u, idx, h_start, h_end, pos_local
    cdef int32_t u, i, j, p, h, k_h, fld, seq_it, min_index
    cdef int sampled, k, m, m_best, f, tries, u_has_uf, u_has_hist, u_has_seq, use_v_u, n_feat_neg
    cdef float sw, margin_r, dot_val, s_m, ut_ui, ut_uj, pairwise_utility, min_pairwise_utility
    cdef float z_pos, z_max, e_pos, sum_neg, denom, g_pos, pk, base_step, emb_step, cos_grad_mult, bias_lr_scale, eff_inv
    cdef float norm_scale, w_h, wm, feat_neg_sum, feat_renorm, feat_pk_thresh, d_outer, scale_step, max_hs, hs_sum, dot_proto
    cdef const int32_t *u_seen_ptr
    cdef const float *u_win_ptr
    cdef const float *p_h_row
    cdef const float *ge_row
    cdef const float *vuf_row
    cdef const float *guf_row
    cdef double local_ll = 0.0

    bias_lr_scale = clamp_f32(2.0 * ctx.temperature, 0.15, 1.0) if use_cosine else 1.0
    cos_grad_mult = clamp_f32(0.24 * inv_temp, 1.0, 2.0) if use_cosine else 1.0

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
            if sw <= 0.0:
                continue

            n_seen_u = ctx.ui_indptr_ptr[u + 1] - ctx.ui_indptr_ptr[u]
            if n_seen_u >= I:
                continue
            u_seen_ptr = &ctx.ui_indices_ptr[ctx.ui_indptr_ptr[u]] if ctx.ui_indices_ptr != NULL else NULL

            margin_r = ctx.margin
            if ctx.weight_margin_scale != 0.0 and ctx.margin_sig_ptr != NULL:
                margin_r += ctx.weight_margin_scale * ctx.margin_sig_ptr[row]

            seq_it = ctx.prev_items_ptr[row] if (use_seq_anchor and ctx.prev_items_ptr != NULL) else -1
            if seq_it == i:
                seq_it = -1
            u_has_seq = 1 if seq_it >= 0 else 0
            u_has_uf = 1 if (x_uf_any and ctx.uf_indptr_ptr[u + 1] > ctx.uf_indptr_ptr[u]) else 0
            u_has_hist = 0
            if use_history and ctx.uh_indptr_ptr != NULL:
                h_start = ctx.uh_indptr_ptr[u]
                h_end = ctx.uh_indptr_ptr[u + 1]
                if h_end - h_start > 1 or (h_end - h_start == 1 and ctx.uh_indices_ptr[h_start] != i):
                    u_has_hist = 1

            if (u_has_uf or u_has_hist or u_has_seq) and user_dropout > 0.0:
                if user_dropout >= 1.0 or rfm_rng_float01(&rng) < user_dropout:
                    use_v_u = 0
                else:
                    use_v_u = 1
            else:
                use_v_u = 1

            _encode_single_user_v050(
                F, M,
                u, u, u, seq_it, i,
                use_v_u, u_has_uf, use_history, use_seq_anchor,
                use_field_gates, use_event_gates, use_cosine,
                ctx.history_exponent, ctx.inv_route_temp,
                ctx.v_u_ptr, ctx.v_uf_ptr,
                ctx.uf_indptr_ptr, ctx.uf_indices_ptr, ctx.uf_data_ptr,
                ctx.uf_field_ptr, ctx.gamma_uf_ptr,
                ctx.uh_indptr_ptr, ctx.uh_indices_ptr, ctx.uh_weights_ptr, ctx.uh_kinds_ptr,
                ctx.p_hist_ptr, ctx.g_event_ptr,
                ctx.p_seq_ptr, ctx.g_seq_ptr,
                ctx.c_proto_ptr,
                u_base_buf, u_hist_buf, head_w_buf, route_buf,
                u_raw_buf, u_out_buf, u_inv_buf
            )

            rfm_vec_copy(F, &ctx.v_eff_ptr[<int64_t>i * <int64_t>F], i_emb_buf)

            # Evaluate positive across M user interest heads
            if M == 1:
                m_best = 0
                head_pi[0] = 1.0
                dot_val = rfm_dot(F, u_out_buf, i_emb_buf)
                ut_ui = ctx.b_eff_ptr[i] + (inv_temp * dot_val if use_cosine else dot_val)
            else:
                m_best = 0
                max_hs = -1e30
                for m in range(M):
                    dot_val = rfm_dot(F, &u_out_buf[<int64_t>m * <int64_t>F], i_emb_buf)
                    s_m = ctx.b_eff_ptr[i] + (inv_temp * dot_val if use_cosine else dot_val)
                    head_scores[m] = s_m
                    if s_m > max_hs:
                        max_hs = s_m
                        m_best = m
                if ctx.head_agg_temp <= 0.0:
                    ut_ui = max_hs
                    for m in range(M):
                        head_pi[m] = 1.0 if m == m_best else 0.0
                else:
                    hs_sum = 0.0
                    for m in range(M):
                        head_pi[m] = <float>exp(<double>((head_scores[m] - max_hs) / ctx.head_agg_temp))
                        hs_sum += head_pi[m]
                    ut_ui = max_hs + ctx.head_agg_temp * <float>log(<double>max(hs_sum, <float>1e-12))
                    for m in range(M):
                        head_pi[m] /= max(hs_sum, <float>1e-12)

            if loss_mode <= 1:
                # BPR / WARP pairwise hinge-rank search with v0.5.0 encoder & gates
                min_index = -1
                min_pairwise_utility = 1e6

                for sampled in range(1, max_samples + 1):
                    tries = 0
                    while True:
                        j = rfm_alias_draw(&rng, uI, ctx.alias_prob_ptr, ctx.alias_idx_ptr) if use_alias else <int32_t>rfm_rng_bounded(&rng, uI)
                        tries += 1
                        if j != i and (tries >= 8 or u_seen_ptr == NULL or not item_observed(j, u_seen_ptr, n_seen_u)):
                            break

                    if M == 1:
                        dot_val = rfm_dot(F, u_out_buf, &ctx.v_eff_ptr[<int64_t>j * <int64_t>F])
                        ut_uj = ctx.b_eff_ptr[j] + (inv_temp * dot_val if use_cosine else dot_val)
                    else:
                        ut_uj = -1e30
                        for m in range(M):
                            dot_val = rfm_dot(F, &u_out_buf[<int64_t>m * <int64_t>F], &ctx.v_eff_ptr[<int64_t>j * <int64_t>F])
                            s_m = ctx.b_eff_ptr[j] + (inv_temp * dot_val if use_cosine else dot_val)
                            if s_m > ut_uj:
                                ut_uj = s_m

                    pairwise_utility = (ut_ui - margin_r) - (ut_uj - (logq_weight * (ctx.log_q_ptr[j] - ctx.log_q_ptr[i]) if logq_weight != 0.0 and ctx.log_q_ptr != NULL else 0.0))
                    if min_index == -1 or pairwise_utility < min_pairwise_utility:
                        min_index = j
                        min_pairwise_utility = pairwise_utility
                    if pairwise_utility < 1.0:
                        break

                j = min_index
                pairwise_utility = min_pairwise_utility
                if ctx.verbose:
                    local_ll += <double>safe_log_sigmoid(pairwise_utility)

                d_outer = safe_sigmoid_neg(pairwise_utility)
                scale_step = eta * sw * ctx.mult_lut_ptr[sampled] * d_outer
                rfm_vec_copy(F, &ctx.v_eff_ptr[<int64_t>j * <int64_t>F], neg_cent_buf)
                rfm_vec_sub(F, i_emb_buf, neg_cent_buf, g_hat_buf)
                rfm_vec_scale(F, scale_step, g_hat_buf)

                u_win_ptr = &u_out_buf[<int64_t>m_best * <int64_t>F]
                if use_cosine:
                    eff_inv = clamp_f32(u_inv_buf[m_best], 0.0, 3.0)
                    rfm_cosine_grad_project(F, eff_inv, u_win_ptr, g_hat_buf, grad_u_raw)

                    eff_inv = clamp_f32(ctx.inv_norm_eff_ptr[i], 0.0, 3.0)
                    for f in range(F):
                        tmp_vec[f] = scale_step * u_win_ptr[f]
                    rfm_cosine_grad_project(F, eff_inv, i_emb_buf, tmp_vec, grad_i_raw)
                    _update_single_item_v050(ctx, i, scale_step * bias_lr_scale, grad_i_raw, 1.0, d_gamma_if, d_a_gate, d_b_gate, tmp_vec)

                    eff_inv = clamp_f32(ctx.inv_norm_eff_ptr[j], 0.0, 3.0)
                    for f in range(F):
                        tmp_vec[f] = -scale_step * u_win_ptr[f]
                    rfm_cosine_grad_project(F, eff_inv, neg_cent_buf, tmp_vec, grad_i_raw)
                    _update_single_item_v050(ctx, j, -scale_step * bias_lr_scale, grad_i_raw, 1.0, d_gamma_if, d_a_gate, d_b_gate, tmp_vec)
                else:
                    rfm_vec_copy(F, g_hat_buf, grad_u_raw)
                    eff_inv = rfm_norm2(F, grad_u_raw)
                    if eff_inv > 0.85:
                        rfm_vec_scale(F, 0.85 / eff_inv, grad_u_raw)
                    eff_inv = rfm_norm2(F, u_win_ptr)
                    d_outer = scale_step if (scale_step * eff_inv <= 0.85) else (0.85 / max(eff_inv, <float>1e-6))
                    for f in range(F):
                        grad_i_raw[f] = d_outer * u_win_ptr[f]
                    _update_single_item_v050(ctx, i, clamp_f32(scale_step, -0.5, 0.5), grad_i_raw, 1.0, d_gamma_if, d_a_gate, d_b_gate, tmp_vec)
                    for f in range(F):
                        grad_i_raw[f] = -d_outer * u_win_ptr[f]
                    _update_single_item_v050(ctx, j, -clamp_f32(scale_step, -0.5, 0.5), grad_i_raw, 1.0, d_gamma_if, d_a_gate, d_b_gate, tmp_vec)

            else:
                # Cosine InfoNCE (loss_mode == 2) / Contrastive SimpleX (loss_mode == 3) with LogQ & Alias Negatives
                z_pos = ut_ui - margin_r
                if logq_weight != 0.0 and ctx.log_q_ptr != NULL:
                    z_pos -= logq_weight * ctx.log_q_ptr[i]
                z_max = z_pos

                for k in range(max_samples):
                    tries = 0
                    while True:
                        j = rfm_alias_draw(&rng, uI, ctx.alias_prob_ptr, ctx.alias_idx_ptr) if use_alias else <int32_t>rfm_rng_bounded(&rng, uI)
                        tries += 1
                        if j != i and (tries >= 8 or u_seen_ptr == NULL or not item_observed(j, u_seen_ptr, n_seen_u)):
                            break
                    neg_ids[k] = j
                    if M == 1:
                        dot_val = rfm_dot(F, u_out_buf, &ctx.v_eff_ptr[<int64_t>j * <int64_t>F])
                        ut_uj = ctx.b_eff_ptr[j] + (inv_temp * dot_val if use_cosine else dot_val)
                    else:
                        ut_uj = -1e30
                        for m in range(M):
                            dot_val = rfm_dot(F, &u_out_buf[<int64_t>m * <int64_t>F], &ctx.v_eff_ptr[<int64_t>j * <int64_t>F])
                            s_m = ctx.b_eff_ptr[j] + (inv_temp * dot_val if use_cosine else dot_val)
                            if s_m > ut_uj:
                                ut_uj = s_m
                    if logq_weight != 0.0 and ctx.log_q_ptr != NULL:
                        ut_uj -= logq_weight * ctx.log_q_ptr[j]
                    z_neg_buf[k] = ut_uj
                    if ut_uj > z_max:
                        z_max = ut_uj

                e_pos = <float>exp(<double>(z_pos - z_max))
                sum_neg = 0.0
                for k in range(max_samples):
                    z_neg_buf[k] = <float>exp(<double>(z_neg_buf[k] - z_max))
                    sum_neg += z_neg_buf[k]

                denom = e_pos + ctx.neg_weight * sum_neg
                if denom <= 1e-20:
                    continue

                if ctx.verbose:
                    local_ll += log(max(1e-20, <double>e_pos / <double>denom))

                g_pos = 0.0
                rfm_vec_zero(F, neg_cent_buf)
                feat_neg_sum = 0.0
                n_feat_neg = 0

                for k in range(max_samples):
                    pk = (ctx.neg_weight * z_neg_buf[k]) / denom
                    if loss_mode == 3:
                        # SimpleX smooth margin amplification on hard negatives
                        pk *= 1.25
                    z_neg_buf[k] = pk
                    g_pos += pk
                    j = neg_ids[k]
                    rfm_vec_axpy(F, pk, &ctx.v_eff_ptr[<int64_t>j * <int64_t>F], neg_cent_buf)

                if g_pos <= 1e-7:
                    continue

                base_step = eta * sw
                emb_step = base_step * cos_grad_mult
                for f in range(F):
                    g_hat_buf[f] = emb_step * (g_pos * i_emb_buf[f] - neg_cent_buf[f])

                u_win_ptr = &u_out_buf[<int64_t>m_best * <int64_t>F]
                if use_cosine:
                    eff_inv = clamp_f32(u_inv_buf[m_best], 0.0, 3.0)
                    rfm_cosine_grad_project(F, eff_inv, u_win_ptr, g_hat_buf, grad_u_raw)
                else:
                    rfm_vec_copy(F, g_hat_buf, grad_u_raw)

                # Positive item i update
                for f in range(F):
                    tmp_vec[f] = (emb_step * g_pos) * u_win_ptr[f]
                if use_cosine:
                    eff_inv = clamp_f32(ctx.inv_norm_eff_ptr[i], 0.0, 3.0)
                    rfm_cosine_grad_project(F, eff_inv, i_emb_buf, tmp_vec, grad_i_raw)
                else:
                    rfm_vec_copy(F, tmp_vec, grad_i_raw)
                _update_single_item_v050(
                    ctx, i, base_step * bias_lr_scale * g_pos, grad_i_raw, g_pos,
                    d_gamma_if, d_a_gate, d_b_gate, tmp_vec
                )

                # Negative item updates (exact InfoNCE softmax weights: sum_k pk == g_pos)
                for k in range(max_samples):
                    pk = z_neg_buf[k]
                    if pk < 1e-6:
                        continue
                    j = neg_ids[k]
                    scale_step = -emb_step * pk
                    for f in range(F):
                        tmp_vec[f] = scale_step * u_win_ptr[f]
                    if use_cosine:
                        eff_inv = clamp_f32(ctx.inv_norm_eff_ptr[j], 0.0, 3.0)
                        rfm_cosine_grad_project(F, eff_inv, &ctx.v_eff_ptr[<int64_t>j * <int64_t>F], tmp_vec, grad_i_raw)
                    else:
                        rfm_vec_copy(F, tmp_vec, grad_i_raw)
                    _update_single_item_v050(
                        ctx, j, -base_step * pk * bias_lr_scale, grad_i_raw, pk,
                        d_gamma_if, d_a_gate, d_b_gate, tmp_vec
                    )

            # ---- Shared User-Side Backprop from grad_u_raw on winning head m_best ----
            if use_v_u:
                rfm_vec_reg_axpy(
                    F, reg_a_keep, 1.0, grad_u_raw,
                    &ctx.v_u_ptr[(<int64_t>u * <int64_t>M + <int64_t>m_best) * <int64_t>F]
                )

            if u_has_uf:
                for idx in range(ctx.uf_indptr_ptr[u], ctx.uf_indptr_ptr[u + 1]):
                    p = ctx.uf_indices_ptr[idx]
                    w_h = ctx.uf_data_ptr[idx]
                    if w_h == 0.0:
                        continue
                    vuf_row = &ctx.v_uf_ptr[<int64_t>p * <int64_t>F]
                    if use_field_gates and ctx.uf_field_ptr != NULL and ctx.gamma_uf_ptr != NULL:
                        fld = ctx.uf_field_ptr[p]
                        guf_row = &ctx.gamma_uf_ptr[<int64_t>fld * <int64_t>F]
                        for f in range(F):
                            d_gamma_uf[<int64_t>fld * <int64_t>F + f] += w_h * grad_u_raw[f] * vuf_row[f]
                        rfm_vec_reg_mul_axpy(F, reg_b_keep, w_h, guf_row, grad_u_raw, &ctx.v_uf_ptr[<int64_t>p * <int64_t>F])
                    else:
                        rfm_vec_reg_axpy(F, reg_b_keep, w_h, grad_u_raw, &ctx.v_uf_ptr[<int64_t>p * <int64_t>F])

            if u_has_seq and ctx.p_seq_ptr != NULL:
                p_h_row = &ctx.p_seq_ptr[<int64_t>seq_it * <int64_t>F]
                if ctx.g_seq_ptr != NULL:
                    for f in range(F):
                        d_g_seq[f] += grad_u_raw[f] * p_h_row[f]
                    rfm_vec_reg_mul_axpy(F, reg_a_keep, 1.0, ctx.g_seq_ptr, grad_u_raw, &ctx.p_seq_ptr[<int64_t>seq_it * <int64_t>F])
                else:
                    rfm_vec_reg_axpy(F, reg_a_keep, 1.0, grad_u_raw, &ctx.p_seq_ptr[<int64_t>seq_it * <int64_t>F])

            if u_has_hist and head_w_buf[m_best] > 1e-8:
                norm_scale = 1.0 / <float>pow(<double>head_w_buf[m_best], <double>ctx.history_exponent)
                h_start = ctx.uh_indptr_ptr[u]
                h_end = ctx.uh_indptr_ptr[u + 1]
                pos_local = 0
                for idx in range(h_start, h_end):
                    h = ctx.uh_indices_ptr[idx]
                    if h == i or h < 0:
                        pos_local += 1
                        continue
                    w_h = ctx.uh_weights_ptr[idx] if ctx.uh_weights_ptr != NULL else 1.0
                    if w_h <= 0.0:
                        pos_local += 1
                        continue
                    wm = w_h * norm_scale
                    if M > 1:
                        wm *= route_buf[pos_local * <int64_t>M + m_best]
                    if wm > 1e-7:
                        p_h_row = &ctx.p_hist_ptr[<int64_t>h * <int64_t>F]
                        if use_event_gates and ctx.g_event_ptr != NULL:
                            k_h = ctx.uh_kinds_ptr[idx] if ctx.uh_kinds_ptr != NULL else 0
                            ge_row = &ctx.g_event_ptr[<int64_t>k_h * <int64_t>F]
                            for f in range(F):
                                d_g_event[<int64_t>k_h * <int64_t>F + f] += wm * grad_u_raw[f] * p_h_row[f]
                            rfm_vec_reg_mul_axpy(F, reg_a_keep, wm, ge_row, grad_u_raw, &ctx.p_hist_ptr[<int64_t>h * <int64_t>F])
                        else:
                            rfm_vec_reg_axpy(F, reg_a_keep, wm, grad_u_raw, &ctx.p_hist_ptr[<int64_t>h * <int64_t>F])

                        if M > 1 and ctx.c_proto_ptr != NULL:
                            # Pull winning prototype c_proto[m_best] gently toward matched source item p_hist[h]
                            dot_proto = wm * eta * sw * 0.05
                            rfm_vec_axpy(F, dot_proto, p_h_row, &d_c_proto[<int64_t>m_best * <int64_t>F])
                    pos_local += 1

        if n_threads <= 1:
            break

    if ctx.verbose:
        ctx.ll_slots[tid * 8] += local_ll


def _compute_effective_items_v050(
    float[::1] w_i,
    float[::1] w_if,
    float[:, ::1] v_i,
    float[:, ::1] v_if,
    if_indptr_arr,
    int32_t[::1] if_indices,
    float[::1] if_data,
    int x_if_any,
    int32_t[::1] if_field,
    int32_t[::1] if_is_dyn,
    float[:, ::1] gamma_if,
    float[::1] item_log_freq,
    float[::1] a_gate,
    float[::1] b_gate,
    bint use_field_gates,
    bint use_freq_gate,
    float freq_gate_static_drop,
    bint use_cosine,
    float out_scale=1.0,
    int n_threads=1
):
    """compute (item_bias, v_i_eff, v_hat_unit) in multi-threaded nogil C with field gates, freq gates, and cosine scaling"""

    cdef int32_t I = <int32_t>v_i.shape[0]
    cdef int F = <int>v_i.shape[1]

    b_eff = np.empty(I, dtype=np.float32)
    v_raw = np.empty((I, F), dtype=np.float32)
    v_eff = np.empty((I, F), dtype=np.float32)
    if I <= 0:
        return b_eff, v_eff, v_raw

    if_indptr_i64 = np.ascontiguousarray(if_indptr_arr, dtype=np.int64)
    cdef int64_t[::1] if_indptr = if_indptr_i64
    cdef float[::1] b_eff_mv = b_eff
    cdef float[:, ::1] v_raw_mv = v_raw
    cdef float[:, ::1] v_eff_mv = v_eff

    cdef RefreshItemsV050Ctx ctx
    memset(&ctx, 0, sizeof(RefreshItemsV050Ctx))
    ctx.w_i_ptr = &w_i[0]
    ctx.w_if_ptr = &w_if[0] if w_if.shape[0] > 0 else NULL
    ctx.v_i_ptr = &v_i[0, 0]
    ctx.v_if_ptr = &v_if[0, 0] if v_if.shape[0] > 0 else NULL
    ctx.if_indptr_ptr = &if_indptr[0]
    ctx.if_indices_ptr = &if_indices[0] if if_indices.shape[0] > 0 else NULL
    ctx.if_data_ptr = &if_data[0] if if_data.shape[0] > 0 else NULL
    ctx.if_field_ptr = &if_field[0] if if_field.shape[0] > 0 else NULL
    ctx.if_is_dyn_ptr = &if_is_dyn[0] if if_is_dyn.shape[0] > 0 else NULL
    ctx.gamma_if_ptr = &gamma_if[0, 0] if gamma_if.shape[0] > 0 else NULL
    ctx.item_log_freq_ptr = &item_log_freq[0] if item_log_freq.shape[0] > 0 else NULL
    ctx.a_gate_ptr = &a_gate[0] if a_gate.shape[0] > 0 else NULL
    ctx.b_gate_ptr = &b_gate[0] if b_gate.shape[0] > 0 else NULL
    ctx.b_eff_ptr = &b_eff_mv[0]
    ctx.v_raw_eff_ptr = &v_raw_mv[0, 0]
    ctx.v_eff_ptr = &v_eff_mv[0, 0]
    ctx.inv_norm_eff_ptr = NULL
    ctx.g_item_ptr = NULL
    ctx.s_item_ptr = NULL
    ctx.freq_gate_static_drop = freq_gate_static_drop
    ctx.out_scale = out_scale
    ctx.I = I
    ctx.F = F
    ctx.x_if_any = x_if_any
    ctx.use_field_gates = 1 if use_field_gates else 0
    ctx.use_freq_gate = 1 if use_freq_gate else 0
    ctx.use_cosine = 1 if use_cosine else 0

    cdef int eff_threads = n_threads if (n_threads > 1 and I * F >= 8192) else 1
    cdef rfm_pool_t *pool = NULL
    with nogil:
        pool = rfm_pool_create(eff_threads)
        rfm_pool_run(pool, _refresh_eff_items_v050_worker, &ctx)
        rfm_pool_destroy(pool)

    if use_cosine and out_scale != 1.0:
        v_unit = np.asarray(v_eff, dtype=np.float32) * np.float32(1.0 / out_scale)
    else:
        v_unit = v_eff
    return b_eff, v_eff, np.ascontiguousarray(v_unit, dtype=np.float32)


def _encode_users_v050(
    int32_t[::1] u_v_map,
    int32_t[::1] u_uf_map,
    int32_t[::1] u_uh_map,
    int32_t[::1] last_items,
    float[:, ::1] v_u,
    float[:, ::1] v_uf,
    uf_indptr_arr,
    int32_t[::1] uf_indices,
    float[::1] uf_data,
    int x_uf_any,
    int32_t[::1] uf_field,
    float[:, ::1] gamma_uf,
    uh_indptr_arr,
    int32_t[::1] uh_indices,
    float[::1] uh_weights,
    int32_t[::1] uh_kinds,
    float[:, ::1] p_hist,
    float[:, ::1] g_event,
    float[:, ::1] p_seq,
    float[::1] g_seq,
    float[:, ::1] c_proto,
    int M,
    int max_hist_len,
    bint use_v_u,
    bint use_history,
    bint use_seq_anchor,
    bint use_field_gates,
    bint use_event_gates,
    bint use_cosine,
    float history_exponent,
    float interest_temperature,
    int n_threads=1
):
    """batch encode users into (n_users, M * F) float32 representations in parallel nogil C"""

    cdef int64_t n_users = u_v_map.shape[0]
    cdef int F = <int>v_u.shape[1]
    out = np.zeros((n_users, M * F), dtype=np.float32)
    if n_users <= 0 or F <= 0:
        return out

    uf_indptr_i64 = np.ascontiguousarray(uf_indptr_arr, dtype=np.int64)
    uh_indptr_i64 = np.ascontiguousarray(uh_indptr_arr, dtype=np.int64)
    cdef int64_t[::1] uf_indptr = uf_indptr_i64
    cdef int64_t[::1] uh_indptr = uh_indptr_i64
    cdef float[:, ::1] out_mv = out

    cdef EncodeUsersV050Ctx ctx
    memset(&ctx, 0, sizeof(EncodeUsersV050Ctx))
    ctx.u_v_map_ptr = &u_v_map[0]
    ctx.u_uf_map_ptr = &u_uf_map[0] if u_uf_map.shape[0] > 0 else NULL
    ctx.u_uh_map_ptr = &u_uh_map[0] if u_uh_map.shape[0] > 0 else NULL
    ctx.last_items_ptr = &last_items[0] if last_items.shape[0] > 0 else NULL
    ctx.v_u_ptr = &v_u[0, 0] if v_u.shape[0] > 0 else NULL
    ctx.v_uf_ptr = &v_uf[0, 0] if v_uf.shape[0] > 0 else NULL
    ctx.uf_indptr_ptr = &uf_indptr[0]
    ctx.uf_indices_ptr = &uf_indices[0] if uf_indices.shape[0] > 0 else NULL
    ctx.uf_data_ptr = &uf_data[0] if uf_data.shape[0] > 0 else NULL
    ctx.uf_field_ptr = &uf_field[0] if uf_field.shape[0] > 0 else NULL
    ctx.gamma_uf_ptr = &gamma_uf[0, 0] if gamma_uf.shape[0] > 0 else NULL
    ctx.uh_indptr_ptr = &uh_indptr[0]
    ctx.uh_indices_ptr = &uh_indices[0] if uh_indices.shape[0] > 0 else NULL
    ctx.uh_weights_ptr = &uh_weights[0] if uh_weights.shape[0] > 0 else NULL
    ctx.uh_kinds_ptr = &uh_kinds[0] if uh_kinds.shape[0] > 0 else NULL
    ctx.p_hist_ptr = &p_hist[0, 0] if p_hist.shape[0] > 0 else NULL
    ctx.g_event_ptr = &g_event[0, 0] if g_event.shape[0] > 0 else NULL
    ctx.p_seq_ptr = &p_seq[0, 0] if p_seq.shape[0] > 0 else NULL
    ctx.g_seq_ptr = &g_seq[0] if g_seq.shape[0] > 0 else NULL
    ctx.c_proto_ptr = &c_proto[0, 0] if c_proto.shape[0] > 0 else NULL
    ctx.out_emb_ptr = &out_mv[0, 0]
    ctx.n_users = n_users
    ctx.F = F
    ctx.M = M
    ctx.max_hist_len = max(1, max_hist_len)
    ctx.use_v_u = 1 if use_v_u else 0
    ctx.x_uf_any = x_uf_any
    ctx.use_history = 1 if use_history else 0
    ctx.use_seq_anchor = 1 if use_seq_anchor else 0
    ctx.use_field_gates = 1 if use_field_gates else 0
    ctx.use_event_gates = 1 if use_event_gates else 0
    ctx.use_cosine = 1 if use_cosine else 0
    ctx.history_exponent = history_exponent
    ctx.inv_route_temp = 1.0 / max(interest_temperature, <float>1e-3)

    cdef int eff_threads = n_threads if (n_threads > 1 and n_users * M * F >= 16384) else 1
    cdef rfm_pool_t *pool = NULL
    with nogil:
        pool = rfm_pool_create(eff_threads)
        rfm_pool_run(pool, _encode_users_v050_worker, &ctx)
        rfm_pool_destroy(pool)
    return out


def _fit_v050(
    const int32_t[::1] u_idx,
    const int32_t[::1] i_idx,
    sample_weight,
    float[::1] margin_signal,
    int32_t[::1] prev_items,
    ui_indptr_arr,
    int32_t[::1] ui_indices,
    uf_indptr_arr,
    int32_t[::1] uf_indices,
    float[::1] uf_data,
    int x_uf_any,
    int32_t[::1] uf_field,
    if_indptr_arr,
    int32_t[::1] if_indices,
    float[::1] if_data,
    int x_if_any,
    int32_t[::1] if_field,
    int32_t[::1] if_is_dyn,
    uh_indptr_arr,
    int32_t[::1] uh_indices,
    float[::1] uh_weights,
    int32_t[::1] uh_kinds,
    float[::1] item_log_freq,
    float[::1] alias_prob,
    int32_t[::1] alias_idx,
    float[::1] log_q,
    float[::1] w_i,
    float[::1] w_if,
    float[:, ::1] v_u,
    float[:, ::1] v_i,
    float[:, ::1] v_uf,
    float[:, ::1] v_if,
    float[:, ::1] p_hist,
    float[:, ::1] p_seq,
    float[:, ::1] g_event,
    float[::1] g_seq,
    float[:, ::1] gamma_uf,
    float[:, ::1] gamma_if,
    float[::1] a_gate,
    float[::1] b_gate,
    float[:, ::1] c_proto,
    int M,
    int max_hist_len,
    int loss_mode,
    bint use_cosine,
    bint use_alias,
    bint use_history,
    bint use_seq_anchor,
    bint use_field_gates,
    bint use_event_gates,
    bint use_freq_gate,
    float alpha,
    float beta,
    float learning_rate,
    str learning_schedule,
    float learning_exponent,
    int max_samples,
    int epochs,
    bint verbose,
    float user_dropout,
    float temperature,
    float logq_weight,
    float margin,
    float weight_margin_scale,
    float neg_weight,
    float history_exponent,
    float interest_temperature,
    float head_agg_temperature,
    float freq_gate_static_drop,
    int seed=1492,
    int n_threads=1
):
    """multi-threaded nogil SGD trainer for RankFM v0.5.0"""

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
    cdef int F = <int>v_i.shape[1]
    if N == 0 or I <= 1 or epochs <= 0:
        return

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
    uh_indptr_i64 = np.ascontiguousarray(uh_indptr_arr, dtype=np.int64)

    cdef int64_t[::1] ui_indptr = ui_indptr_i64
    cdef int64_t[::1] uf_indptr = uf_indptr_i64
    cdef int64_t[::1] if_indptr = if_indptr_i64
    cdef int64_t[::1] uh_indptr = uh_indptr_i64

    cdef int eff_threads = n_threads if (n_threads > 1 and N >= 4096 and I >= 128) else 1
    if eff_threads > 1 and N // eff_threads < 256:
        eff_threads = max(1, <int>(N // 256))

    cdef int K_event = max(1, <int>g_event.shape[0])
    cdef int G_uf = max(1, <int>gamma_uf.shape[0])
    cdef int G_if = max(1, <int>gamma_if.shape[0])
    cdef int max_h = max(1, max_hist_len)
    cdef int eff_samples = max(1, max_samples)

    cdef int32_t U_cnt = <int32_t>max(<int64_t>1, <int64_t>uf_indptr_arr.shape[0] - 1)
    uf_fld_scale_arr = np.ones(G_uf, dtype=np.float32)
    if use_field_gates and uf_field.shape[0] > 0 and uf_data.shape[0] > 0:
        uf_fids = np.asarray(uf_field)[np.asarray(uf_indices)]
        uf_sq = np.bincount(uf_fids, weights=np.square(np.asarray(uf_data, dtype=np.float64)), minlength=G_uf) / float(U_cnt)
        uf_fld_scale_arr = (1.0 / np.maximum(uf_sq, 0.25)).astype(np.float32)
    if_fld_scale_arr = np.ones(G_if, dtype=np.float32)
    if use_field_gates and if_field.shape[0] > 0 and if_data.shape[0] > 0:
        if_fids = np.asarray(if_field)[np.asarray(if_indices)]
        if_sq = np.bincount(if_fids, weights=np.square(np.asarray(if_data, dtype=np.float64)), minlength=G_if) / float(I)
        if_fld_scale_arr = (1.0 / np.maximum(if_sq, 0.25)).astype(np.float32)
    cdef float[::1] uf_fld_scale = uf_fld_scale_arr
    cdef float[::1] if_fld_scale = if_fld_scale_arr

    cdef size_t raw_sc = <size_t>((7 + 3 * M) * F + 4 * M + eff_samples + max_h * M)
    cdef size_t scratch_stride = (raw_sc + 15u) & ~(<size_t>15u)
    cdef size_t neg_id_stride = (<size_t>eff_samples + 15u) & ~(<size_t>15u)
    cdef size_t raw_gg = <size_t>((K_event + 1 + G_uf + G_if + 2 + M) * F)
    cdef size_t gate_grad_stride = (raw_gg + 15u) & ~(<size_t>15u)

    cdef float *mult_lut = <float *>rfm_aligned_malloc(<size_t>(eff_samples + 1) * sizeof(float))
    cdef float *scratch_base = <float *>rfm_aligned_malloc(<size_t>eff_threads * scratch_stride * sizeof(float))
    cdef int32_t *neg_id_base = <int32_t *>rfm_aligned_malloc(<size_t>eff_threads * neg_id_stride * sizeof(int32_t))
    cdef float *gate_grad_base = <float *>rfm_aligned_malloc(<size_t>eff_threads * gate_grad_stride * sizeof(float))
    cdef double *ll_slots = <double *>calloc(<size_t>(eff_threads * 8), sizeof(double))

    cdef float *b_eff_alloc = <float *>rfm_aligned_malloc(<size_t>I * sizeof(float))
    cdef float *v_raw_eff_alloc = <float *>rfm_aligned_malloc(<size_t>I * <size_t>F * sizeof(float))
    cdef float *v_eff_alloc = <float *>rfm_aligned_malloc(<size_t>I * <size_t>F * sizeof(float))
    cdef float *inv_norm_alloc = <float *>rfm_aligned_malloc(<size_t>I * sizeof(float))
    cdef float *g_item_alloc = <float *>rfm_aligned_malloc(<size_t>I * <size_t>F * sizeof(float)) if use_freq_gate else NULL
    cdef float *s_item_alloc = <float *>rfm_aligned_malloc(<size_t>I * <size_t>F * sizeof(float)) if use_freq_gate else NULL
    cdef rfm_pool_t *pool = NULL

    if (
        mult_lut == NULL or scratch_base == NULL or neg_id_base == NULL or gate_grad_base == NULL
        or ll_slots == NULL or b_eff_alloc == NULL or v_raw_eff_alloc == NULL
        or v_eff_alloc == NULL or inv_norm_alloc == NULL
        or (use_freq_gate and (g_item_alloc == NULL or s_item_alloc == NULL))
    ):
        rfm_aligned_free(mult_lut)
        rfm_aligned_free(scratch_base)
        rfm_aligned_free(neg_id_base)
        rfm_aligned_free(gate_grad_base)
        free(ll_slots)
        rfm_aligned_free(b_eff_alloc)
        rfm_aligned_free(v_raw_eff_alloc)
        rfm_aligned_free(v_eff_alloc)
        rfm_aligned_free(inv_norm_alloc)
        rfm_aligned_free(g_item_alloc)
        rfm_aligned_free(s_item_alloc)
        raise MemoryError("failed to allocate RankFM v0.5.0 training buffers")

    cdef int s_idx
    cdef double denom_log_I = log(<double>I)
    mult_lut[0] = 1.0
    for s_idx in range(1, eff_samples + 1):
        mult_lut[s_idx] = <float>(log(<double>(I - 1) / <double>s_idx) / denom_log_I)

    cdef int64_t sync_interval
    if use_cosine and N <= 16384:
        sync_interval = max(<int64_t>128, N // 4)
    elif N <= 2048:
        sync_interval = max(<int64_t>64, N // 4)
    else:
        sync_interval = max(<int64_t>4096, min(N, <int64_t>16384))

    cdef RefreshItemsV050Ctx ref_ctx
    memset(&ref_ctx, 0, sizeof(RefreshItemsV050Ctx))
    ref_ctx.w_i_ptr = &w_i[0]
    ref_ctx.w_if_ptr = &w_if[0] if w_if.shape[0] > 0 else NULL
    ref_ctx.v_i_ptr = &v_i[0, 0]
    ref_ctx.v_if_ptr = &v_if[0, 0] if v_if.shape[0] > 0 else NULL
    ref_ctx.if_indptr_ptr = &if_indptr[0]
    ref_ctx.if_indices_ptr = &if_indices[0] if if_indices.shape[0] > 0 else NULL
    ref_ctx.if_data_ptr = &if_data[0] if if_data.shape[0] > 0 else NULL
    ref_ctx.if_field_ptr = &if_field[0] if if_field.shape[0] > 0 else NULL
    ref_ctx.if_is_dyn_ptr = &if_is_dyn[0] if if_is_dyn.shape[0] > 0 else NULL
    ref_ctx.gamma_if_ptr = &gamma_if[0, 0] if gamma_if.shape[0] > 0 else NULL
    ref_ctx.item_log_freq_ptr = &item_log_freq[0] if item_log_freq.shape[0] > 0 else NULL
    ref_ctx.a_gate_ptr = &a_gate[0] if a_gate.shape[0] > 0 else NULL
    ref_ctx.b_gate_ptr = &b_gate[0] if b_gate.shape[0] > 0 else NULL
    ref_ctx.b_eff_ptr = b_eff_alloc
    ref_ctx.v_raw_eff_ptr = v_raw_eff_alloc
    ref_ctx.v_eff_ptr = v_eff_alloc
    ref_ctx.inv_norm_eff_ptr = inv_norm_alloc
    ref_ctx.g_item_ptr = g_item_alloc
    ref_ctx.s_item_ptr = s_item_alloc
    ref_ctx.freq_gate_static_drop = freq_gate_static_drop
    ref_ctx.out_scale = 1.0
    ref_ctx.I = I
    ref_ctx.F = F
    ref_ctx.x_if_any = x_if_any
    ref_ctx.use_field_gates = 1 if use_field_gates else 0
    ref_ctx.use_freq_gate = 1 if use_freq_gate else 0
    ref_ctx.use_cosine = 1 if use_cosine else 0

    cdef FitV050Ctx ctx
    memset(&ctx, 0, sizeof(FitV050Ctx))
    ctx.u_ptr = &u_idx[0]
    ctx.i_ptr = &i_idx[0]
    ctx.sw_ptr = &sw_mv[0] if has_weights else NULL
    ctx.margin_sig_ptr = &margin_signal[0] if margin_signal.shape[0] > 0 else NULL
    ctx.prev_items_ptr = &prev_items[0] if prev_items.shape[0] > 0 else NULL
    ctx.ui_indptr_ptr = &ui_indptr[0]
    ctx.ui_indices_ptr = &ui_indices[0] if ui_indices.shape[0] > 0 else NULL
    ctx.uf_indptr_ptr = &uf_indptr[0]
    ctx.uf_indices_ptr = &uf_indices[0] if uf_indices.shape[0] > 0 else NULL
    ctx.uf_data_ptr = &uf_data[0] if uf_data.shape[0] > 0 else NULL
    ctx.uf_field_ptr = &uf_field[0] if uf_field.shape[0] > 0 else NULL
    ctx.if_indptr_ptr = &if_indptr[0]
    ctx.if_indices_ptr = &if_indices[0] if if_indices.shape[0] > 0 else NULL
    ctx.if_data_ptr = &if_data[0] if if_data.shape[0] > 0 else NULL
    ctx.if_field_ptr = &if_field[0] if if_field.shape[0] > 0 else NULL
    ctx.if_is_dyn_ptr = &if_is_dyn[0] if if_is_dyn.shape[0] > 0 else NULL
    ctx.uh_indptr_ptr = &uh_indptr[0]
    ctx.uh_indices_ptr = &uh_indices[0] if uh_indices.shape[0] > 0 else NULL
    ctx.uh_weights_ptr = &uh_weights[0] if uh_weights.shape[0] > 0 else NULL
    ctx.uh_kinds_ptr = &uh_kinds[0] if uh_kinds.shape[0] > 0 else NULL
    ctx.item_log_freq_ptr = &item_log_freq[0] if item_log_freq.shape[0] > 0 else NULL
    ctx.alias_prob_ptr = &alias_prob[0] if alias_prob.shape[0] > 0 else NULL
    ctx.alias_idx_ptr = &alias_idx[0] if alias_idx.shape[0] > 0 else NULL
    ctx.log_q_ptr = &log_q[0] if log_q.shape[0] > 0 else NULL
    ctx.mult_lut_ptr = mult_lut
    ctx.w_i_ptr = &w_i[0]
    ctx.w_if_ptr = &w_if[0] if w_if.shape[0] > 0 else NULL
    ctx.v_u_ptr = &v_u[0, 0] if v_u.shape[0] > 0 else NULL
    ctx.v_i_ptr = &v_i[0, 0]
    ctx.v_uf_ptr = &v_uf[0, 0] if v_uf.shape[0] > 0 else NULL
    ctx.v_if_ptr = &v_if[0, 0] if v_if.shape[0] > 0 else NULL
    ctx.p_hist_ptr = &p_hist[0, 0] if p_hist.shape[0] > 0 else NULL
    ctx.p_seq_ptr = &p_seq[0, 0] if p_seq.shape[0] > 0 else NULL
    ctx.g_event_ptr = &g_event[0, 0] if g_event.shape[0] > 0 else NULL
    ctx.g_seq_ptr = &g_seq[0] if g_seq.shape[0] > 0 else NULL
    ctx.gamma_uf_ptr = &gamma_uf[0, 0] if gamma_uf.shape[0] > 0 else NULL
    ctx.gamma_if_ptr = &gamma_if[0, 0] if gamma_if.shape[0] > 0 else NULL
    ctx.a_gate_ptr = &a_gate[0] if a_gate.shape[0] > 0 else NULL
    ctx.b_gate_ptr = &b_gate[0] if b_gate.shape[0] > 0 else NULL
    ctx.c_proto_ptr = &c_proto[0, 0] if c_proto.shape[0] > 0 else NULL
    ctx.b_eff_ptr = b_eff_alloc
    ctx.v_raw_eff_ptr = v_raw_eff_alloc
    ctx.v_eff_ptr = v_eff_alloc
    ctx.inv_norm_eff_ptr = inv_norm_alloc
    ctx.g_item_ptr = g_item_alloc
    ctx.s_item_ptr = s_item_alloc
    ctx.scratch_base = scratch_base
    ctx.scratch_stride = scratch_stride
    ctx.neg_id_base = neg_id_base
    ctx.neg_id_stride = neg_id_stride
    ctx.gate_grad_base = gate_grad_base
    ctx.gate_grad_stride = gate_grad_stride
    ctx.ll_slots = ll_slots
    ctx.I = I
    ctx.F = F
    ctx.M = M
    ctx.K_event = K_event
    ctx.G_uf = G_uf
    ctx.G_if = G_if
    ctx.max_samples = eff_samples
    ctx.max_hist_len = max_h
    ctx.loss_mode = loss_mode
    ctx.use_cosine = 1 if use_cosine else 0
    ctx.use_alias = 1 if (use_alias and alias_prob.shape[0] == I and alias_idx.shape[0] == I) else 0
    ctx.x_uf_any = x_uf_any
    ctx.x_if_any = x_if_any
    ctx.use_history = 1 if use_history else 0
    ctx.use_seq_anchor = 1 if use_seq_anchor else 0
    ctx.use_field_gates = 1 if use_field_gates else 0
    ctx.use_event_gates = 1 if use_event_gates else 0
    ctx.use_freq_gate = 1 if use_freq_gate else 0
    ctx.verbose = 1 if verbose else 0
    ctx.user_dropout = user_dropout
    ctx.temperature = max(temperature, <float>1e-3)
    ctx.inv_temp = 1.0 / ctx.temperature
    ctx.logq_weight = logq_weight
    ctx.margin = margin
    ctx.weight_margin_scale = weight_margin_scale
    ctx.neg_weight = neg_weight
    ctx.history_exponent = history_exponent
    ctx.inv_route_temp = 1.0 / max(interest_temperature, <float>1e-3)
    ctx.head_agg_temp = head_agg_temperature
    ctx.freq_gate_static_drop = freq_gate_static_drop
    ctx.base_seed = <uint64_t>(<uint32_t>seed)

    cdef int epoch, t_i, f_i, m1, m2
    cdef int64_t b_start, b_end, block_len, g_off
    cdef float eta, d_reg_a = 2.0 * alpha, d_reg_b = 2.0 * beta, gate_lr, barrier_reg, g_acc, dot_mm
    cdef float *gg_t
    cdef double log_likelihood
    cdef bint warned_large = False

    try:
        pool = rfm_pool_create(eff_threads)

        for epoch in range(epochs):
            if schedule_mode == 0:
                eta = learning_rate
            else:
                eta = learning_rate / <float>pow(<double>(epoch + 1), <double>learning_exponent)

            ctx.eta = eta
            ctx.reg_a_keep = max(<float>0.5, <float>1.0 - eta * d_reg_a)
            ctx.reg_b_keep = max(<float>0.5, <float>1.0 - eta * d_reg_b)

            if verbose:
                memset(ll_slots, 0, <size_t>(eff_threads * 8) * sizeof(double))

            with nogil:
                rfm_perm_init(&ctx.perm, <uint64_t>N, (ctx.base_seed + 0xC3C3C3C3ULL) ^ ((<uint64_t>(epoch + 1)) * 0x9E3779B97F4A7C15ULL))
                b_start = 0
                while b_start < N:
                    b_end = b_start + sync_interval
                    if b_end > N:
                        b_end = N
                    block_len = max(<int64_t>1, b_end - b_start)

                    rfm_pool_run(pool, _refresh_eff_items_v050_worker, &ref_ctx)

                    ctx.block_start = b_start
                    ctx.block_end = b_end
                    ctx.cursor = b_start
                    ctx.grain = max(<int64_t>64, min(<int64_t>256, block_len // max(<int64_t>1, <int64_t>(eff_threads * 4))))
                    ctx.stream_epoch_base = ((<uint64_t>(epoch + 1)) << 40) | (1ULL << 38)
                    rfm_pool_run(pool, _fit_v050_worker, &ctx)

                    # Deterministic barrier reduction of small shared gate & prototype tensors
                    gate_lr = 0.45 / <float>sqrt(max(1.0, <double>block_len / 16.0))
                    barrier_reg = eta * 0.02 * (<float>block_len / <float>max(<int64_t>1, N))

                    if ctx.use_event_gates and ctx.g_event_ptr != NULL:
                        for f_i in range(K_event * F):
                            g_acc = 0.0
                            for t_i in range(eff_threads):
                                gg_t = &gate_grad_base[<size_t>t_i * gate_grad_stride]
                                g_acc += gg_t[f_i]
                            ctx.g_event_ptr[f_i] = clamp_f32(
                                ctx.g_event_ptr[f_i] + clamp_f32(gate_lr * g_acc, -0.10, 0.10) - (eta * 0.008) * (ctx.g_event_ptr[f_i] - 1.0),
                                0.10, 2.8
                            )

                    g_off = <int64_t>(K_event * F)
                    if ctx.use_seq_anchor and ctx.g_seq_ptr != NULL:
                        for f_i in range(F):
                            g_acc = 0.0
                            for t_i in range(eff_threads):
                                gg_t = &gate_grad_base[<size_t>t_i * gate_grad_stride]
                                g_acc += gg_t[g_off + f_i]
                            ctx.g_seq_ptr[f_i] = clamp_f32(
                                ctx.g_seq_ptr[f_i] + clamp_f32(gate_lr * g_acc, -0.10, 0.10) - (eta * 0.008) * (ctx.g_seq_ptr[f_i] - 1.0),
                                0.10, 2.8
                            )

                    gate_lr = 8.0 / <float>max(<int64_t>16, block_len)
                    g_off += F
                    if ctx.use_field_gates:
                        if ctx.gamma_uf_ptr != NULL:
                            for f_i in range(G_uf * F):
                                g_acc = 0.0
                                for t_i in range(eff_threads):
                                    gg_t = &gate_grad_base[<size_t>t_i * gate_grad_stride]
                                    g_acc += gg_t[g_off + f_i]
                                g_acc *= uf_fld_scale[f_i // F]
                                ctx.gamma_uf_ptr[f_i] = clamp_f32(
                                    ctx.gamma_uf_ptr[f_i] + clamp_f32(gate_lr * g_acc, -0.05, 0.05) - barrier_reg * (ctx.gamma_uf_ptr[f_i] - 1.0),
                                    0.35, 1.85
                                )
                        g_off += G_uf * F
                        if ctx.gamma_if_ptr != NULL:
                            for f_i in range(G_if * F):
                                g_acc = 0.0
                                for t_i in range(eff_threads):
                                    gg_t = &gate_grad_base[<size_t>t_i * gate_grad_stride]
                                    g_acc += gg_t[g_off + f_i]
                                g_acc *= if_fld_scale[f_i // F]
                                ctx.gamma_if_ptr[f_i] = clamp_f32(
                                    ctx.gamma_if_ptr[f_i] + clamp_f32(gate_lr * g_acc, -0.05, 0.05) - barrier_reg * (ctx.gamma_if_ptr[f_i] - 1.0),
                                    0.35, 1.85
                                )
                        g_off += G_if * F
                    else:
                        g_off += (G_uf + G_if) * F

                    if ctx.use_freq_gate and ctx.a_gate_ptr != NULL and ctx.b_gate_ptr != NULL:
                        for f_i in range(F):
                            g_acc = 0.0
                            for t_i in range(eff_threads):
                                gg_t = &gate_grad_base[<size_t>t_i * gate_grad_stride]
                                g_acc += gg_t[g_off + f_i]
                            ctx.a_gate_ptr[f_i] = clamp_f32(
                                ctx.a_gate_ptr[f_i] * (1.0 - barrier_reg) + gate_lr * g_acc,
                                -4.0, 4.0
                            )
                        g_off += F
                        for f_i in range(F):
                            g_acc = 0.0
                            for t_i in range(eff_threads):
                                gg_t = &gate_grad_base[<size_t>t_i * gate_grad_stride]
                                g_acc += gg_t[g_off + f_i]
                            ctx.b_gate_ptr[f_i] = clamp_f32(
                                ctx.b_gate_ptr[f_i] * (1.0 - barrier_reg) + gate_lr * g_acc,
                                -4.0, 4.0
                            )
                        g_off += F
                    else:
                        g_off += 2 * F

                    if M > 1 and ctx.c_proto_ptr != NULL:
                        for f_i in range(M * F):
                            g_acc = 0.0
                            for t_i in range(eff_threads):
                                gg_t = &gate_grad_base[<size_t>t_i * gate_grad_stride]
                                g_acc += gg_t[g_off + f_i]
                            ctx.c_proto_ptr[f_i] += clamp_f32(g_acc / <float>max(<int64_t>1, block_len // 64), -0.25, 0.25)
                        # Normalize prototypes in-place & apply gentle orthogonality repulsion so heads stay distinct
                        for m1 in range(M):
                            rfm_normalize_inplace(F, &ctx.c_proto_ptr[m1 * F], 1e-6)
                        for m1 in range(M):
                            for m2 in range(m1 + 1, M):
                                dot_mm = rfm_dot(F, &ctx.c_proto_ptr[m1 * F], &ctx.c_proto_ptr[m2 * F])
                                if dot_mm > 0.05 or dot_mm < -0.05:
                                    rfm_vec_axpy(F, -0.18 * dot_mm, &ctx.c_proto_ptr[m2 * F], &ctx.c_proto_ptr[m1 * F])
                                    rfm_vec_axpy(F, -0.18 * dot_mm, &ctx.c_proto_ptr[m1 * F], &ctx.c_proto_ptr[m2 * F])
                        for m1 in range(M):
                            rfm_normalize_inplace(F, &ctx.c_proto_ptr[m1 * F], 1e-6)

                    b_start = b_end

            # same finite and magnitude checks as the classic path, raised explicitly so `python -O` cannot strip them
            large_label, large_value = check_weights(w_i, w_if, v_u, v_i, v_uf, v_if)
            for extra_label, extra_weights, in_use in (
                ("history embeddings [p_hist]", p_hist, use_history),
                ("sequential embeddings [p_seq]", p_seq, use_seq_anchor),
            ):
                if in_use and extra_weights.size > 0:
                    extra_value = max(float(np.max(extra_weights)), -float(np.min(extra_weights)))
                    if not np.isfinite(extra_value):
                        raise AssertionError("{} are not finite - try decreasing feature/sample_weight magnitudes".format(extra_label))
                    if extra_value > large_value:
                        large_label, large_value = extra_label, extra_value
            if not warned_large and large_value > LARGE_WEIGHT_THRESHOLD:
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
                print("\ntraining epoch:", epoch)
                print("objective:", round(log_likelihood, 2))

    finally:
        with nogil:
            rfm_pool_destroy(pool)
        rfm_aligned_free(mult_lut)
        rfm_aligned_free(scratch_base)
        rfm_aligned_free(neg_id_base)
        rfm_aligned_free(gate_grad_base)
        free(ll_slots)
        rfm_aligned_free(b_eff_alloc)
        rfm_aligned_free(v_raw_eff_alloc)
        rfm_aligned_free(v_eff_alloc)
        rfm_aligned_free(inv_norm_alloc)
        rfm_aligned_free(g_item_alloc)
        rfm_aligned_free(s_item_alloc)


cdef struct TopKV050ChunkCtx:
    const float *scores_ptr
    const float *bias_ptr
    const int32_t *u_seen_map_ptr
    const int32_t *u_hist_map_ptr
    const int64_t *seen_indptr_ptr
    const int32_t *seen_indices_ptr
    const int64_t *uh_indptr_ptr
    const int32_t *uh_indices_ptr
    const float *uh_weights_ptr
    const int64_t *s_indptr_ptr
    const int32_t *s_indices_ptr
    const float *s_data_ptr
    const int64_t *ctx_indptr_ptr
    const int32_t *ctx_indices_ptr
    const float *ctx_bonus_ptr
    const float *p_hist_ptr
    const float *v_unit_ptr
    int32_t *out_ptr
    int64_t B
    int64_t cursor
    int64_t grain
    int32_t I
    int32_t K
    int32_t C_heap
    int F
    int M
    int filter_previous
    int has_sparse_res
    int has_ctx
    int use_target_att
    float head_agg_temp
    float sparse_res_weight
    float target_att_weight
    float inv_sqrt_F


cdef void _select_topk_v050_worker(int tid, int n_threads, void *raw_ctx) noexcept nogil:
    cdef TopKV050ChunkCtx *ctx = <TopKV050ChunkCtx *>raw_ctx
    cdef int64_t B = ctx.B
    if B <= 0:
        return

    cdef int32_t I = ctx.I
    cdef int32_t K = ctx.K
    cdef int32_t C_heap = ctx.C_heap
    cdef int F = ctx.F
    cdef int M = ctx.M
    cdef int filter_previous = ctx.filter_previous
    cdef int need_row_buf = 1 if (M > 1 or ctx.has_sparse_res or ctx.has_ctx) else 0
    cdef int64_t grain = ctx.grain

    cdef float *heap_s = <float *>rfm_aligned_malloc(<size_t>C_heap * sizeof(float))
    cdef int32_t *heap_i = <int32_t *>rfm_aligned_malloc(<size_t>C_heap * sizeof(int32_t))
    cdef float *row_buf = <float *>rfm_aligned_malloc(<size_t>I * sizeof(float)) if need_row_buf else NULL
    cdef int64_t *seen_tag = <int64_t *>rfm_aligned_malloc(<size_t>I * sizeof(int64_t)) if filter_previous else NULL
    cdef float *att_logits = <float *>rfm_aligned_malloc(32 * sizeof(float)) if ctx.use_target_att else NULL

    if (
        heap_s == NULL or heap_i == NULL
        or (need_row_buf and row_buf == NULL)
        or (filter_previous and seen_tag == NULL)
        or (ctx.use_target_att and att_logits == NULL)
    ):
        rfm_aligned_free(heap_s)
        rfm_aligned_free(heap_i)
        rfm_aligned_free(row_buf)
        rfm_aligned_free(seen_tag)
        rfm_aligned_free(att_logits)
        return

    cdef int64_t c_start, c_end, b, idx, s_pos, h_lo, h_hi
    cdef int32_t u_seen, u_hist, i, h, h_size, k_idx, c_item, n_att, l_idx, best_p
    cdef int m
    cdef float s, max_s, lse_sum, w_h, max_att, att_den, att_val, best_s
    cdef const float *row_m0
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
            u_seen = ctx.u_seen_map_ptr[b] if ctx.u_seen_map_ptr != NULL else <int32_t>b
            u_hist = ctx.u_hist_map_ptr[b] if ctx.u_hist_map_ptr != NULL else u_seen
            row_out = &ctx.out_ptr[b * <int64_t>K]

            if filter_previous and ctx.seen_indices_ptr != NULL and u_seen >= 0:
                for idx in range(ctx.seen_indptr_ptr[u_seen], ctx.seen_indptr_ptr[u_seen + 1]):
                    seen_tag[ctx.seen_indices_ptr[idx]] = b

            if need_row_buf:
                row_m0 = &ctx.scores_ptr[(b * <int64_t>M) * <int64_t>I]
                if M == 1:
                    for i in range(I):
                        row_buf[i] = row_m0[i] + ctx.bias_ptr[i]
                elif ctx.head_agg_temp <= 0.0:
                    for i in range(I):
                        max_s = row_m0[i]
                        for m in range(1, M):
                            s = ctx.scores_ptr[(b * <int64_t>M + m) * <int64_t>I + i]
                            if s > max_s:
                                max_s = s
                        row_buf[i] = max_s + ctx.bias_ptr[i]
                else:
                    for i in range(I):
                        max_s = row_m0[i]
                        for m in range(1, M):
                            s = ctx.scores_ptr[(b * <int64_t>M + m) * <int64_t>I + i]
                            if s > max_s:
                                max_s = s
                        lse_sum = 0.0
                        for m in range(M):
                            s = ctx.scores_ptr[(b * <int64_t>M + m) * <int64_t>I + i]
                            lse_sum += <float>exp(<double>((s - max_s) / ctx.head_agg_temp))
                        row_buf[i] = max_s + ctx.head_agg_temp * <float>log(<double>max(lse_sum, <float>1e-12)) + ctx.bias_ptr[i]

                if ctx.has_sparse_res and u_hist >= 0 and ctx.uh_indptr_ptr != NULL and ctx.s_indptr_ptr != NULL:
                    for idx in range(ctx.uh_indptr_ptr[u_hist], ctx.uh_indptr_ptr[u_hist + 1]):
                        h = ctx.uh_indices_ptr[idx]
                        if h < 0 or h >= I:
                            continue
                        w_h = (ctx.uh_weights_ptr[idx] if ctx.uh_weights_ptr != NULL else 1.0) * ctx.sparse_res_weight
                        for s_pos in range(ctx.s_indptr_ptr[h], ctx.s_indptr_ptr[h + 1]):
                            row_buf[ctx.s_indices_ptr[s_pos]] += w_h * ctx.s_data_ptr[s_pos]

                if ctx.has_ctx and ctx.ctx_indptr_ptr != NULL and ctx.ctx_indices_ptr != NULL:
                    for s_pos in range(ctx.ctx_indptr_ptr[b], ctx.ctx_indptr_ptr[b + 1]):
                        row_buf[ctx.ctx_indices_ptr[s_pos]] += ctx.ctx_bonus_ptr[s_pos]
            else:
                row_m0 = &ctx.scores_ptr[b * <int64_t>I]

            h_size = 0
            for i in range(I):
                if filter_previous and seen_tag[i] == b:
                    continue
                s = row_buf[i] if need_row_buf else (row_m0[i] + ctx.bias_ptr[i])
                if h_size < C_heap:
                    heap_s[h_size] = s
                    heap_i[h_size] = i
                    heap_sift_up(heap_s, heap_i, h_size)
                    h_size += 1
                elif s > heap_s[0]:
                    heap_s[0] = s
                    heap_i[0] = i
                    heap_sift_down(heap_s, heap_i, C_heap, 0)

            # Optional DIN-lite target-aware attention re-scoring over the C_heap top candidates
            if ctx.use_target_att and h_size > 0 and u_hist >= 0 and ctx.uh_indptr_ptr != NULL:
                h_lo = ctx.uh_indptr_ptr[u_hist]
                h_hi = ctx.uh_indptr_ptr[u_hist + 1]
                n_att = <int32_t>min(<int64_t>16, h_hi - h_lo)
                if n_att > 0:
                    for k_idx in range(h_size):
                        c_item = heap_i[k_idx]
                        max_att = -1e30
                        for l_idx in range(n_att):
                            h = ctx.uh_indices_ptr[h_lo + l_idx]
                            s = rfm_dot(
                                F,
                                &ctx.p_hist_ptr[<int64_t>h * <int64_t>F],
                                &ctx.v_unit_ptr[<int64_t>c_item * <int64_t>F]
                            ) * ctx.inv_sqrt_F
                            att_logits[l_idx] = s
                            if s > max_att:
                                max_att = s
                        att_den = 0.0
                        att_val = 0.0
                        for l_idx in range(n_att):
                            w_h = <float>exp(<double>(att_logits[l_idx] - max_att))
                            if ctx.uh_weights_ptr != NULL:
                                w_h *= ctx.uh_weights_ptr[h_lo + l_idx]
                            att_den += w_h
                            att_val += w_h * att_logits[l_idx]
                        if att_den > 1e-12:
                            heap_s[k_idx] += ctx.target_att_weight * (att_val / att_den)

                    # Extract sorted top-K directly from the C_heap candidate pool
                    for k_idx in range(min(K, h_size)):
                        best_p = k_idx
                        best_s = heap_s[k_idx]
                        for l_idx in range(k_idx + 1, h_size):
                            if heap_s[l_idx] > best_s:
                                best_s = heap_s[l_idx]
                                best_p = l_idx
                        if best_p != k_idx:
                            heap_s[best_p] = heap_s[k_idx]
                            heap_s[k_idx] = best_s
                            c_item = heap_i[best_p]
                            heap_i[best_p] = heap_i[k_idx]
                            heap_i[k_idx] = c_item
                        row_out[k_idx] = heap_i[k_idx]
                    for k_idx in range(min(K, h_size), K):
                        row_out[k_idx] = -1
                    continue

            # Standard heap extraction (supports both C_heap == K and C_heap > K when user had 0 history)
            while h_size > K:
                h_size -= 1
                heap_s[0] = heap_s[h_size]
                heap_i[0] = heap_i[h_size]
                heap_sift_down(heap_s, heap_i, h_size, 0)

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

    rfm_aligned_free(heap_s)
    rfm_aligned_free(heap_i)
    rfm_aligned_free(row_buf)
    rfm_aligned_free(seen_tag)
    rfm_aligned_free(att_logits)


def _select_topk_v050_chunk(
    float[:, ::1] scores_chunk,
    float[::1] item_bias,
    int32_t[::1] u_seen_map,
    int32_t[::1] u_hist_map,
    seen_indptr_arr,
    int32_t[::1] seen_indices,
    uh_indptr_arr,
    int32_t[::1] uh_indices,
    float[::1] uh_weights,
    s_indptr_arr,
    int32_t[::1] s_indices,
    float[::1] s_data,
    ctx_indptr_arr,
    int32_t[::1] ctx_indices,
    float[::1] ctx_bonus,
    float[:, ::1] p_hist,
    float[:, ::1] v_unit,
    int M,
    int n_items,
    bint filter_previous,
    float head_agg_temperature,
    float sparse_residual_weight,
    int target_attention_topk,
    float target_attention_weight,
    int32_t[:, ::1] out_items,
    int n_threads=1
):
    """parallel nogil top-K selector supporting M-head aggregation, sparse residual S, pairwise context, and DIN-lite attention"""

    cdef int64_t B = out_items.shape[0]
    cdef int32_t I = <int32_t>scores_chunk.shape[1]
    cdef int32_t K = <int32_t>n_items
    if B == 0 or K <= 0 or I <= 0:
        return

    cdef int F = <int>v_unit.shape[1] if v_unit.shape[0] > 0 else 1
    cdef int has_sparse_res = 1 if (sparse_residual_weight != 0.0 and s_indices.shape[0] > 0) else 0
    cdef int has_ctx = 1 if (ctx_indices.shape[0] > 0) else 0
    cdef int use_target_att = 1 if (target_attention_topk > 0 and target_attention_weight != 0.0 and p_hist.shape[0] > 0 and v_unit.shape[0] == I) else 0
    cdef int32_t C_heap = max(K, <int32_t>target_attention_topk) if use_target_att else K
    if C_heap > I:
        C_heap = I

    seen_indptr_i64 = np.ascontiguousarray(seen_indptr_arr, dtype=np.int64)
    uh_indptr_i64 = np.ascontiguousarray(uh_indptr_arr, dtype=np.int64)
    s_indptr_i64 = np.ascontiguousarray(s_indptr_arr, dtype=np.int64)
    ctx_indptr_i64 = np.ascontiguousarray(ctx_indptr_arr, dtype=np.int64)

    cdef int64_t[::1] seen_indptr = seen_indptr_i64
    cdef int64_t[::1] uh_indptr = uh_indptr_i64
    cdef int64_t[::1] s_indptr = s_indptr_i64
    cdef int64_t[::1] ctx_indptr = ctx_indptr_i64

    cdef int eff_threads = n_threads if (n_threads > 1 and B >= 32) else 1
    if eff_threads > B:
        eff_threads = <int>B

    cdef TopKV050ChunkCtx ctx
    memset(&ctx, 0, sizeof(TopKV050ChunkCtx))
    ctx.scores_ptr = &scores_chunk[0, 0]
    ctx.bias_ptr = &item_bias[0]
    ctx.u_seen_map_ptr = &u_seen_map[0] if u_seen_map.shape[0] > 0 else NULL
    ctx.u_hist_map_ptr = &u_hist_map[0] if u_hist_map.shape[0] > 0 else NULL
    ctx.seen_indptr_ptr = &seen_indptr[0] if filter_previous else NULL
    ctx.seen_indices_ptr = &seen_indices[0] if (filter_previous and seen_indices.shape[0] > 0) else NULL
    ctx.uh_indptr_ptr = &uh_indptr[0] if uh_indptr.shape[0] > 1 else NULL
    ctx.uh_indices_ptr = &uh_indices[0] if uh_indices.shape[0] > 0 else NULL
    ctx.uh_weights_ptr = &uh_weights[0] if uh_weights.shape[0] > 0 else NULL
    ctx.s_indptr_ptr = &s_indptr[0] if has_sparse_res else NULL
    ctx.s_indices_ptr = &s_indices[0] if has_sparse_res else NULL
    ctx.s_data_ptr = &s_data[0] if has_sparse_res else NULL
    ctx.ctx_indptr_ptr = &ctx_indptr[0] if has_ctx else NULL
    ctx.ctx_indices_ptr = &ctx_indices[0] if has_ctx else NULL
    ctx.ctx_bonus_ptr = &ctx_bonus[0] if has_ctx else NULL
    ctx.p_hist_ptr = &p_hist[0, 0] if p_hist.shape[0] > 0 else NULL
    ctx.v_unit_ptr = &v_unit[0, 0] if v_unit.shape[0] > 0 else NULL
    ctx.out_ptr = &out_items[0, 0]
    ctx.B = B
    ctx.cursor = 0
    ctx.grain = max(<int64_t>8, min(<int64_t>64, B // max(<int64_t>1, <int64_t>(eff_threads * 4))))
    ctx.I = I
    ctx.K = K
    ctx.C_heap = C_heap
    ctx.F = F
    ctx.M = M
    ctx.filter_previous = 1 if filter_previous else 0
    ctx.has_sparse_res = has_sparse_res
    ctx.has_ctx = has_ctx
    ctx.use_target_att = use_target_att
    ctx.head_agg_temp = head_agg_temperature
    ctx.sparse_res_weight = sparse_residual_weight
    ctx.target_att_weight = target_attention_weight
    ctx.inv_sqrt_F = 1.0 / <float>sqrt(<double>max(1, F))

    cdef rfm_pool_t *pool = NULL
    with nogil:
        pool = rfm_pool_create(eff_threads)
        rfm_pool_run(pool, _select_topk_v050_worker, &ctx)
        rfm_pool_destroy(pool)


ctypedef struct PairAccumCtx:
    const int64_t *u_indptr_ptr
    const int32_t *items_ptr
    const float *rec_ptr
    float *co_ptr
    float *fwd_ptr
    float *fwd21_ptr
    float *fwd30_ptr
    float *tfwd_ptr
    int64_t U
    int32_t I


cdef void _pair_accum_worker(int tid, int n_threads, void *raw_ctx) noexcept nogil:
    cdef PairAccumCtx *ctx = <PairAccumCtx *>raw_ctx
    cdef int64_t U = ctx.U
    cdef int64_t I = <int64_t>ctx.I
    cdef int64_t u, lo, hi, p, q, row_off, diff_pq
    cdef int32_t src, dst
    cdef float rp, rq, gap, w_dir, step_w, rec_dec

    for u in range(U):
        lo = ctx.u_indptr_ptr[u]
        hi = ctx.u_indptr_ptr[u + 1]
        if hi <= lo:
            continue
        for p in range(lo, hi):
            src = ctx.items_ptr[p]
            if src < 0 or src >= ctx.I or (src % n_threads) != tid:
                continue
            row_off = (<int64_t>src) * I
            ctx.co_ptr[row_off + src] += 1.0
            if hi - lo < 2:
                continue
            rp = ctx.rec_ptr[p]
            for q in range(lo, hi):
                if p == q:
                    continue
                dst = ctx.items_ptr[q]
                if dst < 0 or dst >= ctx.I or dst == src:
                    continue
                ctx.co_ptr[row_off + dst] += 1.0
                rq = ctx.rec_ptr[q]
                if rp > rq:
                    w_dir = 1.0
                    gap = rp - rq
                elif rp == rq:
                    w_dir = 0.5
                    gap = 0.0
                else:
                    continue

                ctx.fwd_ptr[row_off + dst] += w_dir
                if gap <= 21.0:
                    ctx.fwd21_ptr[row_off + dst] += w_dir
                if rq <= 30.0:
                    ctx.fwd30_ptr[row_off + dst] += w_dir

                diff_pq = q - p if q > p else p - q
                if (rp > rq and q > p and diff_pq <= 3) or (rp == rq and diff_pq <= 3):
                    if diff_pq == 1:
                        step_w = 1.0
                    elif diff_pq == 2:
                        step_w = 0.55
                    else:
                        step_w = 0.30
                    rec_dec = 0.35 + 0.65 * <float>pow(0.5, <double>rq / 28.0)
                    ctx.tfwd_ptr[row_off + dst] += w_dir * step_w * rec_dec


def _accumulate_pair_counts_omp(
    u_indptr_arr,
    int32_t[::1] item_idx,
    float[::1] rec_days,
    int n_items,
    int n_threads=1
):
    """Parallel lock-free Cython pair-count accumulator over user-capped chronological histories."""
    u_indptr_i64 = np.ascontiguousarray(u_indptr_arr, dtype=np.int64)
    cdef int64_t[::1] u_indptr = u_indptr_i64
    cdef int64_t U = u_indptr.shape[0] - 1 if u_indptr.shape[0] > 0 else 0
    cdef int32_t I = <int32_t>n_items

    co_arr = np.zeros((I, I), dtype=np.float32)
    fwd_arr = np.zeros((I, I), dtype=np.float32)
    fwd21_arr = np.zeros((I, I), dtype=np.float32)
    fwd30_arr = np.zeros((I, I), dtype=np.float32)
    tfwd_arr = np.zeros((I, I), dtype=np.float32)

    if U <= 0 or I <= 0 or item_idx.shape[0] == 0:
        return co_arr, fwd_arr, fwd21_arr, fwd30_arr, tfwd_arr

    cdef float[:, ::1] co_mv = co_arr
    cdef float[:, ::1] fwd_mv = fwd_arr
    cdef float[:, ::1] fwd21_mv = fwd21_arr
    cdef float[:, ::1] fwd30_mv = fwd30_arr
    cdef float[:, ::1] tfwd_mv = tfwd_arr

    cdef int eff_threads = n_threads if (n_threads > 1 and I >= 16 and U >= 64) else 1
    if eff_threads > I:
        eff_threads = I

    cdef PairAccumCtx ctx
    memset(&ctx, 0, sizeof(PairAccumCtx))
    ctx.u_indptr_ptr = &u_indptr[0]
    ctx.items_ptr = &item_idx[0]
    ctx.rec_ptr = &rec_days[0]
    ctx.co_ptr = &co_mv[0, 0]
    ctx.fwd_ptr = &fwd_mv[0, 0]
    ctx.fwd21_ptr = &fwd21_mv[0, 0]
    ctx.fwd30_ptr = &fwd30_mv[0, 0]
    ctx.tfwd_ptr = &tfwd_mv[0, 0]
    ctx.U = U
    ctx.I = I

    cdef rfm_pool_t *pool = NULL
    with nogil:
        pool = rfm_pool_create(eff_threads)
        rfm_pool_run(pool, _pair_accum_worker, &ctx)
        rfm_pool_destroy(pool)

    return co_arr, fwd_arr, fwd21_arr, fwd30_arr, tfwd_arr

