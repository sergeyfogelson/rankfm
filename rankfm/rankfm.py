"""
rankfm main modeling class
"""

import contextlib
import math
import os
from collections import namedtuple

import numpy as np
import pandas as pd

from rankfm._rankfm import (
    _compute_effective_items_v050,
    _csr_dense_matmul,
    _csr_matvec,
    _encode_users_v050,
    _fit_csr,
    _fit_v050,
    _fold_in,
    _locate_pairs_in_csr,
    _predict,
    _select_topk_chunk,
    _select_topk_v050_chunk,
    dense_to_csr,
)
from rankfm.utils import get_data, require

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # optional: without it the BLAS library keeps its own thread count
    threadpool_limits = None


# pre-extracted feature CSR passed between internal calls so repeated inference batches skip re-extraction
_FeatureCSR = namedtuple('_FeatureCSR', ['ids', 'indptr', 'indices', 'data', 'n_cols', 'columns'])


def _available_cpus():
    """count the CPUs this process may actually use

    os.cpu_count() reports every core on the host. inside a container (GKE, Cloud Run, Vertex AI) the process is
    usually restricted to fewer by CPU affinity or a cgroup CPU quota, and starting one thread per host core then
    oversubscribes the quota and gets throttled by the kernel scheduler.
    """

    if hasattr(os, 'sched_getaffinity'):
        cpus = len(os.sched_getaffinity(0))
    else:
        cpus = os.cpu_count() or 1

    quota = None
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:  # cgroup v2
            max_us, period_us = f.read().split()
        if max_us != 'max':
            quota = int(max_us) / int(period_us)
    except (OSError, ValueError):
        try:
            with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as f:  # cgroup v1
                max_us = int(f.read())
            with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as f:
                period_us = int(f.read())
            if max_us > 0 and period_us > 0:
                quota = max_us / period_us
        except (OSError, ValueError):
            pass

    if quota is not None:
        cpus = min(cpus, math.ceil(quota))
    return max(1, cpus)


def _blas_thread_cap():
    """context manager keeping BLAS within the CPUs available to this process

    OpenBLAS and MKL size their thread pools from the host core count, so inside a CPU-limited container
    the matrix product behind recommend() oversubscribes exactly as an uncapped [n_jobs] would
    """

    if threadpool_limits is None:
        return contextlib.nullcontext()
    return threadpool_limits(limits=_available_cpus(), user_api='blas')


def _append_zero_rows(arr, n_extra):
    """return [arr] followed by [n_extra] zero rows, reusing spare capacity when [arr] is the head of a larger buffer

    growing by reallocation copies the whole table every time, so a stream of small fit_partial() calls that each
    add a few users would pay for the full user table on every call. spare capacity makes that cost occasional.
    """

    n_rows = len(arr)
    base = arr.base
    reusable = isinstance(base, np.ndarray) and base.ndim == 2 and base.dtype == arr.dtype and base.flags.c_contiguous
    if reusable:
        reusable = base.shape[1] == arr.shape[1] and len(base) >= n_rows + n_extra
    if reusable:
        reusable = base.__array_interface__['data'][0] == arr.__array_interface__['data'][0]  # [arr] starts the buffer
    if not reusable:
        base = np.zeros((n_rows + n_extra + max(1024, n_rows // 4), arr.shape[1]), dtype=arr.dtype)
        base[:n_rows] = arr

    out = base[:n_rows + n_extra]
    out[n_rows:] = 0
    return out


def _is_sorted_unique_pairs(u_arr, i_arr, chunk=1 << 24):
    """True when the (user, item) pairs are strictly increasing, i.e. already sorted by user then item with no repeats

    checked in chunks so the temporaries stay small however many pairs there are
    """

    n = len(u_arr)
    for start in range(0, n - 1, chunk):
        stop = min(start + chunk, n - 1)
        u_step = np.diff(u_arr[start:stop + 1])
        increasing = (u_step > 0) | ((u_step == 0) & (np.diff(i_arr[start:stop + 1]) > 0))
        if not increasing.all():
            return False
    return True


def _own_int32(arr, inputs):
    """contiguous int32 version of [arr] that is guaranteed not to share memory with any of the caller's [inputs]"""

    out = np.ascontiguousarray(arr, dtype=np.int32)
    if any(isinstance(src, np.ndarray) and np.may_share_memory(out, src) for src in inputs):
        out = out.copy()
    return out


def _is_scipy_sparse(obj):
    """detect scipy.sparse matrices/arrays without importing scipy unless needed"""
    return hasattr(obj, 'tocsr') and hasattr(obj, 'tocoo') and hasattr(obj, 'shape')


def _slice_csr_rows(indptr, indices, data, row_positions):
    """gather arbitrary row indices (including -1 -> empty row) from CSR arrays in vectorized NumPy"""

    n_out = len(row_positions)
    if n_out == 0:
        return (
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
        )

    valid = row_positions >= 0
    safe_rows = np.where(valid, row_positions, 0)
    starts = np.where(valid, indptr[safe_rows], 0).astype(np.int64, copy=False)
    ends = np.where(valid, indptr[safe_rows + 1], 0).astype(np.int64, copy=False)
    lengths = ends - starts

    out_indptr = np.empty(n_out + 1, dtype=np.int64)
    out_indptr[0] = 0
    np.cumsum(lengths, dtype=np.int64, out=out_indptr[1:])
    total = int(out_indptr[-1])

    if total == 0:
        return (
            out_indptr,
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
        )

    nz_mask = lengths > 0
    starts_nz = starts[nz_mask]
    lengths_nz = lengths[nz_mask]
    cum_before = np.cumsum(lengths_nz, dtype=np.int64) - lengths_nz
    shifts = np.repeat(starts_nz - cum_before, lengths_nz)
    flat_pos = np.arange(total, dtype=np.int64) + shifts

    out_indices = np.ascontiguousarray(indices[flat_pos], dtype=np.int32)
    out_data = np.ascontiguousarray(data[flat_pos], dtype=np.float32)
    return out_indptr, out_indices, out_data


def _overwrite_csr_rows(base_indptr, base_indices, base_data, n_rows, target_rows, repl_indptr, repl_indices, repl_data):
    """replace selected rows of an existing CSR matrix in O(nnz) without materializing dense arrays"""

    if len(target_rows) == 0:
        return base_indptr, base_indices, base_data

    # Deduplicate target_rows keeping last occurrence
    if len(target_rows) > 1:
        rev_uniq, rev_idx = np.unique(target_rows[::-1], return_index=True)
        keep_pos = (len(target_rows) - 1) - rev_idx
        order = np.argsort(keep_pos)
        keep_pos = keep_pos[order]
        target_rows = target_rows[keep_pos]
        repl_indptr, repl_indices, repl_data = _slice_csr_rows(repl_indptr, repl_indices, repl_data, keep_pos)

    repl_map = np.full(n_rows, -1, dtype=np.int64)
    repl_map[target_rows] = np.arange(len(target_rows), dtype=np.int64)
    is_repl = repl_map >= 0

    base_lens = np.diff(base_indptr)
    repl_lens = np.diff(repl_indptr)
    new_lens = np.where(is_repl, repl_lens[np.maximum(repl_map, 0)], base_lens)

    out_indptr = np.empty(n_rows + 1, dtype=np.int64)
    out_indptr[0] = 0
    np.cumsum(new_lens, dtype=np.int64, out=out_indptr[1:])
    total = int(out_indptr[-1])
    if total == 0:
        return out_indptr, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    out_indices = np.empty(total, dtype=np.int32)
    out_data = np.empty(total, dtype=np.float32)

    # Copy kept rows from base
    keep_rows = np.flatnonzero(~is_repl)
    if len(keep_rows) > 0:
        b_s = base_indptr[keep_rows]
        b_l = base_lens[keep_rows]
        nz = b_l > 0
        if nz.any():
            kr = keep_rows[nz]
            bs = b_s[nz]
            bl = b_l[nz]
            os_ = out_indptr[kr]
            cb = np.cumsum(bl, dtype=np.int64) - bl
            src_pos = np.arange(int(bl.sum()), dtype=np.int64) + np.repeat(bs - cb, bl)
            dst_pos = np.arange(int(bl.sum()), dtype=np.int64) + np.repeat(os_ - cb, bl)
            out_indices[dst_pos] = base_indices[src_pos]
            out_data[dst_pos] = base_data[src_pos]

    # Copy replacement rows
    if len(target_rows) > 0:
        r_s = repl_indptr[:-1]
        r_l = repl_lens
        nz = r_l > 0
        if nz.any():
            tr = target_rows[nz]
            rs = r_s[nz]
            rl = r_l[nz]
            os_ = out_indptr[tr]
            cb = np.cumsum(rl, dtype=np.int64) - rl
            src_pos = np.arange(int(rl.sum()), dtype=np.int64) + np.repeat(rs - cb, rl)
            dst_pos = np.arange(int(rl.sum()), dtype=np.int64) + np.repeat(os_ - cb, rl)
            out_indices[dst_pos] = repl_indices[src_pos]
            out_data[dst_pos] = repl_data[src_pos]

    return out_indptr, out_indices, out_data


def _fast_unique_and_map(col):
    """fast O(N) bitmap unique+map for non-negative bounded integer IDs, falling back to Pandas Index otherwise"""

    arr = np.asarray(col)
    n = len(arr)
    if n > 0 and arr.dtype.kind in ('i', 'u'):
        c_min = int(arr.min())
        c_max = int(arr.max())
        # Use flat array LUT when integer domain is reasonably dense relative to N
        if c_min >= 0 and c_max <= max(65536, n * 4) and c_max < 200_000_000:
            if c_min == 0:
                present = np.zeros(c_max + 1, dtype=bool)
                present[arr] = True
                if present.all():
                    # Contiguous 0..U-1 fast path: zero hash tables, RangeIndex!
                    r_idx = pd.RangeIndex(0, c_max + 1)
                    mapped = np.ascontiguousarray(arr, dtype=np.int32)
                    return r_idx, mapped
                uniq_vals = np.flatnonzero(present).astype(arr.dtype)
                lut = np.full(c_max + 1, -1, dtype=np.int32)
                lut[uniq_vals] = np.arange(len(uniq_vals), dtype=np.int32)
                return pd.Index(uniq_vals), lut[arr]
            else:
                span = c_max - c_min + 1
                shifted = arr - c_min
                present = np.zeros(span, dtype=bool)
                present[shifted] = True
                uniq_off = np.flatnonzero(present)
                uniq_vals = (uniq_off + c_min).astype(arr.dtype)
                lut = np.full(span, -1, dtype=np.int32)
                lut[uniq_off] = np.arange(len(uniq_vals), dtype=np.int32)
                return pd.Index(uniq_vals), lut[shifted]

    uniq = np.sort(pd.unique(arr))
    idx_obj = pd.Index(uniq)
    mapped = idx_obj.get_indexer(arr).astype(np.int32)
    return idx_obj, mapped


def _fast_get_indexer(index_obj, query_arr):
    """O(1) arithmetic indexer for pd.RangeIndex(0, stop, 1), falling back to Index.get_indexer otherwise"""

    q = np.asarray(query_arr)
    is_zero_based_range = isinstance(index_obj, pd.RangeIndex) and index_obj.start == 0 and index_obj.step == 1
    if is_zero_based_range and q.dtype.kind in ('i', 'u'):
        stop = index_obj.stop
        ok = (q >= 0) & (q < stop)
        if ok.all():
            return q.astype(np.intp, copy=False)
        out = np.full(len(q), -1, dtype=np.intp)
        out[ok] = q[ok]
        return out
    return index_obj.get_indexer(q)


class RankFM():
    """Factorization Machines for Ranking Problems with Implicit Feedback Data"""

    def __init__(
        self,
        factors=10,
        loss='bpr',
        max_samples=10,
        alpha=0.01,
        beta=0.1,
        sigma=0.1,
        learning_rate=0.1,
        learning_schedule='constant',
        learning_exponent=0.25,
        user_dropout=0.0,
        n_jobs=1,
        feature_reg='stochastic',
        normalize_steps=True,
        dense_updates='lockfree',
        # --- RankFM v0.5.0 Extensions ---
        temperature=0.12,
        neg_sampling='uniform',
        neg_pop_exponent=0.75,
        neg_pop_mix=0.50,
        logq_weight=0.0,
        margin=0.0,
        weight_margin_scale=0.0,
        neg_weight=1.0,
        use_history=False,
        history_exponent=0.5,
        max_history_len=64,
        use_seq_anchor=False,
        n_event_types=1,
        use_field_gates=False,
        use_freq_gate=False,
        freq_gate_static_drop=0.7,
        n_interests=1,
        interest_temperature=0.25,
        head_agg_temperature=0.0,
        sparse_residual_k=0,
        sparse_residual_l2=500.0,
        sparse_residual_weight=0.0,
        target_attention_topk=0,
        target_attention_weight=0.0,
    ):
        """store hyperparameters and initialize internal model state

        :param factors: latent factor rank
        :param loss: optimization/loss function to use for training: ['bpr', 'warp']
        :param max_samples: maximum number of negative samples to draw for WARP loss
        :param alpha: L2 regularization penalty on [user, item] model weights
        :param beta: L2 regularization penalty on [user-feature, item-feature] model weights
        :param sigma: standard deviation to use for random initialization of factor weights
        :param learning_rate: initial learning rate for gradient step updates
        :param learning_schedule: schedule for adjusting learning rates by training epoch: ['constant', 'invscaling']
        :param learning_exponent: exponent applied to epoch number to adjust learning rate: scaling = 1 / pow(epoch + 1, learning_exponent)
        :param user_dropout: probability in [0.0, 1.0) of masking user-ID factors during training when user_features are present,
                             encouraging user-feature factors [v_uf] to generalize to unseen cold-start users
        :param n_jobs: number of C worker threads (1 = single-threaded, -1 = all CPU cores, -2 = all cores minus 1)
        :param feature_reg: regularization mode for sparse user/item features: ['stochastic', 'uniform']
        :param normalize_steps: scale each SGD step by the feature mass of the rows it touches, so that adding side features
                                does not raise the effective learning rate. with False the raw [learning_rate] is applied to
                                every weight, which diverges once rows carry more than a few active features
        :param dense_updates: how weights of densely used feature columns (non-zero on at least 1/16 of rows) are updated:
                              'lockfree' writes them on every step like all other weights, which stops scaling beyond
                              a few threads because every thread keeps writing the same rows. 'batched' has each thread
                              collect those updates and apply them once every 64 steps, scaled down when
                              learning_rate * threads * 64 exceeds a stability limit, which restores scaling across
                              threads. not available with feature_reg='uniform'
        :return: None
        """

        require(isinstance(factors, int) and factors >= 1, "[factors] must be a positive integer")
        require(isinstance(loss, str) and loss in ('bpr', 'warp', 'cosine_nce', 'simplex'), "[loss] must be in ('bpr', 'warp', 'cosine_nce', 'simplex')")
        require(isinstance(max_samples, int) and max_samples > 0, "[max_samples] must be a positive integer")
        require(isinstance(alpha, (int, float)) and float(alpha) > 0.0, "[alpha] must be a positive float")
        require(isinstance(beta, (int, float)) and float(beta) > 0.0, "[beta] must be a positive float")
        require(isinstance(sigma, (int, float)) and float(sigma) > 0.0, "[sigma] must be a positive float")
        require(isinstance(learning_rate, (int, float)) and float(learning_rate) > 0.0, "[learning_rate] must be a positive float")
        require(isinstance(learning_schedule, str) and learning_schedule in ('constant', 'invscaling'), "[learning_schedule] must be in ('constant', 'invscaling')")
        require(isinstance(learning_exponent, (int, float)) and float(learning_exponent) > 0.0, "[learning_exponent] must be a positive float")
        max_drop_ok = (use_history or use_seq_anchor or loss in ('cosine_nce', 'simplex'))
        if max_drop_ok:
            require(isinstance(user_dropout, (int, float)) and 0.0 <= float(user_dropout) <= 1.0, "[user_dropout] must be in [0.0, 1.0]")
        else:
            require(isinstance(user_dropout, (int, float)) and 0.0 <= float(user_dropout) < 1.0, "[user_dropout] must be a float in [0.0, 1.0)")
        require(isinstance(n_jobs, int) and n_jobs != 0, "[n_jobs] must be a non-zero integer (e.g. 1, 4, -1)")
        require(isinstance(feature_reg, str) and feature_reg in ('stochastic', 'uniform'), "[feature_reg] must be in ('stochastic', 'uniform')")
        require(isinstance(normalize_steps, bool), "[normalize_steps] must be a boolean value")
        require(isinstance(dense_updates, str) and dense_updates in ('lockfree', 'batched'), "[dense_updates] must be in ('lockfree', 'batched')")
        require(not (dense_updates == 'batched' and feature_reg == 'uniform'), "[dense_updates]='batched' cannot be combined with [feature_reg]='uniform'")

        require(isinstance(temperature, (int, float)) and float(temperature) > 0.0, "[temperature] must be positive")
        require(isinstance(neg_sampling, str) and neg_sampling in ('uniform', 'popularity'), "[neg_sampling] must be in ('uniform', 'popularity')")
        require(isinstance(neg_pop_exponent, (int, float)) and 0.0 <= float(neg_pop_exponent) <= 2.0, "[neg_pop_exponent] must be in [0.0, 2.0]")
        require(isinstance(neg_pop_mix, (int, float)) and 0.0 <= float(neg_pop_mix) <= 1.0, "[neg_pop_mix] must be in [0.0, 1.0]")
        require(isinstance(max_history_len, int) and max_history_len >= 1, "[max_history_len] must be >= 1")
        require(isinstance(n_event_types, int) and n_event_types >= 1, "[n_event_types] must be >= 1")
        require(isinstance(n_interests, int) and n_interests >= 1, "[n_interests] must be >= 1")
        require(isinstance(interest_temperature, (int, float)) and float(interest_temperature) > 0.0, "[interest_temperature] must be positive")
        require(isinstance(head_agg_temperature, (int, float)) and float(head_agg_temperature) >= 0.0, "[head_agg_temperature] must be >= 0.0")
        require(isinstance(sparse_residual_k, int) and sparse_residual_k >= 0, "[sparse_residual_k] must be >= 0")
        require(isinstance(target_attention_topk, int) and target_attention_topk >= 0, "[target_attention_topk] must be >= 0")

        self.factors = int(factors)
        self.loss = loss
        self.max_samples = int(max_samples)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sigma = float(sigma)
        self.learning_rate = float(learning_rate)
        self.learning_schedule = learning_schedule
        self.learning_exponent = float(learning_exponent)
        self.user_dropout = float(user_dropout)
        self.n_jobs = int(n_jobs)
        self.feature_reg = feature_reg
        self.normalize_steps = normalize_steps
        self.dense_updates = dense_updates

        self.temperature = float(temperature)
        self.neg_sampling = neg_sampling
        self.neg_pop_exponent = float(neg_pop_exponent)
        self.neg_pop_mix = float(neg_pop_mix)
        self.logq_weight = float(logq_weight)
        self.margin = float(margin)
        self.weight_margin_scale = float(weight_margin_scale)
        self.neg_weight = float(neg_weight)
        self.use_history = bool(use_history)
        self.history_exponent = float(history_exponent)
        self.max_history_len = int(max_history_len)
        self.use_seq_anchor = bool(use_seq_anchor)
        self.n_event_types = int(n_event_types)
        self.use_field_gates = bool(use_field_gates)
        self.use_freq_gate = bool(use_freq_gate)
        self.freq_gate_static_drop = float(freq_gate_static_drop)
        self.n_interests = int(n_interests)
        self.interest_temperature = float(interest_temperature)
        self.head_agg_temperature = float(head_agg_temperature)
        self.sparse_residual_k = int(sparse_residual_k)
        self.sparse_residual_l2 = float(sparse_residual_l2)
        self.sparse_residual_weight = float(sparse_residual_weight)
        self.target_attention_topk = int(target_attention_topk)
        self.target_attention_weight = float(target_attention_weight)

        self._reset_state()


    # --------------------------------
    # begin private method definitions
    # --------------------------------


    @property
    def _use_cosine(self):
        return self.loss in ('cosine_nce', 'simplex')


    @property
    def _is_v050_mode(self):
        return (
            self.loss in ('cosine_nce', 'simplex')
            or self.neg_sampling != 'uniform'
            or self.logq_weight != 0.0
            or self.margin != 0.0
            or self.weight_margin_scale != 0.0
            or self.neg_weight != 1.0
            or self.use_history
            or self.use_seq_anchor
            or self.use_field_gates
            or self.use_freq_gate
            or self.n_interests > 1
            or self.sparse_residual_k > 0
            or self.target_attention_topk > 0
            or self.user_dropout >= 1.0
        )


    def _resolve_n_jobs(self, n_jobs=None):
        nj = self.n_jobs if n_jobs is None else int(n_jobs)
        require(nj != 0, "[n_jobs] must be non-zero")
        if nj < 0:
            return max(1, _available_cpus() + 1 + nj)
        return nj


    def _invalidate_caches(self):
        """clear cached effective embeddings and materialized dense/dict views"""

        self._cached_item_bias = None
        self._cached_v_i_eff = None
        self._cached_v_i_unit = None
        self._cached_train_u_emb = None
        self._user_items_cache = None
        self._dense_x_uf_cache = None
        self._dense_x_if_cache = None


    def _reset_state(self):
        """initialize or reset internal model state"""

        # Primary Index objects (pd.RangeIndex or pd.Index) + lazy Series wrappers
        self._user_index = None
        self._item_index = None
        self._user_id_series = None
        self._item_id_series = None
        self._user_to_index_series = None
        self._item_to_index_series = None
        self.user_idx = None
        self.item_idx = None

        # [user_idx] and [item_idx] of the interactions used by the most recent fit call, and their importance
        # weights (None means every interaction weighs 1)
        self._intx_u = None
        self._intx_i = None
        self.sample_weight = None

        # CSR representation of observed items per user
        self._ui_indptr = None
        self._ui_indices = None

        # Native CSR [user, item] features and schema metadata
        self._uf_indptr = None
        self._uf_indices = None
        self._uf_data = None
        self._uf_any = 0
        self._n_uf = 1

        self._if_indptr = None
        self._if_indices = None
        self._if_data = None
        self._if_any = 0
        self._n_if = 1

        self.has_uf = False
        self.has_if = False
        self.uf_columns = None
        self.if_columns = None

        # [item, item-feature] scalar weights
        self.w_i = None
        self.w_if = None

        # [user, item, user-feature, item-feature] latent factors
        self.v_u = None
        self.v_i = None
        self.v_uf = None
        self.v_if = None

        # v0.5.0 state tensors
        self.p_hist = None
        self.p_seq = None
        self.g_event = None
        self.g_seq = None
        self.gamma_uf = None
        self.gamma_if = None
        self.uf_field = None
        self.if_field = None
        self.if_is_dyn = None
        self.a_gate = None
        self.b_gate = None
        self.c_proto = None
        self.item_log_freq = None
        self._alias_prob = None
        self._alias_idx = None
        self._log_q = None
        self._uh_indptr = None
        self._uh_indices = None
        self._uh_weights = None
        self._uh_kinds = None
        self._train_last_items = None
        self._s_indptr = None
        self._s_indices = None
        self._s_data = None
        self.w_ctx = None

        self._invalidate_caches()
        self.is_fit = False


    # Lazy backwards-compatible ID/mapping properties (avoiding eager giant Series allocations on RangeIndex)
    @property
    def user_id(self):
        if self._user_index is None:
            return None
        if self._user_id_series is None:
            self._user_id_series = pd.Series(self._user_index.to_numpy())
        return self._user_id_series

    @user_id.setter
    def user_id(self, val):
        self._user_id_series = val
        self._user_index = pd.Index(val.values) if isinstance(val, pd.Series) else (pd.Index(val) if val is not None else None)

    @property
    def item_id(self):
        if self._item_index is None:
            return None
        if self._item_id_series is None:
            self._item_id_series = pd.Series(self._item_index.to_numpy())
        return self._item_id_series

    @item_id.setter
    def item_id(self, val):
        self._item_id_series = val
        self._item_index = pd.Index(val.values) if isinstance(val, pd.Series) else (pd.Index(val) if val is not None else None)

    @property
    def index_to_user(self):
        return self.user_id

    @property
    def index_to_item(self):
        return self.item_id

    @property
    def user_to_index(self):
        if self._user_index is None:
            return None
        if self._user_to_index_series is None:
            self._user_to_index_series = pd.Series(
                data=np.arange(len(self._user_index), dtype=np.int64),
                index=self._user_index,
            )
        return self._user_to_index_series

    @property
    def item_to_index(self):
        if self._item_index is None:
            return None
        if self._item_to_index_series is None:
            self._item_to_index_series = pd.Series(
                data=np.arange(len(self._item_index), dtype=np.int64),
                index=self._item_index,
            )
        return self._item_to_index_series


    @property
    def interactions(self):
        """(N, 2) int32 array of the [user_idx, item_idx] pairs used by the most recent fit call, built on request

        the pairs are stored as two parallel arrays (the item array doubles as the seen-items index when the
        input allows it), so this stacked view is a copy and is not kept
        """
        if self._intx_u is None:
            return None
        return np.column_stack([self._intx_u, self._intx_i])


    @property
    def x_uf(self):
        """lazily materialize dense (U, P_uf) float32 user-feature matrix for backwards compatibility"""
        if self._uf_indptr is None:
            return None
        if self._dense_x_uf_cache is None:
            n_u = len(self._uf_indptr) - 1
            dense = np.zeros((n_u, self._n_uf), dtype=np.float32)
            if self._uf_any and len(self._uf_data) > 0:
                rows = np.repeat(np.arange(n_u, dtype=np.int64), np.diff(self._uf_indptr))
                dense[rows, self._uf_indices] = self._uf_data
            self._dense_x_uf_cache = dense
        return self._dense_x_uf_cache


    @property
    def x_if(self):
        """lazily materialize dense (I, P_if) float32 item-feature matrix for backwards compatibility"""
        if self._if_indptr is None:
            return None
        if self._dense_x_if_cache is None:
            n_i = len(self._if_indptr) - 1
            dense = np.zeros((n_i, self._n_if), dtype=np.float32)
            if self._if_any and len(self._if_data) > 0:
                rows = np.repeat(np.arange(n_i, dtype=np.int64), np.diff(self._if_indptr))
                dense[rows, self._if_indices] = self._if_data
            self._dense_x_if_cache = dense
        return self._dense_x_if_cache


    @property
    def x_uf_csr(self):
        """return zero-copy scipy.sparse.csr_matrix view of internal user features"""
        if self._uf_indptr is None:
            return None
        import scipy.sparse as sp
        n_u = len(self._uf_indptr) - 1
        if not self._uf_any:
            return sp.csr_matrix((n_u, self._n_uf), dtype=np.float32)
        return sp.csr_matrix((self._uf_data, self._uf_indices, self._uf_indptr), shape=(n_u, self._n_uf), copy=False)


    @property
    def x_if_csr(self):
        """return zero-copy scipy.sparse.csr_matrix view of internal item features"""
        if self._if_indptr is None:
            return None
        import scipy.sparse as sp
        n_i = len(self._if_indptr) - 1
        if not self._if_any:
            return sp.csr_matrix((n_i, self._n_if), dtype=np.float32)
        return sp.csr_matrix((self._if_data, self._if_indices, self._if_indptr), shape=(n_i, self._n_if), copy=False)


    @property
    def user_items(self):
        """lazily materialize {user_idx: sorted int32 item_idx array} dictionary from CSR storage"""

        if self._ui_indptr is None:
            return None
        if self._user_items_cache is None:
            indptr = self._ui_indptr
            indices = self._ui_indices
            self._user_items_cache = {
                u: indices[indptr[u]:indptr[u + 1]]
                for u in range(len(indptr) - 1)
            }
        return self._user_items_cache


    @user_items.setter
    def user_items(self, value):
        self._user_items_cache = value


    @staticmethod
    def _pairs_to_csr(u_arr, i_arr, n_users, n_items):
        """convert (user_idx, item_idx) integer arrays into sorted, deduplicated CSR (indptr, indices) arrays"""

        # indptr is always int64: that is what the C kernels read, so they never need a converted copy
        if n_users == 0:
            return np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int32)
        if len(u_arr) == 0 or n_items == 0:
            return np.zeros(n_users + 1, dtype=np.int64), np.zeros(0, dtype=np.int32)

        # in-place sort + adjacent compare: several times faster than np.unique, which hashes on NumPy >= 2.3
        codes = u_arr.astype(np.int64) * np.int64(n_items) + i_arr.astype(np.int64)
        codes.sort()
        is_first = np.empty(len(codes), dtype=bool)
        is_first[0] = True
        np.not_equal(codes[1:], codes[:-1], out=is_first[1:])
        uniq_codes = codes[is_first]
        uniq_u = (uniq_codes // np.int64(n_items)).astype(np.int32)
        uniq_i = np.ascontiguousarray(uniq_codes % np.int64(n_items), dtype=np.int32)

        counts = np.bincount(uniq_u, minlength=n_users)
        indptr = np.empty(n_users + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, dtype=np.int64, out=indptr[1:])
        return indptr, uniq_i


    @staticmethod
    def _merge_pairs_into_csr(indptr, indices, u_arr, i_arr, n_users, n_items):
        """add (user_idx, item_idx) pairs to an existing sorted, deduplicated CSR, growing it to [n_users] rows

        only the incoming pairs are sorted. each is then placed by a binary search inside its own user's row, and
        the stored history is touched once, by the block copy that opens the gaps. nothing is re-sorted or re-keyed.
        """

        if len(indices) == 0:
            return RankFM._pairs_to_csr(u_arr, i_arr, n_users, n_items)

        out_indptr = np.empty(n_users + 1, dtype=np.int64)
        out_indptr[:len(indptr)] = indptr
        out_indptr[len(indptr):] = indptr[-1]
        if len(u_arr) == 0:
            return out_indptr, indices

        codes = u_arr.astype(np.int64) * np.int64(n_items) + i_arr.astype(np.int64)
        codes.sort()
        is_first = np.empty(len(codes), dtype=bool)
        is_first[0] = True
        np.not_equal(codes[1:], codes[:-1], out=is_first[1:])
        codes = codes[is_first]
        new_u = np.ascontiguousarray(codes // np.int64(n_items), dtype=np.int32)
        new_i = np.ascontiguousarray(codes % np.int64(n_items), dtype=np.int32)

        positions, present = _locate_pairs_in_csr(out_indptr, indices, new_u, new_i)
        unseen = ~present
        if unseen.any():
            indices = np.insert(indices, positions[unseen], new_i[unseen])
            out_indptr[1:] += np.cumsum(np.bincount(new_u[unseen], minlength=n_users))
        return out_indptr, np.ascontiguousarray(indices, dtype=np.int32)


    @staticmethod
    def _gather_csr_rows(src_indptr, src_indices, row_ids):
        """vectorized gather of selected CSR rows into flat (local_row_idx, col_idx) arrays without Python loops"""

        starts = src_indptr[row_ids].astype(np.int64, copy=False)
        ends = src_indptr[row_ids + 1].astype(np.int64, copy=False)
        lengths = ends - starts
        total = int(lengths.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

        nz = lengths > 0
        if not nz.all():
            local_ids = np.flatnonzero(nz).astype(np.int32)
            starts_nz = starts[nz]
            lengths_nz = lengths[nz]
            cum_before = np.cumsum(lengths_nz, dtype=np.int64) - lengths_nz
            shifts = np.repeat(starts_nz - cum_before, lengths_nz)
            flat_pos = np.arange(total, dtype=np.int64) + shifts
            return np.repeat(local_ids, lengths_nz), src_indices[flat_pos]

        cum_before = np.cumsum(lengths, dtype=np.int64) - lengths
        shifts = np.repeat(starts - cum_before, lengths)
        flat_pos = np.arange(total, dtype=np.int64) + shifts
        gathered_cols = src_indices[flat_pos]
        local_rows = np.repeat(np.arange(len(row_ids), dtype=np.int32), lengths)
        return local_rows, gathered_cols


    @staticmethod
    def _extract_feature_csr(features, expected_columns=None, expected_width=None, default_index=None, n_threads=1, name='features'):
        """extract (pd.Index of IDs, indptr_i64, indices_i32, data_f32, n_cols, columns_list) from dense or sparse inputs

        Supported formats for [features]:
          1. pd.DataFrame with ID in first column and numeric feature columns in remaining columns
          2. 2D np.ndarray with ID in first column and numeric feature columns in remaining columns
          3. scipy.sparse matrix of shape (len(default_index), P) aligned directly to default_index
          4. (ids, matrix) tuple where [ids] is a 1D array/Series/Index and [matrix] is scipy.sparse or 2D float ndarray
        """

        if isinstance(features, _FeatureCSR):
            return tuple(features)

        cols_out = None

        if _is_scipy_sparse(features):
            if default_index is None:
                raise ValueError("[{}] passed as bare sparse matrix without an ID tuple (ids, csr_matrix)".format(name))
            if features.shape[0] != len(default_index):
                raise KeyError("the rows in sparse [{}] ({}) do not match entity count ({})".format(
                    name, features.shape[0], len(default_index)
                ))
            raw_ids = default_index
            mat_sp = features
        elif isinstance(features, tuple) and len(features) == 2:
            raw_ids_in, mat_in = features
            raw_ids = raw_ids_in if isinstance(raw_ids_in, pd.Index) else pd.Index(np.asarray(raw_ids_in))
            if _is_scipy_sparse(mat_in):
                if len(raw_ids) != mat_in.shape[0]:
                    raise ValueError("[{}] IDs length ({}) != matrix rows ({})".format(name, len(raw_ids), mat_in.shape[0]))
                mat_sp = mat_in
            else:
                mat_np = np.ascontiguousarray(mat_in, dtype=np.float32)
                if mat_np.ndim != 2 or len(raw_ids) != mat_np.shape[0]:
                    raise ValueError("[{}] (ids, dense_matrix) must have matching row count and 2D matrix".format(name))
                mat_sp = None
        else:
            mat_sp = None
            require(isinstance(features, (np.ndarray, pd.DataFrame)), "[{}] must be np.ndarray, pd.DataFrame, scipy.sparse matrix, or (ids, matrix) tuple".format(name))
            require(features.shape[1] >= 2, "[{}] must contain an ID column plus at least one feature column".format(name))

            if isinstance(features, pd.DataFrame):
                id_col = features.columns[0]
                raw_ids = pd.Index(features[id_col].to_numpy())
                feat_df = features.iloc[:, 1:]
                cols_out = list(feat_df.columns)
                if expected_columns is not None and len(cols_out) == len(expected_columns):
                    if cols_out != expected_columns and set(expected_columns).issubset(set(cols_out)):
                        feat_df = feat_df[expected_columns]
                        cols_out = list(feat_df.columns)
                mat_np = np.ascontiguousarray(feat_df.to_numpy(), dtype=np.float32)
            else:
                raw_ids = pd.Index(features[:, 0])
                mat_np = np.ascontiguousarray(features[:, 1:], dtype=np.float32)

        if mat_sp is not None:
            csr = mat_sp.tocsr()
            if not csr.has_sorted_indices:
                csr = csr.sorted_indices()  # a sorted copy: sort_indices() would reorder the caller's matrix in place
            n_cols = int(csr.shape[1])
            if expected_width is not None and n_cols != expected_width:
                raise ValueError("[{}] has {} feature columns, expected {}".format(name, n_cols, expected_width))
            if n_cols < 1:
                raise AssertionError("[{}] must have at least 1 feature column".format(name))

            indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
            indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
            data = np.ascontiguousarray(csr.data, dtype=np.float32)
            require(np.isfinite(data.sum(dtype=np.float64)), "[{}] contains NaN or infinite values".format(name))

            if raw_ids.has_duplicates:
                keep_pos = np.flatnonzero(~raw_ids.duplicated(keep='last'))
                raw_ids = raw_ids[keep_pos]
                indptr, indices, data = _slice_csr_rows(indptr, indices, data, keep_pos)
            return raw_ids, indptr, indices, data, n_cols, cols_out

        # Dense matrix path -> Cython 2-pass parallel dense_to_csr
        n_cols = int(mat_np.shape[1])
        if expected_width is not None and n_cols != expected_width:
            raise ValueError("[{}] has {} feature columns, expected {}".format(name, n_cols, expected_width))

        if raw_ids.has_duplicates:
            keep_mask = ~raw_ids.duplicated(keep='last')
            raw_ids = raw_ids[keep_mask]
            mat_np = np.ascontiguousarray(mat_np[keep_mask], dtype=np.float32)

        indptr, indices, data, has_any = dense_to_csr(mat_np, n_threads)
        if not has_any:
            indices = np.zeros(0, dtype=np.int32)
            data = np.zeros(0, dtype=np.float32)
        require(np.isfinite(data.sum(dtype=np.float64)), "[{}] contains NaN or infinite values".format(name))
        return raw_ids, indptr, indices, data, n_cols, cols_out


    @staticmethod
    def _extract_interaction_columns(interactions):
        """extract (user_col_array, item_col_array) without full DataFrame copies"""

        require(isinstance(interactions, (np.ndarray, pd.DataFrame)), "[interactions] must be np.ndarray, pd.DataFrame, or scipy.sparse matrix")
        require(interactions.shape[1] == 2, "[interactions] should be: [user_id, item_id]")

        if isinstance(interactions, pd.DataFrame):
            return interactions.iloc[:, 0].to_numpy(), interactions.iloc[:, 1].to_numpy()
        return interactions[:, 0], interactions[:, 1]


    def _init_all(
        self,
        interactions,
        user_features=None,
        item_features=None,
        sample_weight=None,
        n_threads=1,
        seed=1492,
        user_field_ids=None,
        item_field_ids=None,
        item_dynamic_mask=None,
    ):
        """index the interaction data and user/item features and initialize model weights"""

        if _is_scipy_sparse(interactions):
            # Zero-Pandas sparse matrix fast path: rows = 0..U-1, cols = 0..I-1, data = sample_weight (unless overridden)
            n_u, n_i = int(interactions.shape[0]), int(interactions.shape[1])
            require(n_u >= 1 and n_i >= 1, "sparse [interactions] must have positive shape (n_users, n_items)")
            self._user_index = pd.RangeIndex(0, n_u)
            self._item_index = pd.RangeIndex(0, n_i)
            self.user_idx = np.arange(n_u, dtype=np.int32)
            self.item_idx = np.arange(n_i, dtype=np.int32)

            coo = interactions.tocoo(copy=False)
            u_arr = np.ascontiguousarray(coo.row, dtype=np.int32)
            i_arr = np.ascontiguousarray(coo.col, dtype=np.int32)
            if sample_weight is not None:
                require(isinstance(sample_weight, (np.ndarray, pd.Series)), "[sample_weight] must be np.ndarray or pd.series")
                require(sample_weight.ndim == 1 and len(sample_weight) == len(u_arr), "[sample_weight] must match sparse nnz")
                self.sample_weight = np.array(get_data(sample_weight), dtype=np.float32)
            elif len(coo.data) > 0 and coo.data.min() == 1 and coo.data.max() == 1:
                self.sample_weight = None  # a binary matrix: no per-interaction weights to store
            else:
                self.sample_weight = np.array(coo.data, dtype=np.float32)

            caller_arrays = (coo.row, coo.col, getattr(interactions, 'indices', None))
            self._index_initial_pairs(u_arr, i_arr, n_u, n_i, caller_arrays, keep_input_order=self._is_v050_mode)
        else:
            u_col, i_col = self._extract_interaction_columns(interactions)
            self._user_index, u_mapped = _fast_unique_and_map(u_col)
            self._item_index, i_mapped = _fast_unique_and_map(i_col)
            self.user_idx = np.arange(len(self._user_index), dtype=np.int32)
            self.item_idx = np.arange(len(self._item_index), dtype=np.int32)

            if sample_weight is not None:
                require(isinstance(sample_weight, (np.ndarray, pd.Series)), "[sample_weight] must be np.ndarray or pd.series")
                require(sample_weight.ndim == 1, "[sample_weight] must a vector (ndim=1)")
                require(len(sample_weight) == len(u_col), "[sample_weight] must have the same length as [interactions]")
                self.sample_weight = np.array(get_data(sample_weight), dtype=np.float32)
            else:
                self.sample_weight = None

            self._index_initial_pairs(
                u_mapped, i_mapped, len(self.user_idx), len(self.item_idx), (u_col, i_col), keep_input_order=self._is_v050_mode
            )

        self._init_features(user_features, item_features, n_threads=n_threads)
        self._init_weights(
            user_features,
            item_features,
            seed=seed,
            user_field_ids=user_field_ids,
            item_field_ids=item_field_ids,
            item_dynamic_mask=item_dynamic_mask,
        )


    def _index_initial_pairs(self, u_arr, i_arr, n_users, n_items, caller_arrays, keep_input_order=False):
        """build the seen-items CSR for a fresh fit and choose the arrays training reads, sharing storage where possible

        three layouts, cheapest first (bytes per interaction, before optional weights):
          8   pairs already sorted by (user, item) with no repeats: the item column is used as the CSR index as is
          8   unique pairs in any order and no weights: training reads the sorted index, the input order is dropped
          12  anything else (repeats, or unsorted pairs with weights): the index plus the pairs in their input order
        repeats are kept as separate training rows in the last layout, so their effect on training is unchanged.

        [keep_input_order] rules out the second layout. The v0.5 training path needs it: prev_items and margin_signal are
        aligned row by row with the interactions as given, and a user's "last item" is read off the input order.
        """

        self._user_items_cache = None

        if _is_sorted_unique_pairs(u_arr, i_arr):
            indptr = np.zeros(n_users + 1, dtype=np.int64)
            np.cumsum(np.bincount(u_arr, minlength=n_users), out=indptr[1:])
            self._ui_indptr = indptr
            self._ui_indices = _own_int32(i_arr, caller_arrays)
            self._intx_u = _own_int32(u_arr, caller_arrays)
            self._intx_i = self._ui_indices
            return

        self._ui_indptr, self._ui_indices = self._pairs_to_csr(u_arr, i_arr, n_users, n_items)
        if not keep_input_order and self.sample_weight is None and len(self._ui_indices) == len(u_arr):
            self._intx_u = np.repeat(np.arange(n_users, dtype=np.int32), np.diff(self._ui_indptr))
            self._intx_i = self._ui_indices
        else:
            self._intx_u = _own_int32(u_arr, caller_arrays)
            self._intx_i = _own_int32(i_arr, caller_arrays)


    def _expand_users_for_partial(self, u_col, i_col, freeze_items=False, seed=1492):
        """append any previously unseen users in interactions to the model's user index and state arrays"""

        valid_item_mask = _fast_get_indexer(self._item_index, i_col) >= 0
        incoming_users = pd.Index(pd.unique(u_col[valid_item_mask]))
        new_users = incoming_users.difference(self._user_index).sort_values().to_numpy()

        if len(new_users) == 0:
            return

        n_new = len(new_users)
        old_index = self._user_index
        continues_range = isinstance(old_index, pd.RangeIndex) and old_index.start == 0 and old_index.step == 1
        if continues_range and new_users.dtype.kind in ('i', 'u'):
            # [new_users] is sorted and unique, so matching endpoints mean it is exactly stop .. stop + n_new - 1
            continues_range = int(new_users[0]) == old_index.stop and int(new_users[-1]) == old_index.stop + n_new - 1
        else:
            continues_range = False
        if continues_range:
            # sequential integer IDs stay a RangeIndex: lookups remain arithmetic and nothing is rebuilt
            self._user_index = pd.RangeIndex(0, old_index.stop + n_new)
        else:
            self._user_index = pd.Index(np.concatenate([old_index.to_numpy(), new_users]))
        self._user_id_series = None
        self._user_to_index_series = None
        self.user_idx = np.arange(len(self._user_index), dtype=np.int32)

        n_rows_new = n_new * self.n_interests
        self.v_u = _append_zero_rows(self.v_u, n_rows_new)
        if not (self.has_uf or freeze_items or self.user_dropout >= 1.0):
            rng = np.random.default_rng(seed)
            self.v_u[-n_rows_new:] = rng.normal(loc=0, scale=self.sigma, size=(n_rows_new, self.factors))

        # Extend CSR user-feature indptr with empty rows in O(n_new) (zero dense matrix allocation!)
        last_ptr = int(self._uf_indptr[-1])
        pad_ptr = np.full(n_new, last_ptr, dtype=np.int64)
        self._uf_indptr = np.ascontiguousarray(np.concatenate([self._uf_indptr, pad_ptr]), dtype=np.int64)
        self._dense_x_uf_cache = None
        if self._uh_indptr is not None:
            last_h_ptr = int(self._uh_indptr[-1])
            self._uh_indptr = np.ascontiguousarray(
                np.concatenate([self._uh_indptr, np.full(n_new, last_h_ptr, dtype=np.int64)]), dtype=np.int64
            )
        if self._train_last_items is not None and len(self._train_last_items) > 0:
            self._train_last_items = np.ascontiguousarray(
                np.concatenate([self._train_last_items, np.full(n_new, -1, dtype=np.int32)]), dtype=np.int32
            )


    def _init_interactions(self, interactions, sample_weight, freeze_items=False, seed=1492):
        """map incremental interaction data to internal user/item indexes and update CSR user_items"""

        if _is_scipy_sparse(interactions):
            coo = interactions.tocoo(copy=False)
            u_col = coo.row
            i_col = coo.col
            if sample_weight is None:
                sample_weight = np.asarray(coo.data, dtype=np.float32)
        else:
            u_col, i_col = self._extract_interaction_columns(interactions)

        if sample_weight is not None:
            require(isinstance(sample_weight, (np.ndarray, pd.Series)), "[sample_weight] must be np.ndarray or pd.series")
            require(sample_weight.ndim == 1, "[sample_weight] must a vector (ndim=1)")
            require(len(sample_weight) == len(u_col), "[sample_weight] must have the same length as [interactions]")

        if self.is_fit:
            self._expand_users_for_partial(u_col, i_col, freeze_items=freeze_items, seed=seed)

        u_mapped = _fast_get_indexer(self._user_index, u_col)
        i_mapped = _fast_get_indexer(self._item_index, i_col)
        valid_mask = (u_mapped >= 0) & (i_mapped >= 0)

        if valid_mask.all():
            u_arr = u_mapped.astype(np.int32)
            i_arr = i_mapped.astype(np.int32)
        else:
            u_arr = u_mapped[valid_mask].astype(np.int32)
            i_arr = i_mapped[valid_mask].astype(np.int32)

        if sample_weight is not None:
            sw_raw = np.asarray(get_data(sample_weight), dtype=np.float32)
            self.sample_weight = np.array(sw_raw[valid_mask] if not valid_mask.all() else sw_raw, dtype=np.float32)
        else:
            self.sample_weight = None

        n_users = len(self.user_idx)
        n_items = len(self.item_idx)

        if self.is_fit and self._ui_indptr is not None:
            self._ui_indptr, self._ui_indices = self._merge_pairs_into_csr(
                self._ui_indptr, self._ui_indices, u_arr, i_arr, n_users, n_items
            )
        else:
            self._ui_indptr, self._ui_indices = self._pairs_to_csr(u_arr, i_arr, n_users, n_items)

        self._user_items_cache = None
        self._intx_u = np.ascontiguousarray(u_arr, dtype=np.int32)
        self._intx_i = np.ascontiguousarray(i_arr, dtype=np.int32)


    def _init_features(self, user_features=None, item_features=None, n_threads=1):
        """initialize or incrementally update native CSR user/item features without dense allocations"""

        if not self.is_fit:
            n_u = len(self.user_idx)
            if user_features is not None:
                self.has_uf = True
                uf_ids, indptr, indices, data, n_cols, cols = self._extract_feature_csr(
                    user_features,
                    default_index=self._user_index,
                    n_threads=n_threads,
                    name='user_features',
                )
                self.uf_columns = cols
                self._n_uf = n_cols
                if uf_ids.equals(self._user_index):
                    # copies: these arrays may still be the caller's own sparse matrix buffers
                    self._uf_indptr, self._uf_indices, self._uf_data = indptr.copy(), indices.copy(), data.copy()
                else:
                    pos = _fast_get_indexer(uf_ids, self._user_index)
                    if (pos < 0).any():
                        raise KeyError('the users in [user_features] do not match the users in [interactions]')
                    self._uf_indptr, self._uf_indices, self._uf_data = _slice_csr_rows(indptr, indices, data, pos)
                self._uf_any = 1 if len(self._uf_data) > 0 else 0
            else:
                self.has_uf = False
                self.uf_columns = None
                self._n_uf = 1
                self._uf_indptr = np.zeros(n_u + 1, dtype=np.int64)
                self._uf_indices = np.zeros(0, dtype=np.int32)
                self._uf_data = np.zeros(0, dtype=np.float32)
                self._uf_any = 0

            n_i = len(self.item_idx)
            if item_features is not None:
                self.has_if = True
                if_ids, indptr, indices, data, n_cols, cols = self._extract_feature_csr(
                    item_features,
                    default_index=self._item_index,
                    n_threads=n_threads,
                    name='item_features',
                )
                self.if_columns = cols
                self._n_if = n_cols
                if if_ids.equals(self._item_index):
                    self._if_indptr, self._if_indices, self._if_data = indptr.copy(), indices.copy(), data.copy()
                else:
                    pos = _fast_get_indexer(if_ids, self._item_index)
                    if (pos < 0).any():
                        raise KeyError('the items in [item_features] do not match the items in [interactions]')
                    self._if_indptr, self._if_indices, self._if_data = _slice_csr_rows(indptr, indices, data, pos)
                self._if_any = 1 if len(self._if_data) > 0 else 0
            else:
                self.has_if = False
                self.if_columns = None
                self._n_if = 1
                self._if_indptr = np.zeros(n_i + 1, dtype=np.int64)
                self._if_indices = np.zeros(0, dtype=np.int32)
                self._if_data = np.zeros(0, dtype=np.float32)
                self._if_any = 0
        else:
            if user_features is not None:
                if not self.has_uf:
                    raise ValueError("cannot supply [user_features] to fit_partial() when model was fit without [user_features]")
                uf_ids, indptr, indices, data, _, _ = self._extract_feature_csr(
                    user_features,
                    expected_columns=self.uf_columns,
                    expected_width=self._n_uf,
                    default_index=self._user_index,
                    n_threads=n_threads,
                    name='user_features',
                )
                target_u = _fast_get_indexer(self._user_index, uf_ids)
                valid = target_u >= 0
                if valid.any():
                    valid_pos = np.flatnonzero(valid)
                    sub_ptr, sub_idx, sub_dat = _slice_csr_rows(indptr, indices, data, valid_pos)
                    self._uf_indptr, self._uf_indices, self._uf_data = _overwrite_csr_rows(
                        self._uf_indptr, self._uf_indices, self._uf_data,
                        len(self.user_idx), target_u[valid], sub_ptr, sub_idx, sub_dat
                    )
                    self._uf_any = 1 if len(self._uf_data) > 0 else 0
                    self._dense_x_uf_cache = None

            if item_features is not None:
                if not self.has_if:
                    raise ValueError("cannot supply [item_features] to fit_partial() when model was fit without [item_features]")
                if_ids, indptr, indices, data, _, _ = self._extract_feature_csr(
                    item_features,
                    expected_columns=self.if_columns,
                    expected_width=self._n_if,
                    default_index=self._item_index,
                    n_threads=n_threads,
                    name='item_features',
                )
                target_i = _fast_get_indexer(self._item_index, if_ids)
                valid = target_i >= 0
                if valid.any():
                    valid_pos = np.flatnonzero(valid)
                    sub_ptr, sub_idx, sub_dat = _slice_csr_rows(indptr, indices, data, valid_pos)
                    self._if_indptr, self._if_indices, self._if_data = _overwrite_csr_rows(
                        self._if_indptr, self._if_indices, self._if_data,
                        len(self.item_idx), target_i[valid], sub_ptr, sub_idx, sub_dat
                    )
                    self._if_any = 1 if len(self._if_data) > 0 else 0
                    self._dense_x_if_cache = None


    @staticmethod
    def _infer_field_ids(columns, n_cols):
        """infer integer field IDs (0..G-1) from feature column prefixes or fallback column indices"""

        if n_cols <= 0:
            return np.zeros(0, dtype=np.int32), 0
        if columns is not None and len(columns) == n_cols:
            prefixes = []
            for c in columns:
                s = str(c)
                if '__' in s:
                    prefixes.append(s.split('__', 1)[0])
                elif '_' in s:
                    prefixes.append(s.split('_', 1)[0])
                else:
                    prefixes.append(s)
            uniq, inv = np.unique(np.asarray(prefixes, dtype=object), return_inverse=True)
            if 1 < len(uniq) <= n_cols:
                return np.ascontiguousarray(inv, dtype=np.int32), int(len(uniq))
        if n_cols <= 128:
            return np.arange(n_cols, dtype=np.int32), int(n_cols)
        return np.ascontiguousarray(np.arange(n_cols, dtype=np.int32) % 32, dtype=np.int32), 32


    def _build_alias_and_logq(self, item_counts):
        """construct O(1) Walker-Vose alias tables and centered log Q(i) popularity adjustments"""

        n_i = len(item_counts)
        if n_i == 0:
            return (
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
            )

        pop = np.power(np.maximum(np.asarray(item_counts, dtype=np.float64), 1.0), self.neg_pop_exponent)
        pop_sum = float(pop.sum())
        pop_dist = pop / pop_sum if pop_sum > 0.0 else np.full(n_i, 1.0 / n_i, dtype=np.float64)
        unif_dist = np.full(n_i, 1.0 / n_i, dtype=np.float64)

        if self.neg_sampling == 'popularity':
            q = self.neg_pop_mix * pop_dist + (1.0 - self.neg_pop_mix) * unif_dist
            q = q / q.sum()

            scaled = (q * n_i).copy()
            prob = np.ones(n_i, dtype=np.float32)
            alias = np.arange(n_i, dtype=np.int32)
            small = list(np.flatnonzero(scaled < 1.0))
            large = list(np.flatnonzero(scaled >= 1.0))

            while small and large:
                s = small.pop()
                l = large.pop()
                prob[s] = np.float32(scaled[s])
                alias[s] = np.int32(l)
                scaled[l] = (scaled[l] + scaled[s]) - 1.0
                if scaled[l] < 1.0:
                    small.append(l)
                else:
                    large.append(l)
            for l in large:
                prob[l] = 1.0
            for s in small:
                prob[s] = 1.0

            log_q = np.ascontiguousarray(np.log(np.maximum(q * n_i, 1e-12)), dtype=np.float32)
            return prob, alias, log_q

        if self.logq_weight != 0.0:
            log_q = np.ascontiguousarray(np.log(np.maximum(pop_dist * n_i, 1e-12)), dtype=np.float32)
        else:
            log_q = np.zeros(0, dtype=np.float32)
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32), log_q


    def _init_weights(
        self,
        user_features=None,
        item_features=None,
        seed=None,
        user_field_ids=None,
        item_field_ids=None,
        item_dynamic_mask=None,
    ):
        """initialize model weights given user/item and user-feature/item-feature shapes"""

        if self._is_v050_mode:
            eff_s = 1492 if seed is None else int(seed)
            rng = np.random.RandomState((eff_s + 13) & 0x7FFFFFFF)
        elif seed is not None:
            rng = np.random.RandomState((int(seed) + 8) & 0x7FFFFFFF)
        else:
            rng = np.random
        n_u = len(self.user_idx)
        n_i = len(self.item_idx)
        F = self.factors
        M = self.n_interests

        self.w_i = np.zeros(n_i, dtype=np.float32)
        self.w_if = np.zeros(self._n_if, dtype=np.float32)

        if self.user_dropout >= 1.0:
            self.v_u = np.zeros((n_u * M, F), dtype=np.float32)
        else:
            self.v_u = rng.normal(loc=0.0, scale=self.sigma, size=(n_u * M, F)).astype(np.float32)
        self.v_i = rng.normal(loc=0.0, scale=self.sigma, size=(n_i, F)).astype(np.float32)

        if user_features is not None:
            scale = (self.alpha / self.beta) * self.sigma
            self.v_uf = rng.normal(loc=0.0, scale=scale, size=(self._n_uf, F)).astype(np.float32)
        else:
            self.v_uf = np.zeros((self._n_uf, F), dtype=np.float32)

        if item_features is not None:
            scale = (self.alpha / self.beta) * self.sigma
            self.v_if = rng.normal(loc=0.0, scale=scale, size=(self._n_if, F)).astype(np.float32)
        else:
            self.v_if = np.zeros((self._n_if, F), dtype=np.float32)

        if not self._is_v050_mode:
            return

        if self.use_history or self.target_attention_topk > 0:
            self.p_hist = rng.normal(loc=0.0, scale=self.sigma, size=(n_i, F)).astype(np.float32)
        else:
            self.p_hist = np.zeros((0, F), dtype=np.float32)

        if self.use_seq_anchor:
            self.p_seq = rng.normal(loc=0.0, scale=self.sigma, size=(n_i, F)).astype(np.float32)
            self.g_seq = np.ones(F, dtype=np.float32)
        else:
            self.p_seq = np.zeros((0, F), dtype=np.float32)
            self.g_seq = np.zeros(0, dtype=np.float32)

        self.g_event = np.ones((max(1, self.n_event_types), F), dtype=np.float32)

        if user_field_ids is not None:
            uf_arr = np.asarray(user_field_ids)
            require(len(uf_arr) == self._n_uf, "[user_field_ids] length must match user feature width")
            uniq_uf, inv_uf = np.unique(uf_arr, return_inverse=True)
            self.uf_field = np.ascontiguousarray(inv_uf, dtype=np.int32)
            n_g_uf = len(uniq_uf)
        elif self.use_field_gates and self.has_uf:
            self.uf_field, n_g_uf = self._infer_field_ids(self.uf_columns, self._n_uf)
        else:
            self.uf_field = np.zeros(0, dtype=np.int32)
            n_g_uf = 0
        self.gamma_uf = np.ones((n_g_uf, F), dtype=np.float32) if (self.use_field_gates and n_g_uf > 0) else np.zeros((0, F), dtype=np.float32)

        if item_field_ids is not None:
            if_arr = np.asarray(item_field_ids)
            require(len(if_arr) == self._n_if, "[item_field_ids] length must match item feature width")
            uniq_if, inv_if = np.unique(if_arr, return_inverse=True)
            self.if_field = np.ascontiguousarray(inv_if, dtype=np.int32)
            n_g_if = len(uniq_if)
        elif self.use_field_gates and self.has_if:
            self.if_field, n_g_if = self._infer_field_ids(self.if_columns, self._n_if)
        else:
            self.if_field = np.zeros(0, dtype=np.int32)
            n_g_if = 0
        self.gamma_if = np.ones((n_g_if, F), dtype=np.float32) if (self.use_field_gates and n_g_if > 0) else np.zeros((0, F), dtype=np.float32)

        if item_dynamic_mask is not None:
            dyn_arr = np.asarray(item_dynamic_mask, dtype=np.int32)
            require(len(dyn_arr) == self._n_if, "[item_dynamic_mask] length must match item feature width")
            self.if_is_dyn = np.ascontiguousarray(dyn_arr, dtype=np.int32)
        elif self.use_freq_gate and self.has_if:
            self.if_is_dyn = np.zeros(self._n_if, dtype=np.int32)
        else:
            self.if_is_dyn = np.zeros(0, dtype=np.int32)

        if self.use_freq_gate:
            self.a_gate = np.full(F, 0.5, dtype=np.float32)
            self.b_gate = np.full(F, -1.0, dtype=np.float32)
        else:
            self.a_gate = np.zeros(0, dtype=np.float32)
            self.b_gate = np.zeros(0, dtype=np.float32)

        if M > 1:
            raw_proto = rng.normal(loc=0.0, scale=1.0, size=(max(F, M), M))
            q_mat, _ = np.linalg.qr(raw_proto)
            proto = q_mat[:F, :M].T.astype(np.float32)
            norms = np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-6)
            self.c_proto = np.ascontiguousarray(proto / norms, dtype=np.float32)
        else:
            self.c_proto = np.zeros((0, F), dtype=np.float32)

        item_counts = np.bincount(self._intx_i, minlength=n_i).astype(np.float64)
        self.item_log_freq = np.ascontiguousarray(np.log1p(item_counts), dtype=np.float32)
        self._alias_prob, self._alias_idx, self._log_q = self._build_alias_and_logq(item_counts)


    def _max_samples_for_loss(self):
        if self.loss == 'bpr':
            return 1
        elif self.loss in ('warp', 'cosine_nce', 'simplex'):
            return self.max_samples
        else:
            raise ValueError('[loss] function not recognized')


    def _loss_mode_code(self):
        if self.loss == 'bpr':
            return 0
        elif self.loss == 'warp':
            return 1
        elif self.loss == 'cosine_nce':
            return 2
        elif self.loss == 'simplex':
            return 3
        raise ValueError('[loss] function not recognized')


    def _parse_raw_history(self, user_history):
        """extract raw (u_col, i_col, weights_f32, kinds_i32) arrays from supported user_history containers"""

        if _is_scipy_sparse(user_history):
            coo = user_history.tocoo(copy=False)
            u_col = coo.row
            i_col = coo.col
            w_arr = np.asarray(coo.data, dtype=np.float32)
            k_arr = np.zeros(len(u_col), dtype=np.int32)
            return u_col, i_col, w_arr, k_arr

        if isinstance(user_history, dict):
            u_col = np.asarray(user_history['user_id'] if 'user_id' in user_history else user_history['users'])
            i_col = np.asarray(user_history['item_id'] if 'item_id' in user_history else user_history['items'])
            if 'weight' in user_history or 'weights' in user_history:
                w_arr = np.asarray(user_history.get('weight', user_history.get('weights')), dtype=np.float32)
            else:
                w_arr = np.ones(len(u_col), dtype=np.float32)
            if 'event_type' in user_history or 'kinds' in user_history:
                k_arr = np.asarray(user_history.get('event_type', user_history.get('kinds')), dtype=np.int32)
            else:
                k_arr = np.zeros(len(u_col), dtype=np.int32)
            return u_col, i_col, w_arr, k_arr

        if isinstance(user_history, tuple):
            require(2 <= len(user_history) <= 4, "[user_history] tuple must be (users, items[, weights[, event_types]])")
            u_col = np.asarray(user_history[0])
            i_col = np.asarray(user_history[1])
            w_arr = np.asarray(user_history[2], dtype=np.float32) if len(user_history) >= 3 and user_history[2] is not None else np.ones(len(u_col), dtype=np.float32)
            k_arr = np.asarray(user_history[3], dtype=np.int32) if len(user_history) == 4 and user_history[3] is not None else np.zeros(len(u_col), dtype=np.int32)
            return u_col, i_col, w_arr, k_arr

        if isinstance(user_history, pd.DataFrame):
            require(user_history.shape[1] >= 2, "[user_history] DataFrame must have at least [user_id, item_id]")
            u_col = user_history.iloc[:, 0].to_numpy()
            i_col = user_history.iloc[:, 1].to_numpy()
            w_arr = user_history.iloc[:, 2].to_numpy(dtype=np.float32) if user_history.shape[1] >= 3 else np.ones(len(u_col), dtype=np.float32)
            k_arr = user_history.iloc[:, 3].to_numpy(dtype=np.int32) if user_history.shape[1] >= 4 else np.zeros(len(u_col), dtype=np.int32)
            return u_col, i_col, w_arr, k_arr

        arr = np.asarray(user_history)
        require(arr.ndim == 2 and arr.shape[1] >= 2, "[user_history] array must be 2D with >= 2 columns")
        u_col = arr[:, 0]
        i_col = arr[:, 1]
        w_arr = np.asarray(arr[:, 2], dtype=np.float32) if arr.shape[1] >= 3 else np.ones(len(u_col), dtype=np.float32)
        k_arr = np.asarray(arr[:, 3], dtype=np.int32) if arr.shape[1] >= 4 else np.zeros(len(u_col), dtype=np.int32)
        return u_col, i_col, w_arr, k_arr


    def _build_history_csr_from_mapped(self, u_arr, i_arr, w_arr, k_arr, n_users):
        """build per-user weight-sorted, top-max_history_len capped CSR arrays in vectorized NumPy"""

        if n_users <= 0:
            return (
                np.zeros(1, dtype=np.int64),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int32),
            )

        n_items = len(self.item_idx)
        n_ev = max(1, self.n_event_types)
        valid = (u_arr >= 0) & (u_arr < n_users) & (i_arr >= 0) & (i_arr < n_items) & (w_arr > 0.0)
        if not valid.any():
            return (
                np.zeros(n_users + 1, dtype=np.int64),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int32),
            )

        if not valid.all():
            u_v = u_arr[valid].astype(np.int32, copy=False)
            i_v = i_arr[valid].astype(np.int32, copy=False)
            w_v = w_arr[valid].astype(np.float32, copy=False)
            k_v = np.clip(k_arr[valid].astype(np.int32, copy=False), 0, n_ev - 1)
        else:
            u_v = u_arr.astype(np.int32, copy=False)
            i_v = i_arr.astype(np.int32, copy=False)
            w_v = w_arr.astype(np.float32, copy=False)
            k_v = np.clip(k_arr.astype(np.int32, copy=False), 0, n_ev - 1)

        # Deduplicate repeated (u, i, event_type) triples by summing weights
        codes = (u_v.astype(np.int64) * np.int64(n_items) + i_v.astype(np.int64)) * np.int64(n_ev) + k_v.astype(np.int64)
        c_order = np.argsort(codes, kind='mergesort')
        codes_s = codes[c_order]
        w_c = w_v[c_order]
        change = np.empty(len(codes_s), dtype=bool)
        change[0] = True
        if len(codes_s) > 1:
            change[1:] = codes_s[1:] != codes_s[:-1]
        if not change.all():
            grp_starts = np.flatnonzero(change)
            uniq_codes = codes_s[grp_starts]
            w_v = np.add.reduceat(w_c, grp_starts).astype(np.float32, copy=False)
            k_v = (uniq_codes % np.int64(n_ev)).astype(np.int32)
            rem = uniq_codes // np.int64(n_ev)
            i_v = (rem % np.int64(n_items)).astype(np.int32)
            u_v = (rem // np.int64(n_items)).astype(np.int32)

        # Sort within each user by descending weight and cap at max_history_len
        order = np.lexsort((-w_v, u_v))
        u_s = u_v[order]
        i_s = i_v[order]
        w_s = w_v[order]
        k_s = k_v[order]

        counts = np.bincount(u_s, minlength=n_users)
        indptr = np.zeros(n_users + 1, dtype=np.int64)
        np.cumsum(counts, dtype=np.int64, out=indptr[1:])

        if counts.max() > self.max_history_len:
            rank_in_u = np.arange(len(u_s), dtype=np.int64) - np.repeat(indptr[:-1], counts)
            keep = rank_in_u < self.max_history_len
            u_s = u_s[keep]
            i_s = i_s[keep]
            w_s = w_s[keep]
            k_s = k_s[keep]
            counts = np.minimum(counts, self.max_history_len)
            np.cumsum(counts, dtype=np.int64, out=indptr[1:])

        return (
            np.ascontiguousarray(indptr, dtype=np.int64),
            np.ascontiguousarray(i_s, dtype=np.int32),
            np.ascontiguousarray(w_s, dtype=np.float32),
            np.ascontiguousarray(k_s, dtype=np.int32),
        )


    def _dense_sample_weight(self):
        """per-interaction training weights as an array (all ones when the caller gave none)"""
        if self.sample_weight is not None:
            return self.sample_weight
        return np.ones(len(self._intx_u), dtype=np.float32)


    def _build_history_csr(self, user_history, target_user_index):
        """map raw user_history to target_user_index and self._item_index CSR arrays"""

        n_users = len(target_user_index)
        if user_history is None:
            if target_user_index.equals(self._user_index) and self._intx_u is not None:
                return self._build_history_csr_from_mapped(
                    self._intx_u,
                    self._intx_i,
                    self._dense_sample_weight(),
                    np.zeros(len(self._intx_u), dtype=np.int32),
                    n_users,
                )
            return (
                np.zeros(n_users + 1, dtype=np.int64),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int32),
            )

        u_col, i_col, w_arr, k_arr = self._parse_raw_history(user_history)
        u_mapped = _fast_get_indexer(target_user_index, u_col).astype(np.int32, copy=False)
        i_mapped = _fast_get_indexer(self._item_index, i_col).astype(np.int32, copy=False)
        return self._build_history_csr_from_mapped(u_mapped, i_mapped, w_arr, k_arr, n_users)


    def _resolve_last_items_for_users(
        self,
        last_items,
        target_user_index,
        known_indexer=None,
        fallback_uh_indptr=None,
        fallback_uh_indices=None,
    ):
        """resolve per-user sequential anchor item indices (int32, -1 if none) for target_user_index"""

        n_users = len(target_user_index)
        if not self.use_seq_anchor or n_users == 0:
            return np.zeros(0, dtype=np.int32)

        out = np.full(n_users, -1, dtype=np.int32)

        if known_indexer is not None and self._train_last_items is not None:
            km = known_indexer >= 0
            if km.any():
                out[km] = self._train_last_items[known_indexer[km]]

        if fallback_uh_indptr is not None and fallback_uh_indices is not None and len(fallback_uh_indices) > 0:
            missing = out < 0
            if missing.any():
                m_rows = np.flatnonzero(missing)
                has_h = fallback_uh_indptr[m_rows + 1] > fallback_uh_indptr[m_rows]
                if has_h.any():
                    h_rows = m_rows[has_h]
                    out[h_rows] = fallback_uh_indices[fallback_uh_indptr[h_rows]]

        if last_items is None:
            return out

        if isinstance(last_items, dict):
            if len(last_items) > 0:
                k_arr = np.asarray(list(last_items.keys()))
                v_arr = np.asarray(list(last_items.values()))
                u_pos = _fast_get_indexer(target_user_index, k_arr)
                i_pos = _fast_get_indexer(self._item_index, v_arr).astype(np.int32)
                valid = u_pos >= 0
                out[u_pos[valid]] = i_pos[valid]
            return out

        if isinstance(last_items, pd.Series):
            u_pos = _fast_get_indexer(target_user_index, last_items.index)
            i_pos = _fast_get_indexer(self._item_index, last_items.to_numpy()).astype(np.int32)
            valid = u_pos >= 0
            if valid.any():
                out[u_pos[valid]] = i_pos[valid]
                return out

        if isinstance(last_items, pd.DataFrame) or (isinstance(last_items, np.ndarray) and last_items.ndim == 2):
            u_col, i_col = self._extract_interaction_columns(last_items)
            u_pos = _fast_get_indexer(target_user_index, u_col)
            i_pos = _fast_get_indexer(self._item_index, i_col).astype(np.int32)
            valid = u_pos >= 0
            out[u_pos[valid]] = i_pos[valid]
            return out

        arr_1d = np.asarray(last_items)
        require(len(arr_1d) == n_users, "[last_items] 1D array length must match unique target users")
        out = np.ascontiguousarray(_fast_get_indexer(self._item_index, arr_1d), dtype=np.int32)
        return out


    def _fit_sparse_residual(self, n_threads=1):
        """compute pruned, FM-subspace-orthogonalized EASE sparse item-item residual S"""

        n_i = len(self.item_idx)
        if self.sparse_residual_k <= 0 or n_i <= 1:
            self._s_indptr = np.zeros(n_i + 1, dtype=np.int64)
            self._s_indices = np.zeros(0, dtype=np.int32)
            self._s_data = np.zeros(0, dtype=np.float32)
            return

        import scipy.sparse as sp

        if self._uh_indptr is not None and len(self._uh_indices) > 0:
            indptr = self._uh_indptr
            indices = self._uh_indices
            weights = self._uh_weights.astype(np.float32, copy=True)
        else:
            indptr = self._ui_indptr.astype(np.int64, copy=False)
            indices = self._ui_indices
            weights = np.ones(len(indices), dtype=np.float32)

        n_u = len(indptr) - 1
        if len(indices) == 0 or n_u == 0:
            self._s_indptr = np.zeros(n_i + 1, dtype=np.int64)
            self._s_indices = np.zeros(0, dtype=np.int32)
            self._s_data = np.zeros(0, dtype=np.float32)
            return

        # Deduplicate (u, i) across event types if needed and row-L2-normalize so heavy users don't dominate Gram matrix
        r_mat = sp.csr_matrix((weights, indices, indptr), shape=(n_u, n_i), dtype=np.float32)
        r_mat.sum_duplicates()
        row_sq = np.asarray(r_mat.power(2).sum(axis=1)).ravel()
        inv_row_norm = 1.0 / np.sqrt(np.maximum(row_sq, 1e-8)).astype(np.float32)
        r_norm = sp.diags(inv_row_norm) @ r_mat

        _, _ = self._compute_effective_items(n_threads=n_threads)
        v_basis = self._cached_v_i_unit if self._cached_v_i_unit is not None else self._cached_v_i_eff
        q_basis, _ = np.linalg.qr(np.asarray(v_basis, dtype=np.float32))

        k_keep = min(int(self.sparse_residual_k), n_i - 1)

        if n_i <= 20000:
            gram = (r_norm.T @ r_norm).toarray().astype(np.float32, copy=False)
            diag_idx = np.arange(n_i)
            gram[diag_idx, diag_idx] += np.float32(self.sparse_residual_l2)
            p_inv = np.linalg.inv(gram).astype(np.float32, copy=False)
            del gram
            p_diag = np.maximum(np.diag(p_inv), 1e-8)
            b_mat = -p_inv / p_diag[None, :]
            del p_inv
            b_mat[diag_idx, diag_idx] = 0.0

            # Project out low-rank FM target-item subspace: B_resid = B - (B @ Q) @ Q.T
            b_proj = b_mat @ q_basis
            b_mat -= b_proj @ q_basis.T
            b_mat[diag_idx, diag_idx] = 0.0

            # Prune each row to top-k_keep strictly positive residual weights
            part_idx = np.argpartition(-b_mat, kth=k_keep - 1, axis=1)[:, :k_keep]
            part_vals = np.take_along_axis(b_mat, part_idx, axis=1)
            del b_mat

            order = np.argsort(-part_vals, axis=1)
            top_idx = np.take_along_axis(part_idx, order, axis=1)
            top_vals = np.take_along_axis(part_vals, order, axis=1)

            pos_mask = top_vals > 1e-5
            row_counts = pos_mask.sum(axis=1, dtype=np.int64)
            s_indptr = np.zeros(n_i + 1, dtype=np.int64)
            np.cumsum(row_counts, dtype=np.int64, out=s_indptr[1:])
            self._s_indptr = np.ascontiguousarray(s_indptr, dtype=np.int64)
            self._s_indices = np.ascontiguousarray(top_idx[pos_mask], dtype=np.int32)
            self._s_data = np.ascontiguousarray(top_vals[pos_mask], dtype=np.float32)
        else:
            # Sparse fallback for massive catalogs (> 20k items)
            cooc = (r_norm.T @ r_norm).tocsr()
            cooc.setdiag(0.0)
            cooc.eliminate_zeros()
            self._s_indptr = np.ascontiguousarray(cooc.indptr, dtype=np.int64)
            self._s_indices = np.ascontiguousarray(cooc.indices, dtype=np.int32)
            self._s_data = np.ascontiguousarray(cooc.data / (cooc.data + self.sparse_residual_l2 * 0.01), dtype=np.float32)


    def _compute_effective_items(self, n_threads=1):
        """return cached effective item biases and item embeddings"""

        if self._cached_item_bias is None or self._cached_v_i_eff is None:
            if not self._is_v050_mode:
                if self.has_if and self._if_any:
                    self._cached_item_bias = _csr_matvec(
                        self._if_indptr, self._if_indices, self._if_data, self.w_if, base=self.w_i, n_threads=n_threads
                    )
                    self._cached_v_i_eff = _csr_dense_matmul(
                        self._if_indptr, self._if_indices, self._if_data, self.v_if, base=self.v_i, n_threads=n_threads
                    )
                else:
                    self._cached_item_bias = np.ascontiguousarray(self.w_i, dtype=np.float32)
                    self._cached_v_i_eff = np.ascontiguousarray(self.v_i, dtype=np.float32)
                self._cached_v_i_unit = self._cached_v_i_eff
            else:
                out_scale = float(1.0 / self.temperature) if self._use_cosine else 1.0
                b_eff, v_eff, v_unit = _compute_effective_items_v050(
                    self.w_i,
                    self.w_if,
                    self.v_i,
                    self.v_if,
                    self._if_indptr,
                    self._if_indices,
                    self._if_data,
                    self._if_any,
                    self.if_field,
                    self.if_is_dyn,
                    self.gamma_if,
                    self.item_log_freq,
                    self.a_gate,
                    self.b_gate,
                    self.use_field_gates,
                    self.use_freq_gate,
                    self.freq_gate_static_drop,
                    self._use_cosine,
                    out_scale,
                    n_threads,
                )
                self._cached_item_bias = b_eff
                self._cached_v_i_eff = v_eff
                self._cached_v_i_unit = v_unit
        return self._cached_item_bias, self._cached_v_i_eff


    def _compute_train_user_embeddings(self, n_threads=1):
        """return cached effective training-user embeddings"""

        if self._cached_train_u_emb is None:
            if not self._is_v050_mode:
                if self.has_uf and self._uf_any:
                    self._cached_train_u_emb = _csr_dense_matmul(
                        self._uf_indptr, self._uf_indices, self._uf_data, self.v_uf, base=self.v_u, n_threads=n_threads
                    )
                else:
                    self._cached_train_u_emb = np.ascontiguousarray(self.v_u, dtype=np.float32)
            else:
                n_u = len(self.user_idx)
                ident_map = np.arange(n_u, dtype=np.int32)
                uh_ptr = self._uh_indptr if self._uh_indptr is not None else np.zeros(n_u + 1, dtype=np.int64)
                uh_idx = self._uh_indices if self._uh_indices is not None else np.zeros(0, dtype=np.int32)
                uh_wt = self._uh_weights if self._uh_weights is not None else np.zeros(0, dtype=np.float32)
                uh_kd = self._uh_kinds if self._uh_kinds is not None else np.zeros(0, dtype=np.int32)
                last_it = self._train_last_items if self._train_last_items is not None else np.zeros(0, dtype=np.int32)
                max_h = max(self.max_history_len, int(np.diff(uh_ptr).max()) if len(uh_ptr) > 1 else 1)

                self._cached_train_u_emb = _encode_users_v050(
                    ident_map,
                    ident_map,
                    ident_map,
                    last_it,
                    self.v_u,
                    self.v_uf,
                    self._uf_indptr,
                    self._uf_indices,
                    self._uf_data,
                    self._uf_any,
                    self.uf_field,
                    self.gamma_uf,
                    uh_ptr,
                    uh_idx,
                    uh_wt,
                    uh_kd,
                    self.p_hist,
                    self.g_event,
                    self.p_seq,
                    self.g_seq,
                    self.c_proto,
                    self.n_interests,
                    max_h,
                    self.user_dropout < 1.0,
                    self.use_history,
                    self.use_seq_anchor,
                    self.use_field_gates,
                    self.n_event_types > 1,
                    self._use_cosine,
                    self.history_exponent,
                    self.interest_temperature,
                    n_threads,
                )
        return self._cached_train_u_emb


    def _prepare_inference_state_v050(
        self,
        users,
        cold_start='nan',
        user_features=None,
        interactions=None,
        sample_weight=None,
        user_history=None,
        last_items=None,
        fold_in_epochs=5,
        need_seen_items=False,
        n_threads=1,
    ):
        """vectorized v0.5.0 inference builder returning (u_emb, user_idx_obj, ui_ptr, ui_idx, item_bias, v_i_eff, uh_ptr, uh_idx, uh_wt, u_hist_map)"""

        if cold_start not in ('nan', 'drop', 'estimate'):
            raise ValueError("param [cold_start] must be set to 'nan', 'drop', or 'estimate'")
        require(isinstance(fold_in_epochs, int) and fold_in_epochs >= 0, "[fold_in_epochs] must be a non-negative integer")

        item_bias, v_i_eff = self._compute_effective_items(n_threads=n_threads)

        # Fast path: cached training users with no dynamic overrides
        if (
            user_features is None
            and interactions is None
            and user_history is None
            and last_items is None
            and cold_start in ('nan', 'drop')
        ):
            u_emb = self._compute_train_user_embeddings(n_threads=n_threads)
            uh_ptr = self._uh_indptr if self._uh_indptr is not None else np.zeros(len(self.user_idx) + 1, dtype=np.int64)
            uh_idx = self._uh_indices if self._uh_indices is not None else np.zeros(0, dtype=np.int32)
            uh_wt = self._uh_weights if self._uh_weights is not None else np.zeros(0, dtype=np.float32)
            return u_emb, self._user_index, self._ui_indptr, self._ui_indices, item_bias, v_i_eff, uh_ptr, uh_idx, uh_wt, None

        unique_idx = pd.Index(pd.unique(np.asarray(users)))

        uf_ids = None
        uf_indptr_in = None
        uf_indices_in = None
        uf_data_in = None
        if user_features is not None:
            if not self.has_uf:
                raise ValueError("model was not fit with [user_features]; cannot use [user_features] at inference")
            uf_ids, uf_indptr_in, uf_indices_in, uf_data_in, _, _ = self._extract_feature_csr(
                user_features,
                expected_columns=self.uf_columns,
                expected_width=self._n_uf,
                default_index=unique_idx,
                n_threads=n_threads,
                name='user_features',
            )

        intx_u_col = None
        intx_i_col = None
        sw_arr = None
        intx_user_idx = None
        if interactions is not None:
            if _is_scipy_sparse(interactions):
                coo = interactions.tocoo(copy=False)
                intx_u_col, intx_i_col = coo.row, coo.col
                sw_arr = np.ascontiguousarray(coo.data, dtype=np.float32) if sample_weight is None else np.asarray(get_data(sample_weight), dtype=np.float32)
            else:
                intx_u_col, intx_i_col = self._extract_interaction_columns(interactions)
                if sample_weight is not None:
                    sw_arr = np.asarray(get_data(sample_weight), dtype=np.float32)
                else:
                    sw_arr = np.ones(len(intx_u_col), dtype=np.float32)
            intx_user_idx = pd.Index(pd.unique(intx_u_col))

        uh_raw_tuple = None
        uh_user_idx = None
        if user_history is not None:
            uh_raw_tuple = self._parse_raw_history(user_history)
            uh_user_idx = pd.Index(pd.unique(uh_raw_tuple[0]))

        known_indexer = _fast_get_indexer(self._user_index, unique_idx)
        uf_indexer = _fast_get_indexer(uf_ids, unique_idx) if uf_ids is not None else None

        if cold_start == 'estimate':
            est_idx = unique_idx
            est_known_indexer = known_indexer.astype(np.int32)
            est_uf_indexer = uf_indexer
        else:
            est_mask = known_indexer >= 0
            if uf_indexer is not None:
                est_mask |= (uf_indexer >= 0)
            if intx_user_idx is not None:
                est_mask |= unique_idx.isin(intx_user_idx)
            if uh_user_idx is not None:
                est_mask |= unique_idx.isin(uh_user_idx)
            if last_items is not None:
                est_mask |= np.ones(len(unique_idx), dtype=bool)
            est_idx = unique_idx[est_mask]
            est_known_indexer = known_indexer[est_mask].astype(np.int32)
            est_uf_indexer = uf_indexer[est_mask] if uf_indexer is not None else None

        u_est_count = len(est_idx)
        if u_est_count == 0:
            empty_emb = np.zeros((0, self.n_interests * self.factors), dtype=np.float32)
            empty_ptr = np.zeros(1, dtype=np.int64)
            empty_idx = np.zeros(0, dtype=np.int32)
            empty_wt = np.zeros(0, dtype=np.float32)
            return empty_emb, est_idx, empty_ptr, empty_idx, item_bias, v_i_eff, empty_ptr, empty_idx, empty_wt, None

        # User feature CSR mapping
        if est_uf_indexer is None:
            uf_ptr_use = self._uf_indptr
            uf_idx_use = self._uf_indices
            uf_dat_use = self._uf_data
            uf_any_use = self._uf_any
            u_uf_map = np.ascontiguousarray(est_known_indexer, dtype=np.int32)
        else:
            # Build unified CSR rows for est_idx (incoming user_features override stored features)
            base_pos = np.where(est_uf_indexer >= 0, -1, est_known_indexer)
            b_ptr, b_idx, b_dat = _slice_csr_rows(self._uf_indptr, self._uf_indices, self._uf_data, base_pos)
            ov_mask = est_uf_indexer >= 0
            if ov_mask.any():
                ov_rows = np.flatnonzero(ov_mask)
                in_ptr, in_idx, in_dat = _slice_csr_rows(uf_indptr_in, uf_indices_in, uf_data_in, est_uf_indexer[ov_mask])
                uf_ptr_use, uf_idx_use, uf_dat_use = _overwrite_csr_rows(
                    b_ptr, b_idx, b_dat, u_est_count, ov_rows, in_ptr, in_idx, in_dat
                )
            else:
                uf_ptr_use, uf_idx_use, uf_dat_use = b_ptr, b_idx, b_dat
            uf_any_use = 1 if len(uf_dat_use) > 0 else 0
            u_uf_map = np.arange(u_est_count, dtype=np.int32)

        # User history CSR mapping
        if uh_raw_tuple is not None:
            u_m = _fast_get_indexer(est_idx, uh_raw_tuple[0]).astype(np.int32, copy=False)
            i_m = _fast_get_indexer(self._item_index, uh_raw_tuple[1]).astype(np.int32, copy=False)
            uh_ptr_use, uh_idx_use, uh_wt_use, uh_kd_use = self._build_history_csr_from_mapped(
                u_m, i_m, uh_raw_tuple[2], uh_raw_tuple[3], u_est_count
            )
            u_uh_map = np.arange(u_est_count, dtype=np.int32)
        elif intx_u_col is not None:
            u_parts, i_parts, w_parts, k_parts = [], [], [], []
            known_mask = est_known_indexer >= 0
            if known_mask.any() and self._uh_indptr is not None and len(self._uh_indices) > 0:
                s_ptr, s_idx, s_wt = _slice_csr_rows(self._uh_indptr, self._uh_indices, self._uh_weights, est_known_indexer)
                _, _, s_kd_f = _slice_csr_rows(
                    self._uh_indptr, self._uh_indices, self._uh_kinds.astype(np.float32), est_known_indexer
                )
                row_lens = np.diff(s_ptr)
                if len(s_idx) > 0:
                    u_parts.append(np.repeat(np.arange(u_est_count, dtype=np.int32), row_lens))
                    i_parts.append(s_idx)
                    w_parts.append(s_wt)
                    k_parts.append(s_kd_f.astype(np.int32))
            u_m = _fast_get_indexer(est_idx, intx_u_col).astype(np.int32)
            i_m = _fast_get_indexer(self._item_index, intx_i_col).astype(np.int32)
            val_m = (u_m >= 0) & (i_m >= 0)
            if val_m.any():
                u_parts.append(u_m[val_m])
                i_parts.append(i_m[val_m])
                w_parts.append(sw_arr[val_m] if not val_m.all() else sw_arr)
                k_parts.append(np.zeros(int(val_m.sum()), dtype=np.int32))
            if u_parts:
                uh_ptr_use, uh_idx_use, uh_wt_use, uh_kd_use = self._build_history_csr_from_mapped(
                    np.concatenate(u_parts), np.concatenate(i_parts), np.concatenate(w_parts), np.concatenate(k_parts), u_est_count
                )
            else:
                uh_ptr_use = np.zeros(u_est_count + 1, dtype=np.int64)
                uh_idx_use = np.zeros(0, dtype=np.int32)
                uh_wt_use = np.zeros(0, dtype=np.float32)
                uh_kd_use = np.zeros(0, dtype=np.int32)
            u_uh_map = np.arange(u_est_count, dtype=np.int32)
        else:
            uh_ptr_use = self._uh_indptr if self._uh_indptr is not None else np.zeros(len(self.user_idx) + 1, dtype=np.int64)
            uh_idx_use = self._uh_indices if self._uh_indices is not None else np.zeros(0, dtype=np.int32)
            uh_wt_use = self._uh_weights if self._uh_weights is not None else np.zeros(0, dtype=np.float32)
            uh_kd_use = self._uh_kinds if self._uh_kinds is not None else np.zeros(0, dtype=np.int32)
            u_uh_map = np.ascontiguousarray(est_known_indexer, dtype=np.int32)

        # Seen-item CSR for filter_previous or fold-in
        need_fold_in = (intx_u_col is not None and fold_in_epochs > 0 and not self.use_history and self.n_interests == 1 and self.user_dropout < 1.0)
        ui_indptr = np.zeros(u_est_count + 1, dtype=np.int32)
        ui_indices = np.zeros(0, dtype=np.int32)
        fold_pairs = None
        fold_sw = None

        if need_seen_items or need_fold_in:
            hist_u_parts, hist_i_parts = [], []
            known_mask = est_known_indexer >= 0
            if known_mask.any():
                known_local = np.flatnonzero(known_mask).astype(np.int32)
                g_loc, g_it = self._gather_csr_rows(self._ui_indptr, self._ui_indices, est_known_indexer[known_mask])
                if len(g_it) > 0:
                    hist_u_parts.append(known_local[g_loc])
                    hist_i_parts.append(g_it)
            if intx_u_col is not None:
                m_u = _fast_get_indexer(est_idx, intx_u_col)
                m_i = _fast_get_indexer(self._item_index, intx_i_col)
                val = (m_u >= 0) & (m_i >= 0)
                if val.any():
                    u_a = m_u[val].astype(np.int32)
                    i_a = m_i[val].astype(np.int32)
                    hist_u_parts.append(u_a)
                    hist_i_parts.append(i_a)
                    if need_fold_in:
                        fold_pairs = (np.ascontiguousarray(u_a, dtype=np.int32), np.ascontiguousarray(i_a, dtype=np.int32))
                        fold_sw = np.ascontiguousarray(sw_arr[val] if not val.all() else sw_arr, dtype=np.float32)
            if uh_raw_tuple is not None:
                m_u = _fast_get_indexer(est_idx, uh_raw_tuple[0])
                m_i = _fast_get_indexer(self._item_index, uh_raw_tuple[1])
                val = (m_u >= 0) & (m_i >= 0)
                if val.any():
                    hist_u_parts.append(m_u[val].astype(np.int32))
                    hist_i_parts.append(m_i[val].astype(np.int32))
            if hist_u_parts:
                all_u = np.concatenate(hist_u_parts) if len(hist_u_parts) > 1 else hist_u_parts[0]
                all_i = np.concatenate(hist_i_parts) if len(hist_i_parts) > 1 else hist_i_parts[0]
                ui_indptr, ui_indices = self._pairs_to_csr(all_u, all_i, u_est_count, len(self.item_idx))

        fb_ptr = uh_ptr_use if (uh_raw_tuple is not None or intx_u_col is not None) else None
        fb_idx = uh_idx_use if (uh_raw_tuple is not None or intx_u_col is not None) else None
        seq_arr = self._resolve_last_items_for_users(
            last_items, est_idx, known_indexer=est_known_indexer, fallback_uh_indptr=fb_ptr, fallback_uh_indices=fb_idx
        )

        v_u_use = self.v_u
        u_v_map = np.ascontiguousarray(est_known_indexer, dtype=np.int32)

        if fold_pairs is not None:
            v_u_batch = np.zeros((u_est_count, self.factors), dtype=np.float32)
            km = est_known_indexer >= 0
            if km.any():
                v_u_batch[km] = self.v_u[est_known_indexer[km]]
            max_h_tmp = max(self.max_history_len, int(np.diff(uh_ptr_use).max()) if len(uh_ptr_use) > 1 else 1)
            u_non_id = _encode_users_v050(
                np.full(u_est_count, -1, dtype=np.int32),
                u_uf_map,
                u_uh_map,
                seq_arr,
                self.v_u,
                self.v_uf,
                uf_ptr_use,
                uf_idx_use,
                uf_dat_use,
                uf_any_use,
                self.uf_field,
                self.gamma_uf,
                uh_ptr_use,
                uh_idx_use,
                uh_wt_use,
                uh_kd_use,
                self.p_hist,
                self.g_event,
                self.p_seq,
                self.g_seq,
                self.c_proto,
                1,
                max_h_tmp,
                False,
                self.use_history,
                self.use_seq_anchor,
                self.use_field_gates,
                self.n_event_types > 1,
                False,
                self.history_exponent,
                self.interest_temperature,
                n_threads,
            )
            _fold_in(
                fold_pairs[0],
                fold_pairs[1],
                fold_sw,
                ui_indptr,
                ui_indices,
                u_non_id,
                item_bias,
                v_i_eff,
                v_u_batch,
                self.alpha,
                self.learning_rate,
                self.learning_schedule,
                self.learning_exponent,
                self._max_samples_for_loss(),
                fold_in_epochs,
                1492,
                n_threads,
            )
            v_u_use = v_u_batch
            u_v_map = np.arange(u_est_count, dtype=np.int32)

        max_h = max(self.max_history_len, int(np.diff(uh_ptr_use).max()) if len(uh_ptr_use) > 1 else 1)
        u_emb = _encode_users_v050(
            u_v_map,
            u_uf_map,
            u_uh_map,
            seq_arr,
            v_u_use,
            self.v_uf,
            uf_ptr_use,
            uf_idx_use,
            uf_dat_use,
            uf_any_use,
            self.uf_field,
            self.gamma_uf,
            uh_ptr_use,
            uh_idx_use,
            uh_wt_use,
            uh_kd_use,
            self.p_hist,
            self.g_event,
            self.p_seq,
            self.g_seq,
            self.c_proto,
            self.n_interests,
            max_h,
            self.user_dropout < 1.0 or (fold_pairs is not None),
            self.use_history,
            self.use_seq_anchor,
            self.use_field_gates,
            self.n_event_types > 1,
            self._use_cosine,
            self.history_exponent,
            self.interest_temperature,
            n_threads,
        )

        # If uh_ptr_use is indexed by training user IDs rather than est_idx, pass u_uh_map so recommend() maps rows properly
        hist_map_for_est = None if (uh_raw_tuple is not None or intx_u_col is not None) else u_uh_map
        return u_emb, est_idx, ui_indptr, ui_indices, item_bias, v_i_eff, uh_ptr_use, uh_idx_use, uh_wt_use, hist_map_for_est


    def _prepare_inference_state(
        self,
        users,
        cold_start='nan',
        user_features=None,
        interactions=None,
        sample_weight=None,
        fold_in_epochs=5,
        need_seen_items=False,
        n_threads=1
    ):
        """vectorized CSR builder for effective user embeddings, index lookup, and seen-item history"""

        if cold_start not in ('nan', 'drop', 'estimate'):
            raise ValueError("param [cold_start] must be set to 'nan', 'drop', or 'estimate'")
        require(isinstance(fold_in_epochs, int) and fold_in_epochs >= 0, "[fold_in_epochs] must be a non-negative integer")

        item_bias, v_i_eff = self._compute_effective_items(n_threads=n_threads)

        # Fast path: standard inference using cached embeddings and CSR history directly
        if user_features is None and interactions is None and cold_start in ('nan', 'drop'):
            u_emb = self._compute_train_user_embeddings(n_threads=n_threads)
            return u_emb, self._user_index, self._ui_indptr, self._ui_indices, item_bias, v_i_eff

        # Dynamic / unseen-user path: pure CSR feature projection (zero dense (U, P_uf) allocation!)
        uf_ids = None
        uf_indptr_in = None
        uf_indices_in = None
        uf_data_in = None
        if user_features is not None:
            if not self.has_uf:
                raise ValueError("model was not fit with [user_features]; cannot use [user_features] at inference")
            unique_users_for_default = pd.Index(pd.unique(np.asarray(users)))
            uf_ids, uf_indptr_in, uf_indices_in, uf_data_in, _, _ = self._extract_feature_csr(
                user_features,
                expected_columns=self.uf_columns,
                expected_width=self._n_uf,
                default_index=unique_users_for_default,
                n_threads=n_threads,
                name='user_features',
            )

        intx_u_col = None
        intx_i_col = None
        sw_arr = None
        intx_user_idx = None
        if interactions is not None:
            if _is_scipy_sparse(interactions):
                coo = interactions.tocoo(copy=False)
                intx_u_col, intx_i_col = coo.row, coo.col
                sw_arr = np.ascontiguousarray(coo.data, dtype=np.float32) if sample_weight is None else np.asarray(get_data(sample_weight), dtype=np.float32)
            else:
                intx_u_col, intx_i_col = self._extract_interaction_columns(interactions)
                if sample_weight is not None:
                    require(isinstance(sample_weight, (np.ndarray, pd.Series)), "[sample_weight] must be np.ndarray or pd.series")
                    require(sample_weight.ndim == 1, "[sample_weight] must a vector (ndim=1)")
                    require(len(sample_weight) == len(intx_u_col), "[sample_weight] must have the same length as [interactions]")
                    sw_arr = np.asarray(get_data(sample_weight), dtype=np.float32)
                else:
                    sw_arr = np.ones(len(intx_u_col), dtype=np.float32)
            intx_user_idx = pd.Index(pd.unique(intx_u_col))

        unique_idx = pd.Index(pd.unique(np.asarray(users)))
        known_indexer = _fast_get_indexer(self._user_index, unique_idx)
        uf_indexer = _fast_get_indexer(uf_ids, unique_idx) if uf_ids is not None else None

        if cold_start == 'estimate':
            est_idx = unique_idx
            est_known_indexer = known_indexer
            est_uf_indexer = uf_indexer
        else:
            est_mask = known_indexer >= 0
            if uf_indexer is not None:
                est_mask |= (uf_indexer >= 0)
            if intx_user_idx is not None:
                est_mask |= unique_idx.isin(intx_user_idx)

            est_idx = unique_idx[est_mask]
            est_known_indexer = known_indexer[est_mask]
            est_uf_indexer = uf_indexer[est_mask] if uf_indexer is not None else None

        u_est_count = len(est_idx)
        if u_est_count == 0:
            empty_emb = np.zeros((0, self.factors), dtype=np.float32)
            empty_indptr = np.zeros(1, dtype=np.int32)
            empty_indices = np.zeros(0, dtype=np.int32)
            return empty_emb, est_idx, empty_indptr, empty_indices, item_bias, v_i_eff

        v_u_batch = np.zeros((u_est_count, self.factors), dtype=np.float32)
        u_feat_emb = np.zeros((u_est_count, self.factors), dtype=np.float32)

        known_mask = est_known_indexer >= 0
        if known_mask.any():
            train_u_ids = est_known_indexer[known_mask]
            v_u_batch[known_mask] = self.v_u[train_u_ids]
            if self.has_uf and self._uf_any:
                use_stored_uf = known_mask & (~(est_uf_indexer >= 0) if est_uf_indexer is not None else True)
                if use_stored_uf.any():
                    stored_u_ids = est_known_indexer[use_stored_uf]
                    s_ptr, s_idx, s_dat = _slice_csr_rows(self._uf_indptr, self._uf_indices, self._uf_data, stored_u_ids)
                    if len(s_dat) > 0:
                        u_feat_emb[use_stored_uf] = _csr_dense_matmul(
                            s_ptr, s_idx, s_dat, self.v_uf, n_threads=n_threads
                        )

        if est_uf_indexer is not None and self.has_uf:
            uf_mask = est_uf_indexer >= 0
            if uf_mask.any():
                in_ptr, in_idx, in_dat = _slice_csr_rows(
                    uf_indptr_in, uf_indices_in, uf_data_in, est_uf_indexer[uf_mask]
                )
                if len(in_dat) > 0:
                    u_feat_emb[uf_mask] = _csr_dense_matmul(
                        in_ptr, in_idx, in_dat, self.v_uf, n_threads=n_threads
                    )
                else:
                    u_feat_emb[uf_mask] = 0.0

        need_item_history = need_seen_items or (intx_u_col is not None and fold_in_epochs > 0)
        ui_indptr = np.zeros(u_est_count + 1, dtype=np.int32)
        ui_indices = np.zeros(0, dtype=np.int32)

        if need_item_history:
            hist_u_parts = []
            hist_i_parts = []

            if known_mask.any():
                known_local = np.flatnonzero(known_mask).astype(np.int32)
                gathered_local, gathered_items = self._gather_csr_rows(
                    self._ui_indptr, self._ui_indices, est_known_indexer[known_mask]
                )
                if len(gathered_items) > 0:
                    hist_u_parts.append(known_local[gathered_local])
                    hist_i_parts.append(gathered_items)

            fold_pairs = None
            fold_sw = None
            if intx_u_col is not None:
                mapped_u = _fast_get_indexer(est_idx, intx_u_col)
                mapped_i = _fast_get_indexer(self._item_index, intx_i_col)
                valid = (mapped_u >= 0) & (mapped_i >= 0)
                if valid.any():
                    u_arr = mapped_u[valid].astype(np.int32)
                    i_arr = mapped_i[valid].astype(np.int32)
                    hist_u_parts.append(u_arr)
                    hist_i_parts.append(i_arr)
                    if fold_in_epochs > 0:
                        fold_pairs = (np.ascontiguousarray(u_arr), np.ascontiguousarray(i_arr))
                        fold_sw = np.ascontiguousarray(sw_arr[valid] if not valid.all() else sw_arr, dtype=np.float32)

            if hist_u_parts:
                all_u = np.concatenate(hist_u_parts) if len(hist_u_parts) > 1 else hist_u_parts[0]
                all_i = np.concatenate(hist_i_parts) if len(hist_i_parts) > 1 else hist_i_parts[0]
                ui_indptr, ui_indices = self._pairs_to_csr(all_u, all_i, u_est_count, len(self.item_idx))

            if fold_pairs is not None:
                _fold_in(
                    fold_pairs[0],
                    fold_pairs[1],
                    fold_sw,
                    ui_indptr,
                    ui_indices,
                    u_feat_emb,
                    item_bias,
                    v_i_eff,
                    v_u_batch,
                    self.alpha,
                    self.learning_rate,
                    self.learning_schedule,
                    self.learning_exponent,
                    self._max_samples_for_loss(),
                    fold_in_epochs,
                    1492,
                    n_threads
                )

        u_emb = np.ascontiguousarray(v_u_batch + u_feat_emb, dtype=np.float32)
        return u_emb, est_idx, ui_indptr, ui_indices, item_bias, v_i_eff


    def _extract_pairwise_context_bonuses(self, pairwise_context):
        """parse pairwise_context into (raw_users, raw_items, scalar_bonuses_f32)"""

        if isinstance(pairwise_context, tuple) and len(pairwise_context) == 2:
            ctx_pairs, ctx_vals = pairwise_context
            u_col, i_col = self._extract_interaction_columns(ctx_pairs)
            vals = np.asarray(ctx_vals, dtype=np.float32)
            if vals.ndim == 1:
                bonuses = vals
            else:
                require(self.w_ctx is not None and vals.shape[1] == len(self.w_ctx), "2D [pairwise_context] features require calling fit_pairwise_context() first with matching width")
                bonuses = vals @ self.w_ctx
            return u_col, i_col, np.ascontiguousarray(bonuses, dtype=np.float32)

        if isinstance(pairwise_context, pd.DataFrame):
            require(pairwise_context.shape[1] >= 3, "[pairwise_context] DataFrame must have [user_id, item_id, ...]")
            u_col = pairwise_context.iloc[:, 0].to_numpy()
            i_col = pairwise_context.iloc[:, 1].to_numpy()
            feat_mat = pairwise_context.iloc[:, 2:].to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(pairwise_context)
            require(arr.ndim == 2 and arr.shape[1] >= 3, "[pairwise_context] array must have >= 3 columns")
            u_col = arr[:, 0]
            i_col = arr[:, 1]
            feat_mat = np.asarray(arr[:, 2:], dtype=np.float32)

        if self.w_ctx is not None and feat_mat.shape[1] == len(self.w_ctx):
            bonuses = feat_mat @ self.w_ctx
        elif feat_mat.shape[1] == 1:
            bonuses = feat_mat[:, 0]
        else:
            raise ValueError("[pairwise_context] has multiple feature columns but [w_ctx] has not been fit")
        return u_col, i_col, np.ascontiguousarray(bonuses, dtype=np.float32)


    def _build_context_csr_for_valid_users(self, pairwise_context, valid_user_ids):
        """build CSR (indptr_i64, indices_i32, bonus_f32) aligned to valid_user_ids rows"""

        n_rows = len(valid_user_ids)
        if pairwise_context is None or n_rows == 0:
            return (
                np.zeros(n_rows + 1, dtype=np.int64),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
            )

        u_col, i_col, bonuses = self._extract_pairwise_context_bonuses(pairwise_context)
        target_idx = pd.Index(valid_user_ids)
        # Handle duplicate requested users in valid_user_ids by mapping via unique index then slicing
        if target_idx.has_duplicates:
            uniq_u = pd.Index(pd.unique(valid_user_ids))
            u_m = _fast_get_indexer(uniq_u, u_col)
            i_m = _fast_get_indexer(self._item_index, i_col)
            val = (u_m >= 0) & (i_m >= 0) & (bonuses != 0.0)
            if not val.any():
                return np.zeros(n_rows + 1, dtype=np.int64), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
            u_v = u_m[val].astype(np.int32)
            i_v = i_m[val].astype(np.int32)
            b_v = bonuses[val]
            order = np.argsort(u_v, kind='mergesort')
            u_v, i_v, b_v = u_v[order], i_v[order], b_v[order]
            counts = np.bincount(u_v, minlength=len(uniq_u))
            u_ptr = np.zeros(len(uniq_u) + 1, dtype=np.int64)
            np.cumsum(counts, dtype=np.int64, out=u_ptr[1:])
            row_map = _fast_get_indexer(uniq_u, valid_user_ids)
            return _slice_csr_rows(u_ptr, i_v, b_v, row_map)

        u_m = _fast_get_indexer(target_idx, u_col)
        i_m = _fast_get_indexer(self._item_index, i_col)
        val = (u_m >= 0) & (i_m >= 0) & (bonuses != 0.0)
        if not val.any():
            return np.zeros(n_rows + 1, dtype=np.int64), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

        u_v = u_m[val].astype(np.int32)
        i_v = i_m[val].astype(np.int32)
        b_v = bonuses[val]
        order = np.argsort(u_v, kind='mergesort')
        u_v = u_v[order]
        i_v = np.ascontiguousarray(i_v[order], dtype=np.int32)
        b_v = np.ascontiguousarray(b_v[order], dtype=np.float32)
        counts = np.bincount(u_v, minlength=n_rows)
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(counts, dtype=np.int64, out=indptr[1:])
        return indptr, i_v, b_v


    # -------------------------------
    # begin public method definitions
    # -------------------------------


    def fit(
        self,
        interactions,
        user_features=None,
        item_features=None,
        sample_weight=None,
        epochs=1,
        verbose=False,
        n_jobs=None,
        seed=None,
        user_history=None,
        prev_items=None,
        last_items=None,
        margin_signal=None,
        user_field_ids=None,
        item_field_ids=None,
        item_dynamic_mask=None,
    ):
        """clear previous model state and learn new model weights using the input data"""

        self._reset_state()
        self.fit_partial(
            interactions,
            user_features=user_features,
            item_features=item_features,
            sample_weight=sample_weight,
            epochs=epochs,
            verbose=verbose,
            freeze_items=False,
            n_jobs=n_jobs,
            seed=seed,
            user_history=user_history,
            prev_items=prev_items,
            last_items=last_items,
            margin_signal=margin_signal,
            user_field_ids=user_field_ids,
            item_field_ids=item_field_ids,
            item_dynamic_mask=item_dynamic_mask,
        )
        return self


    def fit_partial(
        self,
        interactions,
        user_features=None,
        item_features=None,
        sample_weight=None,
        epochs=1,
        verbose=False,
        freeze_items=False,
        n_jobs=None,
        seed=None,
        user_history=None,
        prev_items=None,
        last_items=None,
        margin_signal=None,
        user_field_ids=None,
        item_field_ids=None,
        item_dynamic_mask=None,
    ):
        """learn or update model weights using the input data and resuming from the current model state"""

        require(isinstance(epochs, int) and epochs >= 1, "[epochs] must be a positive integer")
        require(isinstance(verbose, bool), "[verbose] must be a boolean value")
        require(isinstance(freeze_items, bool), "[freeze_items] must be a boolean value")

        if freeze_items and not self.is_fit:
            raise AssertionError("you must fit the model prior to calling fit_partial(..., freeze_items=True)")

        eff_threads = self._resolve_n_jobs(n_jobs)
        eff_seed = 1492 if seed is None else int(seed)
        self._invalidate_caches()

        was_fit = self.is_fit
        if was_fit:
            self._init_interactions(interactions, sample_weight, freeze_items=freeze_items, seed=eff_seed)
            self._init_features(user_features, item_features, n_threads=eff_threads)
        else:
            self._init_all(
                interactions,
                user_features,
                item_features,
                sample_weight,
                n_threads=eff_threads,
                seed=eff_seed,
                user_field_ids=user_field_ids,
                item_field_ids=item_field_ids,
                item_dynamic_mask=item_dynamic_mask,
            )

        if self.sample_weight is not None:
            require(np.isfinite(self.sample_weight.sum(dtype=np.float64)), "[sample_weight] contains NaN or infinite values")
        max_samples = self._max_samples_for_loss()

        if not self._is_v050_mode:
            if freeze_items:
                item_bias, v_i_eff = self._compute_effective_items(n_threads=eff_threads)
                if self.has_uf and self._uf_any:
                    u_feat_emb = _csr_dense_matmul(
                        self._uf_indptr, self._uf_indices, self._uf_data, self.v_uf, n_threads=eff_threads
                    )
                else:
                    u_feat_emb = np.zeros((len(self.user_idx), self.factors), dtype=np.float32)

                _fold_in(
                    self._intx_u,
                    self._intx_i,
                    self.sample_weight,
                    self._ui_indptr,
                    self._ui_indices,
                    u_feat_emb,
                    item_bias,
                    v_i_eff,
                    self.v_u,
                    self.alpha,
                    self.learning_rate,
                    self.learning_schedule,
                    self.learning_exponent,
                    max_samples,
                    epochs,
                    eff_seed,
                    eff_threads
                )
                self._cached_train_u_emb = None
            else:
                _fit_csr(
                    self._intx_u,
                    self._intx_i,
                    self.sample_weight,
                    self._ui_indptr,
                    self._ui_indices,
                    self._uf_indptr,
                    self._uf_indices,
                    self._uf_data,
                    self._uf_any,
                    self._if_indptr,
                    self._if_indices,
                    self._if_data,
                    self._if_any,
                    self.w_i,
                    self.w_if,
                    self.v_u,
                    self.v_i,
                    self.v_uf,
                    self.v_if,
                    self.alpha,
                    self.beta,
                    self.learning_rate,
                    self.learning_schedule,
                    self.learning_exponent,
                    max_samples,
                    epochs,
                    verbose,
                    self.user_dropout,
                    eff_seed,
                    eff_threads,
                    self.feature_reg == 'uniform',
                    self.normalize_steps,
                    self.dense_updates == 'batched'
                )
                self._invalidate_caches()

            self.is_fit = True
            return self

        # --- RankFM v0.5.0 Training & Incremental State Path ---
        n_u = len(self.user_idx)
        need_uh = (self.use_history or self.target_attention_topk > 0 or self.sparse_residual_k > 0)

        if need_uh:
            if user_history is not None or not was_fit or self._uh_indptr is None:
                self._uh_indptr, self._uh_indices, self._uh_weights, self._uh_kinds = self._build_history_csr(
                    user_history, self._user_index
                )
            else:
                # Merge existing history CSR with incremental interactions
                old_lens = np.diff(self._uh_indptr)
                old_u = np.repeat(np.arange(len(old_lens), dtype=np.int32), old_lens)
                comb_u = np.concatenate([old_u, self._intx_u])
                comb_i = np.concatenate([self._uh_indices, self._intx_i])
                comb_w = np.concatenate([self._uh_weights, self._dense_sample_weight()])
                comb_k = np.concatenate([self._uh_kinds, np.zeros(len(self._intx_u), dtype=np.int32)])
                self._uh_indptr, self._uh_indices, self._uh_weights, self._uh_kinds = self._build_history_csr_from_mapped(
                    comb_u, comb_i, comb_w, comb_k, n_u
                )
        else:
            self._uh_indptr = np.zeros(n_u + 1, dtype=np.int64)
            self._uh_indices = np.zeros(0, dtype=np.int32)
            self._uh_weights = np.zeros(0, dtype=np.float32)
            self._uh_kinds = np.zeros(0, dtype=np.int32)

        if self.use_seq_anchor:
            if prev_items is not None:
                prev_raw = np.asarray(get_data(prev_items) if isinstance(prev_items, pd.Series) else prev_items)
                require(len(prev_raw) == len(self._intx_u), "[prev_items] length must match [interactions]")
                prev_arr = np.ascontiguousarray(_fast_get_indexer(self._item_index, prev_raw), dtype=np.int32)
            else:
                prev_arr = np.zeros(0, dtype=np.int32)

            if last_items is not None:
                self._train_last_items = self._resolve_last_items_for_users(last_items, self._user_index)
            else:
                last_map = self._train_last_items if (was_fit and self._train_last_items is not None and len(self._train_last_items) == n_u) else np.full(n_u, -1, dtype=np.int32)
                if len(self._intx_u) > 0:
                    # the training rows are in the caller's order in v0.5 mode, so each user's last row is their latest item
                    rev_u, rev_pos = np.unique(self._intx_u[::-1], return_index=True)
                    last_pos = (len(self._intx_u) - 1) - rev_pos
                    last_map[rev_u] = self._intx_i[last_pos]
                self._train_last_items = np.ascontiguousarray(last_map, dtype=np.int32)
        else:
            prev_arr = np.zeros(0, dtype=np.int32)
            self._train_last_items = np.zeros(0, dtype=np.int32)

        if margin_signal is not None:
            ms_raw = np.asarray(get_data(margin_signal) if isinstance(margin_signal, pd.Series) else margin_signal, dtype=np.float32)
            require(len(ms_raw) == len(self._intx_u), "[margin_signal] length must match [interactions]")
            margin_sig_arr = np.ascontiguousarray(ms_raw, dtype=np.float32)
        elif self.weight_margin_scale != 0.0:
            margin_sig_arr = np.ascontiguousarray(np.log1p(np.maximum(self._dense_sample_weight(), 0.0)), dtype=np.float32)
        else:
            margin_sig_arr = np.zeros(0, dtype=np.float32)

        if freeze_items:
            if self.user_dropout < 1.0 and self.n_interests == 1:
                item_bias, v_i_eff = self._compute_effective_items(n_threads=eff_threads)
                max_h_f = max(self.max_history_len, int(np.diff(self._uh_indptr).max()) if len(self._uh_indptr) > 1 else 1)
                ident_u = np.arange(n_u, dtype=np.int32)
                u_non_id = _encode_users_v050(
                    np.full(n_u, -1, dtype=np.int32),
                    ident_u,
                    ident_u,
                    self._train_last_items if self.use_seq_anchor else np.zeros(0, dtype=np.int32),
                    self.v_u,
                    self.v_uf,
                    self._uf_indptr,
                    self._uf_indices,
                    self._uf_data,
                    self._uf_any,
                    self.uf_field,
                    self.gamma_uf,
                    self._uh_indptr,
                    self._uh_indices,
                    self._uh_weights,
                    self._uh_kinds,
                    self.p_hist,
                    self.g_event,
                    self.p_seq,
                    self.g_seq,
                    self.c_proto,
                    1,
                    max_h_f,
                    False,
                    self.use_history,
                    self.use_seq_anchor,
                    self.use_field_gates,
                    self.n_event_types > 1,
                    False,
                    self.history_exponent,
                    self.interest_temperature,
                    eff_threads,
                )
                _fold_in(
                    self._intx_u,
                    self._intx_i,
                    self.sample_weight,
                    self._ui_indptr,
                    self._ui_indices,
                    u_non_id,
                    item_bias,
                    v_i_eff,
                    self.v_u,
                    self.alpha,
                    self.learning_rate,
                    self.learning_schedule,
                    self.learning_exponent,
                    max_samples,
                    epochs,
                    eff_seed,
                    eff_threads,
                )
            self._cached_train_u_emb = None
        else:
            max_h = max(self.max_history_len, int(np.diff(self._uh_indptr).max()) if len(self._uh_indptr) > 1 else 1)
            _fit_v050(
                self._intx_u,
                self._intx_i,
                self.sample_weight,
                margin_sig_arr,
                prev_arr,
                self._ui_indptr,
                self._ui_indices,
                self._uf_indptr,
                self._uf_indices,
                self._uf_data,
                self._uf_any,
                self.uf_field,
                self._if_indptr,
                self._if_indices,
                self._if_data,
                self._if_any,
                self.if_field,
                self.if_is_dyn,
                self._uh_indptr,
                self._uh_indices,
                self._uh_weights,
                self._uh_kinds,
                self.item_log_freq,
                self._alias_prob,
                self._alias_idx,
                self._log_q,
                self.w_i,
                self.w_if,
                self.v_u,
                self.v_i,
                self.v_uf,
                self.v_if,
                self.p_hist,
                self.p_seq,
                self.g_event,
                self.g_seq,
                self.gamma_uf,
                self.gamma_if,
                self.a_gate,
                self.b_gate,
                self.c_proto,
                self.n_interests,
                max_h,
                self._loss_mode_code(),
                self._use_cosine,
                self.neg_sampling == 'popularity',
                self.use_history,
                self.use_seq_anchor,
                self.use_field_gates,
                self.n_event_types > 1,
                self.use_freq_gate,
                self.alpha,
                self.beta,
                self.learning_rate,
                self.learning_schedule,
                self.learning_exponent,
                max_samples,
                epochs,
                verbose,
                self.user_dropout,
                self.temperature,
                self.logq_weight,
                self.margin,
                self.weight_margin_scale,
                self.neg_weight,
                self.history_exponent,
                self.interest_temperature,
                self.head_agg_temperature,
                self.freq_gate_static_drop,
                eff_seed,
                eff_threads,
                self.normalize_steps,
            )
            self._invalidate_caches()
            if self.sparse_residual_k > 0:
                self._fit_sparse_residual(n_threads=eff_threads)

        self.is_fit = True
        return self


    def fit_pairwise_context(
        self,
        pairs,
        context_features,
        labels,
        l2_reg=10.0,
        max_iter=15,
        user_features=None,
        user_history=None,
        last_items=None,
        n_jobs=None,
    ):
        """fit L2-regularized Newton-Raphson logistic weights [w_ctx] for explicit (user, item) context features on top of frozen RankFM scores"""

        require(self.is_fit, "you must fit the RankFM model prior to calling fit_pairwise_context()")
        x_ctx = np.asarray(context_features, dtype=np.float64)
        if x_ctx.ndim == 1:
            x_ctx = x_ctx[:, None]
        y = np.asarray(labels, dtype=np.float64).ravel()
        require(len(x_ctx) == len(y) and len(y) > 0, "[context_features] and [labels] must have matching positive length")

        prev_w_ctx = self.w_ctx
        self.w_ctx = None
        try:
            base_scores = self.predict(
                pairs,
                cold_start='estimate',
                user_features=user_features,
                user_history=user_history,
                last_items=last_items,
                n_jobs=n_jobs,
            ).astype(np.float64)
        finally:
            self.w_ctx = prev_w_ctx

        valid = np.isfinite(base_scores)
        x_v = x_ctx[valid]
        y_v = np.clip(y[valid], 0.0, 1.0)
        off_v = base_scores[valid]

        d_ctx = x_v.shape[1]
        w = np.zeros(d_ctx, dtype=np.float64)
        reg_eye = float(l2_reg) * np.eye(d_ctx, dtype=np.float64)

        for _ in range(int(max_iter)):
            logits = np.clip(off_v + x_v @ w, -30.0, 30.0)
            prob = 1.0 / (1.0 + np.exp(-logits))
            grad = x_v.T @ (prob - y_v) + float(l2_reg) * w
            wt = np.maximum(prob * (1.0 - prob), 1e-6)
            hess = (x_v * wt[:, None]).T @ x_v + reg_eye
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.linalg.norm(step) < 1e-6:
                break

        self.w_ctx = np.ascontiguousarray(w, dtype=np.float32)
        return self.w_ctx


    def embed_users(
        self,
        users,
        user_features=None,
        interactions=None,
        sample_weight=None,
        user_history=None,
        last_items=None,
        fold_in_epochs=5,
        cold_start='nan',
        n_jobs=None
    ):
        """compute effective latent representations for seen and/or unseen users without mutating model state"""

        require(getattr(users, '__iter__', False), "[users] must be an iterable (e.g. list, array, series)")
        require(self.is_fit, "you must fit the model prior to embedding users")

        eff_threads = self._resolve_n_jobs(n_jobs)
        users_arr = np.asarray(list(users) if not isinstance(users, (np.ndarray, pd.Series, pd.Index, list)) else users)

        if not self._is_v050_mode:
            u_emb, user_index_obj, _, _, _, _ = self._prepare_inference_state(
                users_arr,
                cold_start=cold_start,
                user_features=user_features,
                interactions=interactions,
                sample_weight=sample_weight,
                fold_in_epochs=fold_in_epochs,
                need_seen_items=False,
                n_threads=eff_threads
            )
        else:
            u_emb, user_index_obj, _, _, _, _, _, _, _, _ = self._prepare_inference_state_v050(
                users_arr,
                cold_start=cold_start,
                user_features=user_features,
                interactions=interactions,
                sample_weight=sample_weight,
                user_history=user_history,
                last_items=last_items,
                fold_in_epochs=fold_in_epochs,
                need_seen_items=False,
                n_threads=eff_threads,
            )

        mapped = _fast_get_indexer(user_index_obj, users_arr)
        valid = mapped >= 0
        out_width = self.factors * self.n_interests

        if cold_start == 'drop':
            return pd.DataFrame(u_emb[mapped[valid]], index=users_arr[valid])

        out = np.full((len(users_arr), out_width), np.nan, dtype=np.float32)
        if valid.any():
            out[valid] = u_emb[mapped[valid]]
        return pd.DataFrame(out, index=users_arr)


    def export_mips_embeddings(
        self,
        users=None,
        user_features=None,
        interactions=None,
        sample_weight=None,
        user_history=None,
        last_items=None,
        fold_in_epochs=5,
        cold_start='drop',
        n_jobs=None
    ):
        """export augmented (F + 1)-dimensional float32 user & item embeddings for sub-linear ANN / MIPS engines (FAISS, ScaNN)

        When n_interests == 1: user_mips has shape (U, F + 1) and <user_mips[u], item_mips[i]> == utility(u, i).
        When n_interests > 1:  user_mips has shape (U, M, F + 1) for multi-probe MIPS retrieval.
        Returns (user_ids, user_mips, item_ids, item_mips).
        """

        require(self.is_fit, "you must fit the model prior to exporting MIPS embeddings")
        eff_threads = self._resolve_n_jobs(n_jobs)
        item_bias, v_i_eff = self._compute_effective_items(n_threads=eff_threads)

        F = self.factors
        M = self.n_interests

        item_mips = np.empty((len(item_bias), F + 1), dtype=np.float32)
        item_mips[:, :F] = v_i_eff
        item_mips[:, F] = item_bias
        item_ids = self._item_index.to_numpy()

        if users is None and user_features is None and interactions is None and user_history is None and last_items is None:
            u_emb = self._compute_train_user_embeddings(n_threads=eff_threads)
            u_ids_out = self._user_index.to_numpy()
        else:
            target_users = self._user_index.to_numpy() if users is None else np.asarray(
                list(users) if not isinstance(users, (np.ndarray, pd.Series, pd.Index, list)) else users
            )
            emb_df = self.embed_users(
                target_users,
                user_features=user_features,
                interactions=interactions,
                sample_weight=sample_weight,
                user_history=user_history,
                last_items=last_items,
                fold_in_epochs=fold_in_epochs,
                cold_start=cold_start,
                n_jobs=eff_threads,
            )
            u_emb = emb_df.to_numpy(dtype=np.float32, copy=False)
            u_ids_out = emb_df.index.to_numpy()

        if M == 1:
            user_mips = np.empty((len(u_emb), F + 1), dtype=np.float32)
            user_mips[:, :F] = u_emb
            user_mips[:, F] = np.where(np.isnan(u_emb[:, 0]), np.nan, 1.0) if len(u_emb) > 0 else 1.0
            return u_ids_out, user_mips, item_ids, item_mips

        u_3d = u_emb.reshape(len(u_emb), M, F)
        user_mips = np.empty((len(u_emb), M, F + 1), dtype=np.float32)
        user_mips[:, :, :F] = u_3d
        user_mips[:, :, F] = np.where(np.isnan(u_3d[:, :, 0]), np.nan, 1.0) if len(u_emb) > 0 else 1.0
        return u_ids_out, user_mips, item_ids, item_mips


    def predict(
        self,
        pairs,
        cold_start='nan',
        user_features=None,
        interactions=None,
        sample_weight=None,
        user_history=None,
        last_items=None,
        pairwise_context=None,
        fold_in_epochs=5,
        n_jobs=None
    ):
        """calculate the predicted pointwise utilities for all (user, item) pairs"""

        require(self.is_fit, "you must fit the model prior to generating predictions")
        eff_threads = self._resolve_n_jobs(n_jobs)
        u_col, i_col = self._extract_interaction_columns(pairs)

        if not self._is_v050_mode and pairwise_context is None:
            u_emb, user_index_obj, _, _, item_bias, v_i_eff = self._prepare_inference_state(
                u_col,
                cold_start=cold_start,
                user_features=user_features,
                interactions=interactions,
                sample_weight=sample_weight,
                fold_in_epochs=fold_in_epochs,
                need_seen_items=False,
                n_threads=eff_threads
            )
            u_idx = np.ascontiguousarray(_fast_get_indexer(user_index_obj, u_col), dtype=np.int32)
            i_idx = np.ascontiguousarray(_fast_get_indexer(self._item_index, i_col), dtype=np.int32)
            scores = _predict(u_idx, i_idx, u_emb, item_bias, v_i_eff, eff_threads)
        else:
            if self._is_v050_mode:
                u_emb, user_index_obj, _, _, item_bias, v_i_eff, uh_ptr, uh_idx, uh_wt, hist_map = self._prepare_inference_state_v050(
                    u_col,
                    cold_start=cold_start,
                    user_features=user_features,
                    interactions=interactions,
                    sample_weight=sample_weight,
                    user_history=user_history,
                    last_items=last_items,
                    fold_in_epochs=fold_in_epochs,
                    need_seen_items=False,
                    n_threads=eff_threads,
                )
            else:
                u_emb, user_index_obj, _, _, item_bias, v_i_eff = self._prepare_inference_state(
                    u_col,
                    cold_start=cold_start,
                    user_features=user_features,
                    interactions=interactions,
                    sample_weight=sample_weight,
                    fold_in_epochs=fold_in_epochs,
                    need_seen_items=False,
                    n_threads=eff_threads,
                )
                uh_ptr = np.zeros(1, dtype=np.int64)
                uh_idx = np.zeros(0, dtype=np.int32)
                uh_wt = np.zeros(0, dtype=np.float32)
                hist_map = None

            u_idx = np.ascontiguousarray(_fast_get_indexer(user_index_obj, u_col), dtype=np.int32)
            i_idx = np.ascontiguousarray(_fast_get_indexer(self._item_index, i_col), dtype=np.int32)

            if self.n_interests == 1:
                scores = _predict(u_idx, i_idx, u_emb, item_bias, v_i_eff, eff_threads)
            else:
                n_p = len(u_idx)
                scores = np.full(n_p, np.nan, dtype=np.float32)
                val_p = (u_idx >= 0) & (i_idx >= 0)
                if val_p.any():
                    u_3d = u_emb.reshape(len(u_emb), self.n_interests, self.factors)
                    uh_v = u_3d[u_idx[val_p]]
                    iv_v = v_i_eff[i_idx[val_p]]
                    dots = np.einsum('nmf,nf->nm', uh_v, iv_v)
                    if self.head_agg_temperature <= 0.0:
                        agg = dots.max(axis=1)
                    else:
                        mx = dots.max(axis=1, keepdims=True)
                        agg = mx[:, 0] + self.head_agg_temperature * np.log(
                            np.maximum(np.exp((dots - mx) / self.head_agg_temperature).sum(axis=1), 1e-12)
                        )
                    scores[val_p] = (agg + item_bias[i_idx[val_p]]).astype(np.float32)

            # Optional sparse residual S contribution on requested pairs
            if (
                self.sparse_residual_k > 0
                and self.sparse_residual_weight != 0.0
                and self._s_indices is not None
                and len(self._s_indices) > 0
                and len(uh_idx) > 0
            ):
                val_p = np.flatnonzero((u_idx >= 0) & (i_idx >= 0))
                if len(val_p) > 0:
                    import scipy.sparse as sp
                    n_i = len(item_bias)
                    s_csr = sp.csr_matrix((self._s_data, self._s_indices, self._s_indptr), shape=(n_i, n_i), dtype=np.float32)
                    h_rows = hist_map[u_idx[val_p]] if hist_map is not None else u_idx[val_p]
                    has_hr = h_rows >= 0
                    if has_hr.any():
                        sub_p = val_p[has_hr]
                        sub_hr = h_rows[has_hr]
                        uh_csr = sp.csr_matrix((uh_wt, uh_idx, uh_ptr), shape=(len(uh_ptr) - 1, n_i), dtype=np.float32)
                        # Group by unique h_row to avoid redundant sparse vector products
                        uniq_hr, inv_hr = np.unique(sub_hr, return_inverse=True)
                        res_mat = (uh_csr[uniq_hr] @ s_csr).tocsr()
                        # Extract (inv_hr, i_idx[sub_p]) entries
                        res_vals = np.asarray(res_mat[inv_hr, i_idx[sub_p]]).ravel()
                        scores[sub_p] += np.float32(self.sparse_residual_weight) * res_vals.astype(np.float32)

            # Optional DIN-lite target-aware attention on requested pairs
            if (
                self.target_attention_topk > 0
                and self.target_attention_weight != 0.0
                and self.p_hist is not None
                and len(self.p_hist) > 0
                and len(uh_idx) > 0
            ):
                val_p = np.flatnonzero((u_idx >= 0) & (i_idx >= 0))
                if len(val_p) > 0:
                    v_unit = self._cached_v_i_unit if self._cached_v_i_unit is not None else v_i_eff
                    inv_sqrt_f = np.float32(1.0 / np.sqrt(max(1, self.factors)))
                    h_rows = hist_map[u_idx[val_p]] if hist_map is not None else u_idx[val_p]
                    for idx_local, p_pos in enumerate(val_p):
                        hr = int(h_rows[idx_local])
                        if hr < 0:
                            continue
                        lo = int(uh_ptr[hr])
                        hi = min(int(uh_ptr[hr + 1]), lo + 16)
                        if hi <= lo:
                            continue
                        h_items = uh_idx[lo:hi]
                        att_lg = (self.p_hist[h_items] @ v_unit[i_idx[p_pos]]) * inv_sqrt_f
                        mx_lg = att_lg.max()
                        att_w = np.exp(att_lg - mx_lg) * uh_wt[lo:hi]
                        den = float(att_w.sum())
                        if den > 1e-12:
                            scores[p_pos] += np.float32(self.target_attention_weight * float((att_w @ att_lg) / den))

            # Optional explicit pairwise context bonus on matching (u, i) pairs
            if pairwise_context is not None:
                ctx_u, ctx_i, ctx_b = self._extract_pairwise_context_bonuses(pairwise_context)
                cu_m = _fast_get_indexer(user_index_obj, ctx_u)
                ci_m = _fast_get_indexer(self._item_index, ctx_i)
                c_val = (cu_m >= 0) & (ci_m >= 0)
                if c_val.any():
                    n_i = len(item_bias)
                    c_codes = cu_m[c_val].astype(np.int64) * np.int64(n_i) + ci_m[c_val].astype(np.int64)
                    c_bon = ctx_b[c_val]
                    c_ord = np.argsort(c_codes, kind='mergesort')
                    c_codes_s = c_codes[c_ord]
                    c_bon_s = c_bon[c_ord]
                    chg = np.empty(len(c_codes_s), dtype=bool)
                    chg[0] = True
                    if len(c_codes_s) > 1:
                        chg[1:] = c_codes_s[1:] != c_codes_s[:-1]
                    starts = np.flatnonzero(chg)
                    u_codes = c_codes_s[starts]
                    u_bon = np.add.reduceat(c_bon_s, starts)

                    val_p = np.flatnonzero((u_idx >= 0) & (i_idx >= 0))
                    if len(val_p) > 0:
                        q_codes = u_idx[val_p].astype(np.int64) * np.int64(n_i) + i_idx[val_p].astype(np.int64)
                        pos = np.searchsorted(u_codes, q_codes)
                        hit = (pos < len(u_codes)) & (u_codes[np.minimum(pos, len(u_codes) - 1)] == q_codes)
                        if hit.any():
                            scores[val_p[hit]] += u_bon[pos[hit]].astype(np.float32)

        if cold_start in ('nan', 'estimate'):
            return scores
        elif cold_start == 'drop':
            return scores[~np.isnan(scores)]
        else:
            raise ValueError("param [cold_start] must be set to 'nan', 'drop', or 'estimate'")


    def recommend(
        self,
        users,
        n_items=10,
        filter_previous=False,
        cold_start='nan',
        user_features=None,
        interactions=None,
        sample_weight=None,
        user_history=None,
        last_items=None,
        pairwise_context=None,
        fold_in_epochs=5,
        n_jobs=None
    ):
        """calculate the topN items for each user using batched BLAS SGEMM and parallel nogil C min-heaps"""

        require(getattr(users, '__iter__', False), "[users] must be an iterable (e.g. list, array, series)")
        require(self.is_fit, "you must fit the model prior to generating recommendations")

        eff_threads = self._resolve_n_jobs(n_jobs)
        users_arr = np.asarray(list(users) if not isinstance(users, (np.ndarray, pd.Series, pd.Index, list)) else users)

        if not self._is_v050_mode and pairwise_context is None:
            u_emb, user_index_obj, ui_indptr, ui_indices, item_bias, v_i_eff = self._prepare_inference_state(
                users_arr,
                cold_start=cold_start,
                user_features=user_features,
                interactions=interactions,
                sample_weight=sample_weight,
                fold_in_epochs=fold_in_epochs,
                need_seen_items=filter_previous,
                n_threads=eff_threads
            )
            req_u_idx = _fast_get_indexer(user_index_obj, users_arr)
            valid_mask = req_u_idx >= 0
            valid_u_idx = np.ascontiguousarray(req_u_idx[valid_mask], dtype=np.int32)
            n_valid = len(valid_u_idx)
            n_catalog = len(item_bias)
            out_valid_idx = np.full((n_valid, n_items), -1, dtype=np.int32)

            if n_valid > 0 and n_catalog > 0:
                chunk_size = max(64, min(4096, (32 * 1024 * 1024) // max(1, n_catalog * 4)))
                v_i_eff_T = v_i_eff.T
                with _blas_thread_cap():
                    for start in range(0, n_valid, chunk_size):
                        end = min(start + chunk_size, n_valid)
                        chunk_u = valid_u_idx[start:end]
                        scores_chunk = np.ascontiguousarray(np.dot(u_emb[chunk_u], v_i_eff_T), dtype=np.float32)
                        _select_topk_chunk(
                            scores_chunk,
                            item_bias,
                            chunk_u,
                            ui_indptr,
                            ui_indices,
                            n_items,
                            filter_previous,
                            out_valid_idx[start:end],
                            eff_threads
                        )
        else:
            if self._is_v050_mode:
                u_emb, user_index_obj, ui_indptr, ui_indices, item_bias, v_i_eff, uh_ptr, uh_idx, uh_wt, hist_map = self._prepare_inference_state_v050(
                    users_arr,
                    cold_start=cold_start,
                    user_features=user_features,
                    interactions=interactions,
                    sample_weight=sample_weight,
                    user_history=user_history,
                    last_items=last_items,
                    fold_in_epochs=fold_in_epochs,
                    need_seen_items=filter_previous,
                    n_threads=eff_threads,
                )
            else:
                u_emb, user_index_obj, ui_indptr, ui_indices, item_bias, v_i_eff = self._prepare_inference_state(
                    users_arr,
                    cold_start=cold_start,
                    user_features=user_features,
                    interactions=interactions,
                    sample_weight=sample_weight,
                    fold_in_epochs=fold_in_epochs,
                    need_seen_items=filter_previous,
                    n_threads=eff_threads,
                )
                uh_ptr = np.zeros(1, dtype=np.int64)
                uh_idx = np.zeros(0, dtype=np.int32)
                uh_wt = np.zeros(0, dtype=np.float32)
                hist_map = None

            req_u_idx = _fast_get_indexer(user_index_obj, users_arr)
            valid_mask = req_u_idx >= 0
            valid_u_idx = np.ascontiguousarray(req_u_idx[valid_mask], dtype=np.int32)
            n_valid = len(valid_u_idx)
            n_catalog = len(item_bias)
            out_valid_idx = np.full((n_valid, n_items), -1, dtype=np.int32)

            if n_valid > 0 and n_catalog > 0:
                M = self.n_interests
                F = self.factors
                s_ptr = self._s_indptr if self._s_indptr is not None else np.zeros(1, dtype=np.int64)
                s_idx = self._s_indices if self._s_indices is not None else np.zeros(0, dtype=np.int32)
                s_dat = self._s_data if self._s_data is not None else np.zeros(0, dtype=np.float32)
                p_hist_use = self.p_hist if self.p_hist is not None else np.zeros((0, F), dtype=np.float32)
                v_unit_use = self._cached_v_i_unit if self._cached_v_i_unit is not None else v_i_eff

                if pairwise_context is not None:
                    ctx_full_ptr, ctx_full_idx, ctx_full_bon = self._build_context_csr_for_valid_users(
                        pairwise_context, users_arr[valid_mask]
                    )
                    has_ctx = len(ctx_full_idx) > 0
                else:
                    ctx_full_ptr = None
                    ctx_full_idx = None
                    ctx_full_bon = None
                    has_ctx = False

                empty_ptr_1 = np.zeros(1, dtype=np.int64)
                empty_i32 = np.zeros(0, dtype=np.int32)
                empty_f32 = np.zeros(0, dtype=np.float32)

                chunk_size = max(32, min(4096, (32 * 1024 * 1024) // max(1, n_catalog * M * 4)))
                v_i_eff_T = v_i_eff.T

                with _blas_thread_cap():
                    for start in range(0, n_valid, chunk_size):
                        end = min(start + chunk_size, n_valid)
                        B = end - start
                        chunk_u = valid_u_idx[start:end]
                        chunk_hist_u = np.ascontiguousarray(hist_map[chunk_u], dtype=np.int32) if hist_map is not None else chunk_u

                        u_chunk_2d = u_emb[chunk_u].reshape(B * M, F)
                        scores_chunk = np.ascontiguousarray(np.dot(u_chunk_2d, v_i_eff_T), dtype=np.float32)

                        if has_ctx:
                            p0 = int(ctx_full_ptr[start])
                            p1 = int(ctx_full_ptr[end])
                            if p1 > p0:
                                c_ptr = np.ascontiguousarray(ctx_full_ptr[start:end + 1] - p0, dtype=np.int64)
                                c_idx = ctx_full_idx[p0:p1]
                                c_bon = ctx_full_bon[p0:p1]
                            else:
                                c_ptr, c_idx, c_bon = empty_ptr_1, empty_i32, empty_f32
                        else:
                            c_ptr, c_idx, c_bon = empty_ptr_1, empty_i32, empty_f32

                        _select_topk_v050_chunk(
                            scores_chunk,
                            item_bias,
                            chunk_u,
                            chunk_hist_u,
                            ui_indptr,
                            ui_indices,
                            uh_ptr,
                            uh_idx,
                            uh_wt,
                            s_ptr,
                            s_idx,
                            s_dat,
                            c_ptr,
                            c_idx,
                            c_bon,
                            p_hist_use,
                            v_unit_use,
                            M,
                            n_items,
                            filter_previous,
                            self.head_agg_temperature,
                            self.sparse_residual_weight,
                            self.target_attention_topk,
                            self.target_attention_weight,
                            out_valid_idx[start:end],
                            eff_threads,
                        )

        item_id_vals = self._item_index.to_numpy()
        all_items_valid = (out_valid_idx >= 0).all() if n_valid > 0 else True

        if cold_start == 'drop':
            target_users = users_arr[valid_mask]
            if n_valid == 0:
                return pd.DataFrame(np.empty((0, n_items), dtype=item_id_vals.dtype), index=target_users)
            if all_items_valid:
                return pd.DataFrame(item_id_vals[out_valid_idx], index=target_users)
            mapped = np.where(out_valid_idx >= 0, item_id_vals[np.maximum(out_valid_idx, 0)], np.nan)
            return pd.DataFrame(mapped, index=target_users).dropna(how='any')

        if valid_mask.all() and all_items_valid:
            return pd.DataFrame(item_id_vals[out_valid_idx], index=users_arr)

        # Partial missing users or exhausted item catalog -> fill missing entries with NaN
        out_obj = np.full((len(users_arr), n_items), np.nan, dtype=object if item_id_vals.dtype == object else np.float64)
        if n_valid > 0:
            if all_items_valid:
                out_obj[valid_mask] = item_id_vals[out_valid_idx]
            else:
                valid_cells = out_valid_idx >= 0
                row_mapped = np.full((n_valid, n_items), np.nan, dtype=out_obj.dtype)
                row_mapped[valid_cells] = item_id_vals[out_valid_idx[valid_cells]]
                out_obj[valid_mask] = row_mapped
        return pd.DataFrame(out_obj, index=users_arr)


    def recommend_batches(
        self,
        users,
        batch_size=10000,
        n_items=10,
        filter_previous=False,
        cold_start='nan',
        user_features=None,
        interactions=None,
        sample_weight=None,
        user_history=None,
        last_items=None,
        pairwise_context=None,
        fold_in_epochs=5,
        as_dataframe=True,
        n_jobs=None
    ):
        """stream topN recommendations in bounded-memory batches for 10M-100M+ user inference

        When as_dataframe=True (default), yields a pd.DataFrame per batch identical to recommend().
        When as_dataframe=False, yields (batch_user_ids_ndarray, batch_item_ids_ndarray) tuples.
        """

        require(getattr(users, '__iter__', False), "[users] must be an iterable (e.g. list, array, series)")
        require(isinstance(batch_size, int) and batch_size >= 1, "[batch_size] must be a positive integer")
        require(self.is_fit, "you must fit the model prior to generating recommendations")

        users_arr = np.asarray(list(users) if not isinstance(users, (np.ndarray, pd.Series, pd.Index, list)) else users)
        total = len(users_arr)

        # Index the side inputs once up front. Handing the raw inputs to recommend() per batch would re-scan
        # all of them for every batch, which costs O(n_batches * total input size) across the whole stream.
        if user_features is not None:
            if not self.has_uf:
                raise ValueError("model was not fit with [user_features]; cannot use [user_features] at inference")
            user_features = _FeatureCSR(*self._extract_feature_csr(
                user_features,
                expected_columns=self.uf_columns,
                expected_width=self._n_uf,
                default_index=pd.Index(pd.unique(users_arr)),
                n_threads=self._resolve_n_jobs(n_jobs),
                name='user_features',
            ))

        intx_u = intx_i = intx_sw = intx_users = intx_indptr = intx_order = None
        if interactions is not None:
            if _is_scipy_sparse(interactions):
                coo = interactions.tocoo(copy=False)
                intx_u, intx_i = coo.row, coo.col
                intx_sw = coo.data if sample_weight is None else get_data(sample_weight)
            else:
                intx_u, intx_i = self._extract_interaction_columns(interactions)
                intx_sw = None if sample_weight is None else get_data(sample_weight)
            if intx_sw is not None:
                require(len(intx_sw) == len(intx_u), "[sample_weight] must have the same length as [interactions]")
                intx_sw = np.asarray(intx_sw, dtype=np.float32)

            # group interaction rows by user: [intx_order] lists row positions user by user, [intx_indptr] bounds each user
            intx_users = pd.Index(pd.unique(intx_u))
            user_codes = intx_users.get_indexer(intx_u)
            intx_order = np.argsort(user_codes, kind='stable')
            intx_indptr = np.zeros(len(intx_users) + 1, dtype=np.int64)
            np.cumsum(np.bincount(user_codes, minlength=len(intx_users)), out=intx_indptr[1:])

        for start in range(0, total, batch_size):
            batch_u = users_arr[start:start + batch_size]
            batch_intx = None
            batch_sw = None
            if interactions is not None:
                groups = intx_users.get_indexer(pd.unique(batch_u))
                _, rows = self._gather_csr_rows(intx_indptr, intx_order, groups[groups >= 0])
                rows = np.sort(rows)
                batch_intx = pd.DataFrame({'user': intx_u[rows], 'item': intx_i[rows]})
                batch_sw = None if intx_sw is None else intx_sw[rows]

            df_batch = self.recommend(
                batch_u,
                n_items=n_items,
                filter_previous=filter_previous,
                cold_start=cold_start,
                user_features=user_features,
                interactions=batch_intx,
                sample_weight=batch_sw,
                user_history=user_history,
                last_items=last_items,
                pairwise_context=pairwise_context,
                fold_in_epochs=fold_in_epochs,
                n_jobs=n_jobs,
            )
            if as_dataframe:
                yield df_batch
            else:
                yield df_batch.index.to_numpy(), df_batch.to_numpy()


    def similar_items(self, item_id, n_items=10):
        """find the most similar items wrt latent factor space representation"""

        require(self.is_fit, "you must fit the model prior to generating similarities")
        item_pos = int(_fast_get_indexer(self._item_index, [item_id])[0])
        require(item_pos >= 0, "you must select an [item_id] present in the training data")

        _, lr_all_items = self._compute_effective_items()
        sims = np.dot(lr_all_items, lr_all_items[item_pos])
        sims[item_pos] = -np.inf

        item_id_vals = self._item_index.to_numpy()
        k = min(n_items, len(sims) - 1)
        if k <= 0:
            return np.array([], dtype=item_id_vals.dtype)
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return item_id_vals[top_idx]


    def similar_users(
        self,
        user_id,
        n_users=10,
        user_features=None,
        interactions=None,
        sample_weight=None,
        user_history=None,
        last_items=None,
        fold_in_epochs=5
    ):
        """find the most similar training users wrt latent factor space representation (supports unseen query users)"""

        require(self.is_fit, "you must fit the model prior to generating similarities")

        user_pos = int(_fast_get_indexer(self._user_index, [user_id])[0])
        in_training = user_pos >= 0
        if not in_training and user_features is None and interactions is None and user_history is None and last_items is None:
            raise AssertionError("you must select a [user_id] present in the training data or supply [user_features]/[interactions]/[user_history]")

        lr_all_users = self._compute_train_user_embeddings()

        if in_training and user_features is None and interactions is None and user_history is None and last_items is None:
            lr_user = lr_all_users[user_pos]
        else:
            emb_df = self.embed_users(
                [user_id],
                user_features=user_features,
                interactions=interactions,
                sample_weight=sample_weight,
                user_history=user_history,
                last_items=last_items,
                fold_in_epochs=fold_in_epochs,
                cold_start='nan'
            )
            lr_user = emb_df.iloc[0].to_numpy()
            require(not np.isnan(lr_user).any(), "user_id={} could not be embedded from supplied features/interactions".format(user_id))

        sims = np.dot(lr_all_users, lr_user)
        if user_pos >= 0:
            sims[user_pos] = -np.inf

        user_id_vals = self._user_index.to_numpy()
        max_avail = len(sims) - (1 if user_pos >= 0 else 0)
        k = min(n_users, max_avail)
        if k <= 0:
            return np.array([], dtype=user_id_vals.dtype)
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return user_id_vals[top_idx]
