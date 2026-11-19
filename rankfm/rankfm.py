"""
rankfm main modeling class
"""

import math
import os
from collections import namedtuple

import numpy as np
import pandas as pd

from rankfm._rankfm import (
    _csr_dense_matmul,
    _csr_matvec,
    _fit_csr,
    _fold_in,
    _predict,
    _select_topk_chunk,
    dense_to_csr,
)
from rankfm.utils import get_data


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
    if (
        isinstance(index_obj, pd.RangeIndex)
        and index_obj.start == 0
        and index_obj.step == 1
        and q.dtype.kind in ('i', 'u')
    ):
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
        feature_reg='stochastic'
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
        :return: None
        """

        assert isinstance(factors, int) and factors >= 1, "[factors] must be a positive integer"
        assert isinstance(loss, str) and loss in ('bpr', 'warp'), "[loss] must be in ('bpr', 'warp')"
        assert isinstance(max_samples, int) and max_samples > 0, "[max_samples] must be a positive integer"
        assert isinstance(alpha, float) and alpha > 0.0, "[alpha] must be a positive float"
        assert isinstance(beta, float) and beta > 0.0, "[beta] must be a positive float"
        assert isinstance(sigma, float) and sigma > 0.0, "[sigma] must be a positive float"
        assert isinstance(learning_rate, float) and learning_rate > 0.0, "[learning_rate] must be a positive float"
        assert isinstance(learning_schedule, str) and learning_schedule in ('constant', 'invscaling'), "[learning_schedule] must be in ('constant', 'invscaling')"
        assert isinstance(learning_exponent, float) and learning_exponent > 0.0, "[learning_exponent] must be a positive float"
        assert isinstance(user_dropout, (int, float)) and 0.0 <= float(user_dropout) < 1.0, "[user_dropout] must be a float in [0.0, 1.0)"
        assert isinstance(n_jobs, int) and n_jobs != 0, "[n_jobs] must be a non-zero integer (e.g. 1, 4, -1)"
        assert isinstance(feature_reg, str) and feature_reg in ('stochastic', 'uniform'), "[feature_reg] must be in ('stochastic', 'uniform')"

        self.factors = factors
        self.loss = loss
        self.max_samples = max_samples
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.learning_schedule = learning_schedule
        self.learning_exponent = learning_exponent
        self.user_dropout = float(user_dropout)
        self.n_jobs = n_jobs
        self.feature_reg = feature_reg

        self._reset_state()


    # --------------------------------
    # begin private method definitions
    # --------------------------------


    def _resolve_n_jobs(self, n_jobs=None):
        nj = self.n_jobs if n_jobs is None else int(n_jobs)
        assert nj != 0, "[n_jobs] must be non-zero"
        if nj < 0:
            return max(1, _available_cpus() + 1 + nj)
        return nj


    def _invalidate_caches(self):
        """clear cached effective embeddings and materialized dense/dict views"""

        self._cached_item_bias = None
        self._cached_v_i_eff = None
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

        # user/item interactions and importance weights
        self.interactions = None
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
            assert isinstance(features, (np.ndarray, pd.DataFrame)), (
                "[{}] must be np.ndarray, pd.DataFrame, scipy.sparse matrix, or (ids, matrix) tuple".format(name)
            )
            assert features.shape[1] >= 2, "[{}] must contain an ID column plus at least one feature column".format(name)

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
                csr.sort_indices()
            n_cols = int(csr.shape[1])
            if expected_width is not None and n_cols != expected_width:
                raise ValueError("[{}] has {} feature columns, expected {}".format(name, n_cols, expected_width))
            if n_cols < 1:
                raise AssertionError("[{}] must have at least 1 feature column".format(name))

            indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
            indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
            data = np.ascontiguousarray(csr.data, dtype=np.float32)

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
        return raw_ids, indptr, indices, data, n_cols, cols_out


    @staticmethod
    def _extract_interaction_columns(interactions):
        """extract (user_col_array, item_col_array) without full DataFrame copies"""

        assert isinstance(interactions, (np.ndarray, pd.DataFrame)), "[interactions] must be np.ndarray, pd.DataFrame, or scipy.sparse matrix"
        assert interactions.shape[1] == 2, "[interactions] should be: [user_id, item_id]"

        if isinstance(interactions, pd.DataFrame):
            return interactions.iloc[:, 0].to_numpy(), interactions.iloc[:, 1].to_numpy()
        return interactions[:, 0], interactions[:, 1]


    def _init_all(self, interactions, user_features=None, item_features=None, sample_weight=None, n_threads=1, seed=1492):
        """index the interaction data and user/item features and initialize model weights"""

        if _is_scipy_sparse(interactions):
            # Zero-Pandas sparse matrix fast path: rows = 0..U-1, cols = 0..I-1, data = sample_weight (unless overridden)
            n_u, n_i = int(interactions.shape[0]), int(interactions.shape[1])
            assert n_u >= 1 and n_i >= 1, "sparse [interactions] must have positive shape (n_users, n_items)"
            self._user_index = pd.RangeIndex(0, n_u)
            self._item_index = pd.RangeIndex(0, n_i)
            self.user_idx = np.arange(n_u, dtype=np.int32)
            self.item_idx = np.arange(n_i, dtype=np.int32)

            coo = interactions.tocoo(copy=False)
            u_arr = np.ascontiguousarray(coo.row, dtype=np.int32)
            i_arr = np.ascontiguousarray(coo.col, dtype=np.int32)
            if sample_weight is not None:
                assert isinstance(sample_weight, (np.ndarray, pd.Series)), "[sample_weight] must be np.ndarray or pd.series"
                assert sample_weight.ndim == 1 and len(sample_weight) == len(u_arr), "[sample_weight] must match sparse nnz"
                self.sample_weight = np.ascontiguousarray(get_data(sample_weight), dtype=np.float32)
            else:
                self.sample_weight = np.ascontiguousarray(coo.data, dtype=np.float32)

            self._ui_indptr, self._ui_indices = self._pairs_to_csr(u_arr, i_arr, n_u, n_i)
            self._user_items_cache = None
            self.interactions = np.ascontiguousarray(np.column_stack([u_arr, i_arr]), dtype=np.int32)
        else:
            u_col, i_col = self._extract_interaction_columns(interactions)
            self._user_index, u_mapped = _fast_unique_and_map(u_col)
            self._item_index, i_mapped = _fast_unique_and_map(i_col)
            self.user_idx = np.arange(len(self._user_index), dtype=np.int32)
            self.item_idx = np.arange(len(self._item_index), dtype=np.int32)

            if sample_weight is not None:
                assert isinstance(sample_weight, (np.ndarray, pd.Series)), "[sample_weight] must be np.ndarray or pd.series"
                assert sample_weight.ndim == 1, "[sample_weight] must a vector (ndim=1)"
                assert len(sample_weight) == len(u_col), "[sample_weight] must have the same length as [interactions]"
                self.sample_weight = np.ascontiguousarray(get_data(sample_weight), dtype=np.float32)
            else:
                self.sample_weight = np.ones(len(u_mapped), dtype=np.float32)

            self._ui_indptr, self._ui_indices = self._pairs_to_csr(
                u_mapped, i_mapped, len(self.user_idx), len(self.item_idx)
            )
            self._user_items_cache = None
            self.interactions = np.ascontiguousarray(np.column_stack([u_mapped, i_mapped]), dtype=np.int32)

        self._init_features(user_features, item_features, n_threads=n_threads)
        self._init_weights(user_features, item_features, seed=seed)


    def _expand_users_for_partial(self, u_col, i_col, freeze_items=False, seed=1492):
        """append any previously unseen users in interactions to the model's user index and state arrays"""

        valid_item_mask = _fast_get_indexer(self._item_index, i_col) >= 0
        incoming_users = pd.Index(pd.unique(u_col[valid_item_mask]))
        new_users = incoming_users.difference(self._user_index).sort_values().to_numpy()

        if len(new_users) == 0:
            return

        n_new = len(new_users)
        self._user_index = pd.Index(np.concatenate([self._user_index.to_numpy(), new_users]))
        self._user_id_series = None
        self._user_to_index_series = None
        self.user_idx = np.arange(len(self._user_index), dtype=np.int32)

        if self.has_uf or freeze_items:
            new_v_u = np.zeros((n_new, self.factors), dtype=np.float32)
        else:
            rng = np.random.default_rng(seed)
            new_v_u = rng.normal(loc=0, scale=self.sigma, size=(n_new, self.factors)).astype(np.float32)

        self.v_u = np.ascontiguousarray(np.vstack([self.v_u, new_v_u]), dtype=np.float32)

        # Extend CSR user-feature indptr with empty rows in O(n_new) (zero dense matrix allocation!)
        last_ptr = int(self._uf_indptr[-1])
        pad_ptr = np.full(n_new, last_ptr, dtype=np.int64)
        self._uf_indptr = np.ascontiguousarray(np.concatenate([self._uf_indptr, pad_ptr]), dtype=np.int64)
        self._dense_x_uf_cache = None


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
            assert isinstance(sample_weight, (np.ndarray, pd.Series)), "[sample_weight] must be np.ndarray or pd.series"
            assert sample_weight.ndim == 1, "[sample_weight] must a vector (ndim=1)"
            assert len(sample_weight) == len(u_col), "[sample_weight] must have the same length as [interactions]"

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
            self.sample_weight = np.ascontiguousarray(sw_raw[valid_mask] if not valid_mask.all() else sw_raw, dtype=np.float32)
        else:
            self.sample_weight = np.ones(len(u_arr), dtype=np.float32)

        n_users = len(self.user_idx)
        n_items = len(self.item_idx)

        if self.is_fit and self._ui_indptr is not None:
            old_counts = np.diff(self._ui_indptr)
            old_u = np.repeat(np.arange(len(old_counts), dtype=np.int32), old_counts)
            comb_u = np.concatenate([old_u, u_arr])
            comb_i = np.concatenate([self._ui_indices, i_arr])
            self._ui_indptr, self._ui_indices = self._pairs_to_csr(comb_u, comb_i, n_users, n_items)
        else:
            self._ui_indptr, self._ui_indices = self._pairs_to_csr(u_arr, i_arr, n_users, n_items)

        self._user_items_cache = None
        self.interactions = np.ascontiguousarray(np.column_stack([u_arr, i_arr]), dtype=np.int32)


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
                    self._uf_indptr, self._uf_indices, self._uf_data = indptr, indices, data
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
                    self._if_indptr, self._if_indices, self._if_data = indptr, indices, data
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


    def _init_weights(self, user_features=None, item_features=None, seed=1492):
        """initialize model weights given user/item and user-feature/item-feature shapes

        weights are drawn from a generator seeded with [seed], so fit(seed=...) fixes the starting point as well as the sampling
        """

        rng = np.random.default_rng(seed)

        self.w_i = np.zeros(len(self.item_idx), dtype=np.float32)
        self.w_if = np.zeros(self._n_if, dtype=np.float32)

        self.v_u = rng.normal(loc=0, scale=self.sigma, size=(len(self.user_idx), self.factors)).astype(np.float32)
        self.v_i = rng.normal(loc=0, scale=self.sigma, size=(len(self.item_idx), self.factors)).astype(np.float32)

        if user_features is not None:
            scale = (self.alpha / self.beta) * self.sigma
            self.v_uf = rng.normal(loc=0, scale=scale, size=(self._n_uf, self.factors)).astype(np.float32)
        else:
            self.v_uf = np.zeros((self._n_uf, self.factors), dtype=np.float32)

        if item_features is not None:
            scale = (self.alpha / self.beta) * self.sigma
            self.v_if = rng.normal(loc=0, scale=scale, size=(self._n_if, self.factors)).astype(np.float32)
        else:
            self.v_if = np.zeros((self._n_if, self.factors), dtype=np.float32)


    def _max_samples_for_loss(self):
        if self.loss == 'bpr':
            return 1
        elif self.loss == 'warp':
            return self.max_samples
        else:
            raise ValueError('[loss] function not recognized')


    def _compute_effective_items(self, n_threads=1):
        """return cached effective item biases (w_i + X_if @ w_if) and item embeddings (v_i + X_if @ v_if) via CSR SpMV/SpMM"""

        if self._cached_item_bias is None or self._cached_v_i_eff is None:
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
        return self._cached_item_bias, self._cached_v_i_eff


    def _compute_train_user_embeddings(self, n_threads=1):
        """return cached effective training-user embeddings (v_u + X_uf @ v_uf) via CSR SpMM"""

        if self._cached_train_u_emb is None:
            if self.has_uf and self._uf_any:
                self._cached_train_u_emb = _csr_dense_matmul(
                    self._uf_indptr, self._uf_indices, self._uf_data, self.v_uf, base=self.v_u, n_threads=n_threads
                )
            else:
                self._cached_train_u_emb = np.ascontiguousarray(self.v_u, dtype=np.float32)
        return self._cached_train_u_emb


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
        assert isinstance(fold_in_epochs, int) and fold_in_epochs >= 0, "[fold_in_epochs] must be a non-negative integer"

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
                    assert isinstance(sample_weight, (np.ndarray, pd.Series)), "[sample_weight] must be np.ndarray or pd.series"
                    assert sample_weight.ndim == 1, "[sample_weight] must a vector (ndim=1)"
                    assert len(sample_weight) == len(intx_u_col), "[sample_weight] must have the same length as [interactions]"
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
                # Only compute stored CSR feature projections for known users NOT overridden by incoming user_features
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
                        fold_pairs = np.ascontiguousarray(np.column_stack([u_arr, i_arr]), dtype=np.int32)
                        fold_sw = np.ascontiguousarray(sw_arr[valid] if not valid.all() else sw_arr, dtype=np.float32)

            if hist_u_parts:
                all_u = np.concatenate(hist_u_parts) if len(hist_u_parts) > 1 else hist_u_parts[0]
                all_i = np.concatenate(hist_i_parts) if len(hist_i_parts) > 1 else hist_i_parts[0]
                ui_indptr, ui_indices = self._pairs_to_csr(all_u, all_i, u_est_count, len(self.item_idx))

            if fold_pairs is not None:
                _fold_in(
                    fold_pairs,
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
        seed=1492
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
        seed=1492
    ):
        """learn or update model weights using the input data and resuming from the current model state"""

        assert isinstance(epochs, int) and epochs >= 1, "[epochs] must be a positive integer"
        assert isinstance(verbose, bool), "[verbose] must be a boolean value"
        assert isinstance(freeze_items, bool), "[freeze_items] must be a boolean value"

        if freeze_items and not self.is_fit:
            raise AssertionError("you must fit the model prior to calling fit_partial(..., freeze_items=True)")

        eff_threads = self._resolve_n_jobs(n_jobs)
        self._invalidate_caches()

        if self.is_fit:
            self._init_interactions(interactions, sample_weight, freeze_items=freeze_items, seed=int(seed))
            self._init_features(user_features, item_features, n_threads=eff_threads)
        else:
            self._init_all(interactions, user_features, item_features, sample_weight, n_threads=eff_threads, seed=int(seed))

        max_samples = self._max_samples_for_loss()

        if freeze_items:
            item_bias, v_i_eff = self._compute_effective_items(n_threads=eff_threads)
            if self.has_uf and self._uf_any:
                u_feat_emb = _csr_dense_matmul(
                    self._uf_indptr, self._uf_indices, self._uf_data, self.v_uf, n_threads=eff_threads
                )
            else:
                u_feat_emb = np.zeros((len(self.user_idx), self.factors), dtype=np.float32)

            _fold_in(
                self.interactions,
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
                int(seed),
                eff_threads
            )
            self._cached_train_u_emb = None
        else:
            _fit_csr(
                self.interactions,
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
                int(seed),
                eff_threads,
                self.feature_reg == 'uniform'
            )
            self._invalidate_caches()

        self.is_fit = True
        return self


    def embed_users(
        self,
        users,
        user_features=None,
        interactions=None,
        sample_weight=None,
        fold_in_epochs=5,
        cold_start='nan',
        n_jobs=None
    ):
        """compute effective F-dimensional latent representations for seen and/or unseen users without mutating model state"""

        assert getattr(users, '__iter__', False), "[users] must be an iterable (e.g. list, array, series)"
        assert self.is_fit, "you must fit the model prior to embedding users"

        eff_threads = self._resolve_n_jobs(n_jobs)
        users_arr = np.asarray(list(users) if not isinstance(users, (np.ndarray, pd.Series, pd.Index, list)) else users)
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

        mapped = _fast_get_indexer(user_index_obj, users_arr)
        valid = mapped >= 0

        if cold_start == 'drop':
            return pd.DataFrame(u_emb[mapped[valid]], index=users_arr[valid])

        out = np.full((len(users_arr), self.factors), np.nan, dtype=np.float32)
        if valid.any():
            out[valid] = u_emb[mapped[valid]]
        return pd.DataFrame(out, index=users_arr)


    def export_mips_embeddings(
        self,
        users=None,
        user_features=None,
        interactions=None,
        sample_weight=None,
        fold_in_epochs=5,
        cold_start='drop',
        n_jobs=None
    ):
        """export augmented (F + 1)-dimensional float32 user & item embeddings for sub-linear ANN / MIPS engines (FAISS, ScaNN)

        Inner product <user_mips[u], item_mips[i]> equals exact RankFM utility: item_bias[i] + <u_emb[u], v_i_eff[i]>.
        Returns (user_ids, user_mips, item_ids, item_mips).
        """

        assert self.is_fit, "you must fit the model prior to exporting MIPS embeddings"
        eff_threads = self._resolve_n_jobs(n_jobs)
        item_bias, v_i_eff = self._compute_effective_items(n_threads=eff_threads)

        item_mips = np.empty((len(item_bias), self.factors + 1), dtype=np.float32)
        item_mips[:, :self.factors] = v_i_eff
        item_mips[:, self.factors] = item_bias
        item_ids = self._item_index.to_numpy()

        if users is None and user_features is None and interactions is None:
            u_emb = self._compute_train_user_embeddings(n_threads=eff_threads)
            user_mips = np.empty((len(u_emb), self.factors + 1), dtype=np.float32)
            user_mips[:, :self.factors] = u_emb
            user_mips[:, self.factors] = 1.0
            return self._user_index.to_numpy(), user_mips, item_ids, item_mips

        target_users = self._user_index.to_numpy() if users is None else np.asarray(
            list(users) if not isinstance(users, (np.ndarray, pd.Series, pd.Index, list)) else users
        )
        emb_df = self.embed_users(
            target_users,
            user_features=user_features,
            interactions=interactions,
            sample_weight=sample_weight,
            fold_in_epochs=fold_in_epochs,
            cold_start=cold_start,
            n_jobs=eff_threads,
        )
        u_vals = emb_df.to_numpy(dtype=np.float32, copy=False)
        user_mips = np.empty((len(u_vals), self.factors + 1), dtype=np.float32)
        user_mips[:, :self.factors] = u_vals
        user_mips[:, self.factors] = np.where(np.isnan(u_vals[:, 0]), np.nan, 1.0)
        return emb_df.index.to_numpy(), user_mips, item_ids, item_mips


    def predict(
        self,
        pairs,
        cold_start='nan',
        user_features=None,
        interactions=None,
        sample_weight=None,
        fold_in_epochs=5,
        n_jobs=None
    ):
        """calculate the predicted pointwise utilities for all (user, item) pairs"""

        assert self.is_fit, "you must fit the model prior to generating predictions"
        eff_threads = self._resolve_n_jobs(n_jobs)
        u_col, i_col = self._extract_interaction_columns(pairs)

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
        fold_in_epochs=5,
        n_jobs=None
    ):
        """calculate the topN items for each user using batched BLAS SGEMM and parallel nogil C min-heaps"""

        assert getattr(users, '__iter__', False), "[users] must be an iterable (e.g. list, array, series)"
        assert self.is_fit, "you must fit the model prior to generating recommendations"

        eff_threads = self._resolve_n_jobs(n_jobs)
        users_arr = np.asarray(list(users) if not isinstance(users, (np.ndarray, pd.Series, pd.Index, list)) else users)
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
            # Cap temporary score chunk buffer to ~32 MB so it stays L3-cache-friendly across any catalog size
            chunk_size = max(64, min(4096, (32 * 1024 * 1024) // max(1, n_catalog * 4)))
            v_i_eff_T = v_i_eff.T

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
        fold_in_epochs=5,
        as_dataframe=True,
        n_jobs=None
    ):
        """stream topN recommendations in bounded-memory batches for 10M-100M+ user inference

        When as_dataframe=True (default), yields a pd.DataFrame per batch identical to recommend().
        When as_dataframe=False, yields (batch_user_ids_ndarray, batch_item_ids_ndarray) tuples.
        """

        assert getattr(users, '__iter__', False), "[users] must be an iterable (e.g. list, array, series)"
        assert isinstance(batch_size, int) and batch_size >= 1, "[batch_size] must be a positive integer"
        assert self.is_fit, "you must fit the model prior to generating recommendations"

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
                assert len(intx_sw) == len(intx_u), "[sample_weight] must have the same length as [interactions]"
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
                fold_in_epochs=fold_in_epochs,
                n_jobs=n_jobs,
            )
            if as_dataframe:
                yield df_batch
            else:
                yield df_batch.index.to_numpy(), df_batch.to_numpy()


    def similar_items(self, item_id, n_items=10):
        """find the most similar items wrt latent factor space representation"""

        assert self.is_fit, "you must fit the model prior to generating similarities"
        item_pos = int(_fast_get_indexer(self._item_index, [item_id])[0])
        assert item_pos >= 0, "you must select an [item_id] present in the training data"

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
        fold_in_epochs=5
    ):
        """find the most similar training users wrt latent factor space representation (supports unseen query users)"""

        assert self.is_fit, "you must fit the model prior to generating similarities"

        user_pos = int(_fast_get_indexer(self._user_index, [user_id])[0])
        in_training = user_pos >= 0
        if not in_training and user_features is None and interactions is None:
            raise AssertionError("you must select a [user_id] present in the training data or supply [user_features]/[interactions]")

        lr_all_users = self._compute_train_user_embeddings()

        if in_training and user_features is None and interactions is None:
            lr_user = lr_all_users[user_pos]
        else:
            emb_df = self.embed_users(
                [user_id],
                user_features=user_features,
                interactions=interactions,
                sample_weight=sample_weight,
                fold_in_epochs=fold_in_epochs,
                cold_start='nan'
            )
            lr_user = emb_df.iloc[0].to_numpy()
            assert not np.isnan(lr_user).any(), "user_id={} could not be embedded from supplied features/interactions".format(user_id)

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
