"""
rankfm main modeling class
"""

import numpy as np
import pandas as pd

from rankfm._rankfm import _fit, _fold_in, _predict, _select_topk_chunk
from rankfm.utils import get_data


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
        user_dropout=0.0
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

        self._reset_state()


    # --------------------------------
    # begin private method definitions
    # --------------------------------


    def _invalidate_caches(self):
        """clear cached effective embeddings and materialized user_items dictionary"""

        self._cached_item_bias = None
        self._cached_v_i_eff = None
        self._cached_train_u_emb = None
        self._user_items_cache = None


    def _reset_state(self):
        """initialize or reset internal model state"""

        # [ID, IDX] arrays
        self.user_id = None
        self.item_id = None
        self.user_idx = None
        self.item_idx = None

        # [ID <-> IDX] mappings
        self.index_to_user = None
        self.index_to_item = None
        self.user_to_index = None
        self.item_to_index = None

        # user/item interactions and importance weights
        self.interactions = None
        self.sample_weight = None

        # CSR representation of observed items per user
        self._ui_indptr = None
        self._ui_indices = None

        # [user, item] features and schema metadata
        self.x_uf = None
        self.x_if = None
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

        if n_users == 0:
            return np.zeros(1, dtype=np.int32), np.zeros(0, dtype=np.int32)
        if len(u_arr) == 0 or n_items == 0:
            return np.zeros(n_users + 1, dtype=np.int32), np.zeros(0, dtype=np.int32)

        codes = u_arr.astype(np.int64) * np.int64(n_items) + i_arr.astype(np.int64)
        uniq_codes = np.unique(codes)
        uniq_u = (uniq_codes // np.int64(n_items)).astype(np.int32)
        uniq_i = np.ascontiguousarray(uniq_codes % np.int64(n_items), dtype=np.int32)

        counts = np.bincount(uniq_u, minlength=n_users)
        indptr = np.empty(n_users + 1, dtype=np.int32)
        indptr[0] = 0
        np.cumsum(counts, dtype=np.int32, out=indptr[1:])
        return indptr, uniq_i


    @staticmethod
    def _gather_csr_rows(src_indptr, src_indices, row_ids):
        """vectorized gather of selected CSR rows into flat (local_row_idx, col_idx) arrays without Python loops"""

        starts = src_indptr[row_ids]
        ends = src_indptr[row_ids + 1]
        lengths = ends - starts
        total = int(lengths.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

        cum_before = np.cumsum(lengths) - lengths
        shifts = np.repeat(starts - cum_before, lengths)
        flat_pos = np.arange(total, dtype=np.int64) + shifts
        gathered_cols = src_indices[flat_pos]
        local_rows = np.repeat(np.arange(len(row_ids), dtype=np.int32), lengths)
        return local_rows, gathered_cols


    @staticmethod
    def _extract_feature_arrays(features, expected_columns=None, expected_width=None, name='features'):
        """extract (pd.Index of IDs, contiguous float32 feature matrix) without copying full DataFrames unnecessarily"""

        assert isinstance(features, (np.ndarray, pd.DataFrame)), "[{}] must be np.ndarray or pd.DataFrame".format(name)
        assert features.shape[1] >= 2, "[{}] must contain an ID column plus at least one feature column".format(name)

        if isinstance(features, pd.DataFrame):
            id_col = features.columns[0]
            raw_ids = pd.Index(features[id_col].values)
            feat_df = features.iloc[:, 1:]
            if expected_columns is not None and len(feat_df.columns) == len(expected_columns):
                if list(feat_df.columns) != expected_columns and set(expected_columns).issubset(set(feat_df.columns)):
                    feat_df = feat_df[expected_columns]
            if expected_width is not None and feat_df.shape[1] != expected_width:
                raise ValueError("[{}] has {} feature columns, expected {}".format(name, feat_df.shape[1], expected_width))
            values = np.ascontiguousarray(feat_df.to_numpy(), dtype=np.float32)
        else:
            if expected_width is not None and (features.shape[1] - 1) != expected_width:
                raise ValueError("[{}] has {} feature columns, expected {}".format(name, features.shape[1] - 1, expected_width))
            raw_ids = pd.Index(features[:, 0])
            values = np.ascontiguousarray(features[:, 1:], dtype=np.float32)

        if raw_ids.has_duplicates:
            keep_mask = ~raw_ids.duplicated(keep='last')
            raw_ids = raw_ids[keep_mask]
            values = np.ascontiguousarray(values[keep_mask], dtype=np.float32)

        return raw_ids, values


    @staticmethod
    def _extract_interaction_columns(interactions):
        """extract (user_col_array, item_col_array) without full DataFrame copies"""

        assert isinstance(interactions, (np.ndarray, pd.DataFrame)), "[interactions] must be np.ndarray or pd.dataframe"
        assert interactions.shape[1] == 2, "[interactions] should be: [user_id, item_id]"

        if isinstance(interactions, pd.DataFrame):
            return interactions.iloc[:, 0].to_numpy(), interactions.iloc[:, 1].to_numpy()
        return interactions[:, 0], interactions[:, 1]


    def _init_all(self, interactions, user_features=None, item_features=None, sample_weight=None):
        """index the interaction data and user/item features and initialize model weights"""

        u_col, i_col = self._extract_interaction_columns(interactions)
        self.user_id = pd.Series(np.sort(pd.unique(u_col)))
        self.item_id = pd.Series(np.sort(pd.unique(i_col)))

        self.index_to_user = self.user_id
        self.index_to_item = self.item_id

        self.user_to_index = pd.Series(data=self.index_to_user.index, index=self.index_to_user.values)
        self.item_to_index = pd.Series(data=self.index_to_item.index, index=self.index_to_item.values)

        self.user_idx = np.arange(len(self.user_id), dtype=np.int32)
        self.item_idx = np.arange(len(self.item_id), dtype=np.int32)

        self._init_interactions(interactions, sample_weight)
        self._init_features(user_features, item_features)
        self._init_weights(user_features, item_features)


    def _expand_users_for_partial(self, u_col, i_col, freeze_items=False):
        """append any previously unseen users in interactions to the model's user index and state arrays"""

        valid_item_mask = self.item_to_index.index.get_indexer(i_col) >= 0
        incoming_users = pd.Index(pd.unique(u_col[valid_item_mask]))
        new_users = incoming_users.difference(self.user_to_index.index).sort_values().values

        if len(new_users) == 0:
            return

        n_new = len(new_users)
        self.user_id = pd.Series(np.concatenate([self.user_id.values, new_users]))
        self.index_to_user = self.user_id
        self.user_to_index = pd.Series(data=self.index_to_user.index, index=self.index_to_user.values)
        self.user_idx = np.arange(len(self.user_id), dtype=np.int32)

        if self.has_uf or freeze_items:
            new_v_u = np.zeros((n_new, self.factors), dtype=np.float32)
        else:
            new_v_u = np.random.normal(loc=0, scale=self.sigma, size=(n_new, self.factors)).astype(np.float32)

        self.v_u = np.ascontiguousarray(np.vstack([self.v_u, new_v_u]), dtype=np.float32)
        new_x_uf = np.zeros((n_new, self.x_uf.shape[1]), dtype=np.float32)
        self.x_uf = np.ascontiguousarray(np.vstack([self.x_uf, new_x_uf]), dtype=np.float32)


    def _init_interactions(self, interactions, sample_weight, freeze_items=False):
        """map interaction data to internal user/item indexes and build CSR user_items arrays"""

        u_col, i_col = self._extract_interaction_columns(interactions)

        if sample_weight is not None:
            assert isinstance(sample_weight, (np.ndarray, pd.Series)), "[sample_weight] must be np.ndarray or pd.series"
            assert sample_weight.ndim == 1, "[sample_weight] must a vector (ndim=1)"
            assert len(sample_weight) == len(u_col), "[sample_weight] must have the same length as [interactions]"

        if self.is_fit:
            self._expand_users_for_partial(u_col, i_col, freeze_items=freeze_items)

        u_mapped = self.user_to_index.index.get_indexer(u_col)
        i_mapped = self.item_to_index.index.get_indexer(i_col)
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


    def _init_features(self, user_features=None, item_features=None):
        """initialize or incrementally update user/item features via vectorized indexers"""

        if not self.is_fit:
            if user_features is not None:
                self.has_uf = True
                self.uf_columns = list(user_features.columns[1:]) if isinstance(user_features, pd.DataFrame) else None
                uf_ids, uf_vals = self._extract_feature_arrays(user_features, name='user_features')
                pos = uf_ids.get_indexer(self.user_to_index.index)
                if (pos < 0).any():
                    raise KeyError('the users in [user_features] do not match the users in [interactions]')
                self.x_uf = np.ascontiguousarray(uf_vals[pos], dtype=np.float32)
            else:
                self.has_uf = False
                self.uf_columns = None
                self.x_uf = np.zeros((len(self.user_idx), 1), dtype=np.float32)

            if item_features is not None:
                self.has_if = True
                self.if_columns = list(item_features.columns[1:]) if isinstance(item_features, pd.DataFrame) else None
                if_ids, if_vals = self._extract_feature_arrays(item_features, name='item_features')
                pos = if_ids.get_indexer(self.item_to_index.index)
                if (pos < 0).any():
                    raise KeyError('the items in [item_features] do not match the items in [interactions]')
                self.x_if = np.ascontiguousarray(if_vals[pos], dtype=np.float32)
            else:
                self.has_if = False
                self.if_columns = None
                self.x_if = np.zeros((len(self.item_idx), 1), dtype=np.float32)
        else:
            if user_features is not None:
                if not self.has_uf:
                    raise ValueError("cannot supply [user_features] to fit_partial() when model was fit without [user_features]")
                uf_ids, uf_vals = self._extract_feature_arrays(
                    user_features,
                    expected_columns=self.uf_columns,
                    expected_width=self.x_uf.shape[1],
                    name='user_features'
                )
                target_u = self.user_to_index.index.get_indexer(uf_ids)
                valid = target_u >= 0
                if valid.any():
                    self.x_uf[target_u[valid]] = uf_vals[valid]

            if item_features is not None:
                if not self.has_if:
                    raise ValueError("cannot supply [item_features] to fit_partial() when model was fit without [item_features]")
                if_ids, if_vals = self._extract_feature_arrays(
                    item_features,
                    expected_columns=self.if_columns,
                    expected_width=self.x_if.shape[1],
                    name='item_features'
                )
                target_i = self.item_to_index.index.get_indexer(if_ids)
                valid = target_i >= 0
                if valid.any():
                    self.x_if[target_i[valid]] = if_vals[valid]


    def _init_weights(self, user_features=None, item_features=None):
        """initialize model weights given user/item and user-feature/item-feature indexes/shapes"""

        self.w_i = np.zeros(len(self.item_idx), dtype=np.float32)
        self.w_if = np.zeros(self.x_if.shape[1], dtype=np.float32)

        self.v_u = np.random.normal(loc=0, scale=self.sigma, size=(len(self.user_idx), self.factors)).astype(np.float32)
        self.v_i = np.random.normal(loc=0, scale=self.sigma, size=(len(self.item_idx), self.factors)).astype(np.float32)

        if user_features is not None:
            scale = (self.alpha / self.beta) * self.sigma
            self.v_uf = np.random.normal(loc=0, scale=scale, size=(self.x_uf.shape[1], self.factors)).astype(np.float32)
        else:
            self.v_uf = np.zeros((self.x_uf.shape[1], self.factors), dtype=np.float32)

        if item_features is not None:
            scale = (self.alpha / self.beta) * self.sigma
            self.v_if = np.random.normal(loc=0, scale=scale, size=(self.x_if.shape[1], self.factors)).astype(np.float32)
        else:
            self.v_if = np.zeros((self.x_if.shape[1], self.factors), dtype=np.float32)


    def _max_samples_for_loss(self):
        if self.loss == 'bpr':
            return 1
        elif self.loss == 'warp':
            return self.max_samples
        else:
            raise ValueError('[loss] function not recognized')


    def _compute_effective_items(self):
        """return cached effective item biases (w_i + x_if @ w_if) and item embeddings (v_i + x_if @ v_if)"""

        if self._cached_item_bias is None or self._cached_v_i_eff is None:
            if self.has_if:
                self._cached_item_bias = np.ascontiguousarray(self.w_i + np.dot(self.x_if, self.w_if), dtype=np.float32)
                self._cached_v_i_eff = np.ascontiguousarray(self.v_i + np.dot(self.x_if, self.v_if), dtype=np.float32)
            else:
                self._cached_item_bias = np.ascontiguousarray(self.w_i, dtype=np.float32)
                self._cached_v_i_eff = np.ascontiguousarray(self.v_i, dtype=np.float32)
        return self._cached_item_bias, self._cached_v_i_eff


    def _compute_train_user_embeddings(self):
        """return cached effective training-user embeddings (v_u + x_uf @ v_uf)"""

        if self._cached_train_u_emb is None:
            if self.has_uf:
                self._cached_train_u_emb = np.ascontiguousarray(self.v_u + np.dot(self.x_uf, self.v_uf), dtype=np.float32)
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
        need_seen_items=False
    ):
        """vectorized builder for effective user embeddings, index lookup, and CSR seen-item history"""

        if cold_start not in ('nan', 'drop', 'estimate'):
            raise ValueError("param [cold_start] must be set to 'nan', 'drop', or 'estimate'")
        assert isinstance(fold_in_epochs, int) and fold_in_epochs >= 0, "[fold_in_epochs] must be a non-negative integer"

        item_bias, v_i_eff = self._compute_effective_items()

        # Fast path: standard inference using cached embeddings and CSR history directly
        if user_features is None and interactions is None and cold_start in ('nan', 'drop'):
            u_emb = self._compute_train_user_embeddings()
            return u_emb, self.user_to_index.index, self._ui_indptr, self._ui_indices, item_bias, v_i_eff

        # Dynamic / unseen-user path: fully vectorized index alignment & CSR construction
        uf_ids = None
        uf_vals = None
        if user_features is not None:
            if not self.has_uf:
                raise ValueError("model was not fit with [user_features]; cannot use [user_features] at inference")
            uf_ids, uf_vals = self._extract_feature_arrays(
                user_features,
                expected_columns=self.uf_columns,
                expected_width=self.x_uf.shape[1],
                name='user_features'
            )

        intx_u_col = None
        intx_i_col = None
        sw_arr = None
        intx_user_idx = None
        if interactions is not None:
            intx_u_col, intx_i_col = self._extract_interaction_columns(interactions)
            intx_user_idx = pd.Index(pd.unique(intx_u_col))

            if sample_weight is not None:
                assert isinstance(sample_weight, (np.ndarray, pd.Series)), "[sample_weight] must be np.ndarray or pd.series"
                assert sample_weight.ndim == 1, "[sample_weight] must a vector (ndim=1)"
                assert len(sample_weight) == len(intx_u_col), "[sample_weight] must have the same length as [interactions]"
                sw_arr = np.asarray(get_data(sample_weight), dtype=np.float32)
            else:
                sw_arr = np.ones(len(intx_u_col), dtype=np.float32)

        unique_idx = pd.Index(pd.unique(np.asarray(users)))
        known_indexer = self.user_to_index.index.get_indexer(unique_idx)
        uf_indexer = uf_ids.get_indexer(unique_idx) if uf_ids is not None else None

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
        x_uf_batch = np.zeros((u_est_count, self.x_uf.shape[1]), dtype=np.float32)

        known_mask = est_known_indexer >= 0
        if known_mask.any():
            train_u_ids = est_known_indexer[known_mask]
            v_u_batch[known_mask] = self.v_u[train_u_ids]
            x_uf_batch[known_mask] = self.x_uf[train_u_ids]

        if est_uf_indexer is not None:
            uf_mask = est_uf_indexer >= 0
            if uf_mask.any():
                x_uf_batch[uf_mask] = uf_vals[est_uf_indexer[uf_mask]]

        if self.has_uf:
            u_feat_emb = np.ascontiguousarray(np.dot(x_uf_batch, self.v_uf), dtype=np.float32)
        else:
            u_feat_emb = np.zeros((u_est_count, self.factors), dtype=np.float32)

        need_item_history = need_seen_items or (intx_u_col is not None and fold_in_epochs > 0)
        ui_indptr = np.zeros(u_est_count + 1, dtype=np.int32)
        ui_indices = np.zeros(0, dtype=np.int32)

        if need_item_history:
            hist_u_parts = []
            hist_i_parts = []

            if known_mask.any():
                known_local = np.flatnonzero(known_mask).astype(np.int32)
                gathered_local, gathered_items = self._gather_csr_rows(self._ui_indptr, self._ui_indices, est_known_indexer[known_mask])
                if len(gathered_items) > 0:
                    hist_u_parts.append(known_local[gathered_local])
                    hist_i_parts.append(gathered_items)

            fold_pairs = None
            fold_sw = None
            if intx_u_col is not None:
                mapped_u = est_idx.get_indexer(intx_u_col)
                mapped_i = self.item_to_index.index.get_indexer(intx_i_col)
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
                    fold_in_epochs
                )

        u_emb = np.ascontiguousarray(v_u_batch + u_feat_emb, dtype=np.float32)
        return u_emb, est_idx, ui_indptr, ui_indices, item_bias, v_i_eff


    # -------------------------------
    # begin public method definitions
    # -------------------------------


    def fit(self, interactions, user_features=None, item_features=None, sample_weight=None, epochs=1, verbose=False):
        """clear previous model state and learn new model weights using the input data"""

        self._reset_state()
        self.fit_partial(interactions, user_features, item_features, sample_weight, epochs, verbose)
        return self


    def fit_partial(
        self,
        interactions,
        user_features=None,
        item_features=None,
        sample_weight=None,
        epochs=1,
        verbose=False,
        freeze_items=False
    ):
        """learn or update model weights using the input data and resuming from the current model state"""

        assert isinstance(epochs, int) and epochs >= 1, "[epochs] must be a positive integer"
        assert isinstance(verbose, bool), "[verbose] must be a boolean value"
        assert isinstance(freeze_items, bool), "[freeze_items] must be a boolean value"

        if freeze_items and not self.is_fit:
            raise AssertionError("you must fit the model prior to calling fit_partial(..., freeze_items=True)")

        self._invalidate_caches()

        if self.is_fit:
            self._init_interactions(interactions, sample_weight, freeze_items=freeze_items)
            self._init_features(user_features, item_features)
        else:
            self._init_all(interactions, user_features, item_features, sample_weight)

        max_samples = self._max_samples_for_loss()

        if freeze_items:
            item_bias, v_i_eff = self._compute_effective_items()
            if self.has_uf:
                u_feat_emb = np.ascontiguousarray(np.dot(self.x_uf, self.v_uf), dtype=np.float32)
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
                epochs
            )
            self._cached_train_u_emb = None
        else:
            _fit(
                self.interactions,
                self.sample_weight,
                self._ui_indptr,
                self._ui_indices,
                self.x_uf,
                self.x_if,
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
                self.user_dropout
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
        cold_start='nan'
    ):
        """compute effective F-dimensional latent representations for seen and/or unseen users without mutating model state"""

        assert getattr(users, '__iter__', False), "[users] must be an iterable (e.g. list, array, series)"
        assert self.is_fit, "you must fit the model prior to embedding users"

        users_arr = np.asarray(list(users) if not isinstance(users, (np.ndarray, pd.Series, pd.Index, list)) else users)
        u_emb, user_index_obj, _, _, _, _ = self._prepare_inference_state(
            users_arr,
            cold_start=cold_start,
            user_features=user_features,
            interactions=interactions,
            sample_weight=sample_weight,
            fold_in_epochs=fold_in_epochs,
            need_seen_items=False
        )

        mapped = user_index_obj.get_indexer(users_arr)
        valid = mapped >= 0

        if cold_start == 'drop':
            return pd.DataFrame(u_emb[mapped[valid]], index=users_arr[valid])

        out = np.full((len(users_arr), self.factors), np.nan, dtype=np.float32)
        if valid.any():
            out[valid] = u_emb[mapped[valid]]
        return pd.DataFrame(out, index=users_arr)


    def predict(
        self,
        pairs,
        cold_start='nan',
        user_features=None,
        interactions=None,
        sample_weight=None,
        fold_in_epochs=5
    ):
        """calculate the predicted pointwise utilities for all (user, item) pairs"""

        assert self.is_fit, "you must fit the model prior to generating predictions"
        u_col, i_col = self._extract_interaction_columns(pairs)

        u_emb, user_index_obj, _, _, item_bias, v_i_eff = self._prepare_inference_state(
            u_col,
            cold_start=cold_start,
            user_features=user_features,
            interactions=interactions,
            sample_weight=sample_weight,
            fold_in_epochs=fold_in_epochs,
            need_seen_items=False
        )

        u_idx = np.ascontiguousarray(user_index_obj.get_indexer(u_col), dtype=np.int32)
        i_idx = np.ascontiguousarray(self.item_to_index.index.get_indexer(i_col), dtype=np.int32)

        scores = _predict(u_idx, i_idx, u_emb, item_bias, v_i_eff)

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
        fold_in_epochs=5
    ):
        """calculate the topN items for each user using batched BLAS SGEMM and a nogil C min-heap"""

        assert getattr(users, '__iter__', False), "[users] must be an iterable (e.g. list, array, series)"
        assert self.is_fit, "you must fit the model prior to generating recommendations"

        users_arr = np.asarray(list(users) if not isinstance(users, (np.ndarray, pd.Series, pd.Index, list)) else users)
        u_emb, user_index_obj, ui_indptr, ui_indices, item_bias, v_i_eff = self._prepare_inference_state(
            users_arr,
            cold_start=cold_start,
            user_features=user_features,
            interactions=interactions,
            sample_weight=sample_weight,
            fold_in_epochs=fold_in_epochs,
            need_seen_items=filter_previous
        )

        req_u_idx = user_index_obj.get_indexer(users_arr)
        valid_mask = req_u_idx >= 0
        valid_u_idx = np.ascontiguousarray(req_u_idx[valid_mask], dtype=np.int32)
        n_valid = len(valid_u_idx)
        n_catalog = len(item_bias)

        out_valid_idx = np.full((n_valid, n_items), -1, dtype=np.int32)

        if n_valid > 0 and n_catalog > 0:
            # Cap temporary score chunk buffer to ~32 MB so it stays cache-friendly across any catalog size
            chunk_size = max(64, min(2048, (32 * 1024 * 1024) // max(1, n_catalog * 4)))
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
                    out_valid_idx[start:end]
                )

        item_id_vals = self.index_to_item.values
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


    def similar_items(self, item_id, n_items=10):
        """find the most similar items wrt latent factor space representation"""

        assert self.is_fit, "you must fit the model prior to generating similarities"
        item_pos = self.item_to_index.index.get_loc(item_id) if item_id in self.item_to_index.index else -1
        assert item_pos >= 0, "you must select an [item_id] present in the training data"

        _, lr_all_items = self._compute_effective_items()
        sims = np.dot(lr_all_items, lr_all_items[item_pos])
        sims[item_pos] = -np.inf

        k = min(n_items, len(sims) - 1)
        if k <= 0:
            return np.array([], dtype=self.index_to_item.values.dtype)
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return self.index_to_item.values[top_idx]


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

        in_training = user_id in self.user_to_index.index
        if not in_training and user_features is None and interactions is None:
            raise AssertionError("you must select a [user_id] present in the training data or supply [user_features]/[interactions]")

        lr_all_users = self._compute_train_user_embeddings()

        if in_training and user_features is None and interactions is None:
            user_pos = self.user_to_index.index.get_loc(user_id)
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
            user_pos = self.user_to_index.index.get_loc(user_id) if in_training else -1

        sims = np.dot(lr_all_users, lr_user)
        if user_pos >= 0:
            sims[user_pos] = -np.inf

        max_avail = len(sims) - (1 if user_pos >= 0 else 0)
        k = min(n_users, max_avail)
        if k <= 0:
            return np.array([], dtype=self.index_to_user.values.dtype)
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return self.index_to_user.values[top_idx]
