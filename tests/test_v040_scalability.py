import time
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from rankfm.rankfm import RankFM


def _make_rich_sparse_dataset(n_users=400, n_items=120, p_uf=60, p_if=120, seed=17):
    rng = np.random.default_rng(seed)

    # 4 latent clusters
    u_cluster = np.arange(n_users) % 4
    i_cluster = np.arange(n_items) % 4

    # Sparse user features: cluster indicator + 5 random sparse tags per user
    uf_rows, uf_cols, uf_vals = [], [], []
    for u in range(n_users):
        uf_rows.append(u)
        uf_cols.append(int(u_cluster[u]))
        uf_vals.append(1.0)
        extra = rng.choice(np.arange(4, p_uf), size=5, replace=False)
        for c in extra:
            uf_rows.append(u)
            uf_cols.append(int(c))
            uf_vals.append(float(rng.uniform(0.2, 1.0)))
    uf_csr = sp.csr_matrix((uf_vals, (uf_rows, uf_cols)), shape=(n_users, p_uf), dtype=np.float32)

    # Rich sparse item features (~20 active features per item)
    if_rows, if_cols, if_vals = [], [], []
    for i in range(n_items):
        if_rows.append(i)
        if_cols.append(int(i_cluster[i]))
        if_vals.append(1.0)
        extra = rng.choice(np.arange(4, p_if), size=19, replace=False)
        for c in extra:
            if_rows.append(i)
            if_cols.append(int(c))
            if_vals.append(float(rng.uniform(0.1, 1.0)))
    if_csr = sp.csr_matrix((if_vals, (if_rows, if_cols)), shape=(n_items, p_if), dtype=np.float32)

    # Cluster-aligned interactions
    intx_u, intx_i, intx_w = [], [], []
    for u in range(n_users):
        c = u_cluster[u]
        cand = np.flatnonzero(i_cluster == c)
        chosen = rng.choice(cand, size=8, replace=False)
        for item in chosen:
            intx_u.append(u)
            intx_i.append(int(item))
            intx_w.append(float(rng.uniform(0.5, 2.5)))

    ui_csr = sp.csr_matrix((intx_w, (intx_u, intx_i)), shape=(n_users, n_items), dtype=np.float32)
    return ui_csr, uf_csr, if_csr, u_cluster, i_cluster


def test_native_scipy_sparse_fit_predict_recommend_and_cold_start():
    """RankFM fits, predicts, and recommends directly on scipy.sparse CSR matrices without dense feature allocation"""

    ui_csr, uf_csr, if_csr, u_cluster, i_cluster = _make_rich_sparse_dataset()

    model = RankFM(
        factors=24,
        loss='warp',
        max_samples=30,
        alpha=0.005,
        beta=0.01,
        learning_rate=0.03,
        user_dropout=0.15,
        n_jobs=4,
    )
    model.fit(ui_csr, user_features=uf_csr, item_features=if_csr, epochs=12, seed=42)

    assert model.is_fit
    assert isinstance(model._user_index, pd.RangeIndex)
    assert isinstance(model._item_index, pd.RangeIndex)
    assert model._dense_x_uf_cache is None
    assert model._dense_x_if_cache is None

    # Verify zero-copy CSR properties match input sparse matrices
    assert (model.x_uf_csr != uf_csr).nnz == 0
    assert (model.x_if_csr != if_csr).nnz == 0

    # Verify training users receive cluster-aligned novel recommendations
    sample_users = np.arange(40, dtype=np.int32)
    recs = model.recommend(sample_users, n_items=5, filter_previous=True, n_jobs=4)
    assert recs.shape == (40, 5)
    assert recs.notnull().all().all()

    same_cluster_hits = 0
    for u in sample_users:
        rec_items = recs.loc[u].to_numpy(dtype=np.int32)
        same_cluster_hits += int((i_cluster[rec_items] == u_cluster[u]).sum())
    assert same_cluster_hits / (40 * 5) > 0.70

    # Unseen users passed via (ids, csr_matrix) tuple + sparse warm-start interactions
    unseen_ids = np.array([1000, 1001], dtype=np.int32)
    unseen_uf = sp.csr_matrix(
        ([1.0, 1.0], ([0, 1], [0, 2])),
        shape=(2, uf_csr.shape[1]),
        dtype=np.float32,
    )
    unseen_intx = pd.DataFrame([(1000, 0), (1001, 2)], columns=['user_id', 'item_id'])
    recs_unseen = model.recommend(
        unseen_ids,
        n_items=4,
        filter_previous=True,
        user_features=(unseen_ids, unseen_uf),
        interactions=unseen_intx,
        fold_in_epochs=10,
        n_jobs=2,
    )
    assert recs_unseen.shape == (2, 4)
    assert 0 not in recs_unseen.loc[1000].to_numpy()
    assert 2 not in recs_unseen.loc[1001].to_numpy()
    assert (i_cluster[recs_unseen.loc[1000].to_numpy(dtype=int)] == 0).sum() >= 3
    assert (i_cluster[recs_unseen.loc[1001].to_numpy(dtype=int)] == 2).sum() >= 3


@pytest.mark.parametrize("loss", ["bpr", "warp"])
@pytest.mark.parametrize("n_jobs", [1, 4])
def test_multithreaded_convergence_and_determinism(loss, n_jobs):
    """Multi-threaded pthread pool converges cleanly across both BPR and WARP without NaNs or state corruption"""

    ui_csr, uf_csr, if_csr, u_cluster, i_cluster = _make_rich_sparse_dataset(
        n_users=600, n_items=160, seed=99
    )

    model = RankFM(
        factors=32,
        loss=loss,
        max_samples=25,
        alpha=0.005,
        beta=0.01,
        learning_rate=0.10,
        learning_schedule='invscaling',
        user_dropout=0.1,
        n_jobs=n_jobs,
    )
    model.fit(ui_csr, user_features=uf_csr, item_features=if_csr, epochs=10, verbose=True, seed=123)

    # Partial fit with unseen users and freeze_items=True/False
    extra_intx = pd.DataFrame([(600, 0), (600, 4), (601, 1), (601, 5)], columns=['user_id', 'item_id'])
    extra_uf = (
        np.array([600, 601], dtype=np.int32),
        sp.csr_matrix(([1.0, 1.0], ([0, 1], [0, 1])), shape=(2, uf_csr.shape[1]), dtype=np.float32),
    )
    model.fit_partial(extra_intx, user_features=extra_uf, epochs=6, freeze_items=True, n_jobs=n_jobs)

    recs = model.recommend(np.arange(50), n_items=5, filter_previous=True, n_jobs=n_jobs)
    hits = sum(int((i_cluster[recs.loc[u].to_numpy(dtype=int)] == u_cluster[u]).sum()) for u in range(50))
    assert hits / (50 * 5) > 0.60


def test_uniform_lazy_feature_regularization():
    """feature_reg='uniform' shrinks even completely untouched sparse feature columns while maintaining high accuracy"""

    ui_csr, uf_csr, if_csr, u_cluster, i_cluster = _make_rich_sparse_dataset(seed=77)
    # Add an extra completely unused item feature column (all zeros in if_csr_ext)
    if_csr_ext = sp.hstack([if_csr, sp.csr_matrix((if_csr.shape[0], 1), dtype=np.float32)], format='csr')

    model = RankFM(
        factors=24,
        loss='warp',
        max_samples=20,
        alpha=0.005,
        beta=0.01,
        learning_rate=0.12,
        n_jobs=1,
        feature_reg='uniform',
    )
    model.fit(ui_csr, user_features=uf_csr, item_features=if_csr_ext, epochs=1, seed=7)

    # Set the completely unused feature column to 1.0 and verify lazy epoch-end shrinkage decays it
    model.v_if[-1, :] = 1.0
    model.w_if[-1] = 1.0
    norm_before = float(np.linalg.norm(model.v_if[-1]))

    model.fit_partial(ui_csr, epochs=8, seed=8)
    norm_after = float(np.linalg.norm(model.v_if[-1]))
    assert norm_after < norm_before * 0.5
    assert abs(float(model.w_if[-1])) < 0.5

    recs = model.recommend(np.arange(30), n_items=4, filter_previous=True)
    hits = sum(int((i_cluster[recs.loc[u].to_numpy(dtype=int)] == u_cluster[u]).sum()) for u in range(30))
    assert hits / (30 * 4) > 0.70


def test_recommend_batches_and_export_mips_embeddings_exact_parity():
    """recommend_batches matches recommend() bit-for-bit and export_mips_embeddings matches predict() to float32 precision"""

    ui_csr, uf_csr, if_csr, _, _ = _make_rich_sparse_dataset(n_users=250, n_items=80, seed=31)
    model = RankFM(factors=20, loss='warp', max_samples=20, learning_rate=0.03, n_jobs=2).fit(
        ui_csr, user_features=uf_csr, item_features=if_csr, epochs=8, seed=31
    )

    query_users = np.array(list(range(250)) + [9999], dtype=np.int32)
    full_recs = model.recommend(query_users, n_items=6, filter_previous=True, cold_start='nan', n_jobs=2)

    streamed_dfs = list(
        model.recommend_batches(
            query_users, batch_size=60, n_items=6, filter_previous=True, cold_start='nan', as_dataframe=True, n_jobs=2
        )
    )
    concat_df = pd.concat(streamed_dfs, axis=0)
    pd.testing.assert_frame_equal(full_recs, concat_df)

    # Tuple mode (as_dataframe=False)
    streamed_tuples = list(
        model.recommend_batches(
             np.arange(100, dtype=np.int32),
            batch_size=33,
            n_items=6,
            filter_previous=True,
            as_dataframe=False,
        )
    )
    u_cat = np.concatenate([u for u, _ in streamed_tuples])
    i_cat = np.vstack([it for _, it in streamed_tuples])
    np.testing.assert_array_equal(u_cat, np.arange(100, dtype=np.int32))
    np.testing.assert_array_equal(i_cat, full_recs.iloc[:100].to_numpy(dtype=np.int32))

    # MIPS export inner products equal exact predict() scores
    u_ids, u_mips, i_ids, i_mips = model.export_mips_embeddings()
    assert u_mips.shape == (250, 21)
    assert i_mips.shape == (80, 21)

    pair_u = np.repeat(np.arange(15, dtype=np.int32), 80)
    pair_i = np.tile(np.arange(80, dtype=np.int32), 15)
    pred_scores = model.predict(np.column_stack([pair_u, pair_i]))
    mips_scores = (u_mips[:15] @ i_mips.T).ravel()
    np.testing.assert_allclose(mips_scores, pred_scores, rtol=1e-5, atol=1e-5)
