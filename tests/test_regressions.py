"""regression tests for seeding, CSR construction, small-catalog WARP, CPU detection, and batched inference"""

import warnings

import numpy as np
import pandas as pd
import pytest

import rankfm.rankfm as rankfm_module
from rankfm.rankfm import RankFM, _append_zero_rows, _available_cpus


def _make_interactions(n_users=300, n_items=60, per_user=12, seed=0):
    rng = np.random.default_rng(seed)
    users = np.repeat(np.arange(n_users), per_user)
    items = rng.integers(0, n_items, size=len(users))
    return pd.DataFrame({'user_id': users, 'item_id': items})


def test_seed_makes_single_threaded_fit_reproducible():
    """fit(seed=...) fixes weight initialization as well as sampling, with no reliance on the global NumPy seed"""

    interactions = _make_interactions()
    first = RankFM(factors=8, loss='warp', max_samples=5).fit(interactions, epochs=3, seed=7, n_jobs=1)
    np.random.seed(999)
    second = RankFM(factors=8, loss='warp', max_samples=5).fit(interactions, epochs=3, seed=7, n_jobs=1)
    other = RankFM(factors=8, loss='warp', max_samples=5).fit(interactions, epochs=3, seed=8, n_jobs=1)

    for name in ('w_i', 'v_u', 'v_i'):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert not np.array_equal(first.v_i, other.v_i)


def test_pairs_to_csr_sorts_and_deduplicates():
    """observed-item CSR rows are sorted and unique, and the row pointer is int64 as the C kernels expect"""

    u_arr = np.array([2, 0, 2, 0, 2, 0, 3], dtype=np.int32)
    i_arr = np.array([5, 1, 5, 4, 0, 1, 2], dtype=np.int32)
    indptr, indices = RankFM._pairs_to_csr(u_arr, i_arr, n_users=4, n_items=6)

    assert indptr.dtype == np.int64
    assert indptr.tolist() == [0, 2, 2, 4, 5]
    assert indices.tolist() == [1, 4, 0, 5, 2]


def test_warp_max_samples_is_capped_at_catalog_size():
    """[max_samples] beyond (n_items - 1) is capped: past that point the WARP multiplier log((I - 1) / sampled) is negative"""

    interactions = _make_interactions(n_users=200, n_items=6, per_user=2, seed=3)
    capped = RankFM(factors=4, loss='warp', max_samples=5).fit(interactions, epochs=10, seed=1, n_jobs=1)
    beyond = RankFM(factors=4, loss='warp', max_samples=50).fit(interactions, epochs=10, seed=1, n_jobs=1)

    for name in ('w_i', 'v_u', 'v_i'):
        assert np.array_equal(getattr(capped, name), getattr(beyond, name))


def test_uniform_feature_reg_decays_unused_features_on_any_thread_count():
    """feature_reg='uniform' shrinks a feature that no row uses, single-threaded and multi-threaded alike"""

    interactions = _make_interactions(n_users=500, n_items=60, per_user=10)
    feature_values = np.zeros((60, 3), dtype=np.float32)
    feature_values[:, 0] = np.arange(60) % 2
    feature_values[:, 1] = np.arange(60) % 3 == 0
    item_features = (np.arange(60), feature_values)  # column 2 is never active

    norms = {}
    for n_jobs in (1, 4):
        model = RankFM(factors=6, beta=0.1, feature_reg='uniform').fit(
            interactions, item_features=item_features, epochs=1, seed=3, n_jobs=n_jobs
        )
        model.v_if[2, :] = 1.0
        model.w_if[2] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            model.fit_partial(interactions, epochs=6, seed=4, n_jobs=n_jobs)
        norms[n_jobs] = float(np.linalg.norm(model.v_if[2])) / np.sqrt(6)
        assert norms[n_jobs] < 0.9
        assert abs(float(model.w_if[2])) < 0.9

    # an untouched feature decays by the same closed-form factor however many threads ran
    assert norms[4] == pytest.approx(norms[1], rel=1e-4)


def test_n_jobs_all_cores_respects_available_cpus(monkeypatch):
    """n_jobs=-1 resolves from the CPUs available to the process, not from the host core count"""

    assert 1 <= _available_cpus() <= (rankfm_module.os.cpu_count() or 1)

    monkeypatch.setattr(rankfm_module, '_available_cpus', lambda: 3)
    assert RankFM(n_jobs=-1)._resolve_n_jobs() == 3
    assert RankFM(n_jobs=-2)._resolve_n_jobs() == 2
    assert RankFM(n_jobs=5)._resolve_n_jobs() == 5


def test_recommend_batches_matches_recommend_for_unseen_users():
    """batched inference with user features and warm-start interactions returns exactly what one recommend() call does"""

    interactions = _make_interactions(n_users=200, n_items=40, per_user=8, seed=5)
    all_features = pd.DataFrame({
        'user_id': np.arange(260),
        'f0': (np.arange(260) % 2).astype(float),
        'f1': (np.arange(260) % 3 == 0).astype(float),
    })
    model = RankFM(factors=6).fit(
        interactions, user_features=all_features.iloc[:200], epochs=3, seed=2, n_jobs=1
    )

    rng = np.random.default_rng(9)
    new_interactions = pd.DataFrame({
        'user_id': rng.integers(180, 260, 300),
        'item_id': rng.integers(0, 40, 300),
    })
    new_weights = rng.uniform(0.5, 1.5, 300).astype(np.float32)
    query_users = np.arange(150, 260)

    kwargs = dict(
        n_items=5, filter_previous=True, cold_start='nan', user_features=all_features,
        interactions=new_interactions, sample_weight=new_weights, fold_in_epochs=3, n_jobs=1,
    )
    whole = model.recommend(query_users, **kwargs)
    batched = pd.concat(list(model.recommend_batches(query_users, batch_size=len(query_users), **kwargs)))
    pd.testing.assert_frame_equal(whole, batched)

    small_batches = list(model.recommend_batches(query_users, batch_size=25, **kwargs))
    assert sum(len(batch) for batch in small_batches) == len(query_users)
    for batch in small_batches:
        expected = model.recommend(batch.index.to_numpy(), **{
            **kwargs,
            'interactions': new_interactions[new_interactions['user_id'].isin(batch.index)],
            'sample_weight': new_weights[new_interactions['user_id'].isin(batch.index).to_numpy()],
        })
        pd.testing.assert_frame_equal(batch, expected)


def test_invalid_input_raises_value_error_and_assertion_error():
    """validation raises an error that is both a ValueError and (for older callers) an AssertionError"""

    with pytest.raises(ValueError):
        RankFM(factors=-3)
    with pytest.raises(AssertionError):
        RankFM(loss='hinge')
    with pytest.raises(ValueError):
        RankFM().recommend([1, 2, 3])


def _dense_item_features(n_items=80, n_feats=60, seed=4):
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.5, 1.0, size=(n_items, n_feats)) * (rng.random((n_items, n_feats)) < 0.4)
    return np.arange(n_items), values.astype(np.float32)


def test_step_normalization_keeps_feature_heavy_training_stable():
    """with many active features per row, normalized steps stay bounded where raw steps run away"""

    interactions = _make_interactions(n_users=300, n_items=80, per_user=8, seed=4)
    item_features = _dense_item_features()
    settings = dict(factors=16, loss='warp', alpha=0.01, beta=0.01, learning_rate=0.3)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        stable = RankFM(**settings).fit(interactions, item_features=item_features, epochs=15, seed=1, n_jobs=1)
    assert max(np.abs(stable.v_i).max(), np.abs(stable.v_if).max(), np.abs(stable.v_u).max()) < 10.0

    # raw steps either trip the magnitude warning or reach non-finite weights, depending on how fast they blow up
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            RankFM(normalize_steps=False, **settings).fit(interactions, item_features=item_features, epochs=15, seed=1, n_jobs=1)
            diverged = any("diverging" in str(w.message) for w in caught)
        except AssertionError:
            diverged = True
    assert diverged


def test_step_normalization_leaves_feature_free_models_unchanged():
    """without side features there is nothing to normalize, so both settings train the identical model"""

    interactions = _make_interactions()
    on = RankFM(factors=8, loss='warp', normalize_steps=True).fit(interactions, epochs=3, seed=5, n_jobs=1)
    off = RankFM(factors=8, loss='warp', normalize_steps=False).fit(interactions, epochs=3, seed=5, n_jobs=1)
    assert np.array_equal(on.v_u, off.v_u) and np.array_equal(on.v_i, off.v_i)


def test_non_finite_inputs_are_rejected_before_training():
    """NaN or infinite feature values and sample weights raise immediately instead of after a wasted epoch"""

    interactions = _make_interactions(n_users=50, n_items=20, per_user=5)
    bad = np.ones((20, 3), dtype=np.float32)
    bad[7, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        RankFM(factors=4).fit(interactions, item_features=(np.arange(20), bad), epochs=1)

    sp = pytest.importorskip("scipy.sparse")
    bad_sparse = sp.csr_matrix(np.where(np.eye(50, 6, dtype=bool), np.inf, 0.0).astype(np.float32))
    with pytest.raises(ValueError, match="NaN or infinite"):
        RankFM(factors=4).fit(interactions, user_features=(np.arange(50), bad_sparse), epochs=1)

    weights = np.ones(len(interactions), dtype=np.float32)
    weights[3] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        RankFM(factors=4).fit(interactions, sample_weight=weights, epochs=1)


def test_model_does_not_alias_or_reorder_caller_sparse_features():
    """the model keeps its own copy of sparse features and never sorts the caller's matrix in place"""

    sp = pytest.importorskip("scipy.sparse")
    interactions = _make_interactions(n_users=40, n_items=12, per_user=4)
    features = sp.csr_matrix(
        (np.array([1.0, 0.5, 1.0, 0.25] * 6, dtype=np.float32), np.array([2, 0, 1, 3] * 6), np.arange(0, 25, 2)),
        shape=(12, 4),
    )
    features.indptr = features.indptr.astype(np.int64)
    features.indices = features.indices.astype(np.int32)
    indices_before = features.indices.copy()

    model = RankFM(factors=4).fit(interactions, item_features=features, epochs=1, seed=1)
    assert np.array_equal(features.indices, indices_before)

    stored_before = model.x_if.copy()
    features.data[:] = 99.0
    assert np.array_equal(model.x_if, stored_before)


def test_user_table_growth_reuses_spare_capacity():
    """adding users appends into spare capacity instead of reallocating the whole table on every call"""

    table = np.arange(12, dtype=np.float32).reshape(4, 3)
    grown = _append_zero_rows(table, 2)
    assert grown.shape == (6, 3) and np.array_equal(grown[:4], table) and not grown[4:].any()

    grown[4:] = 7.0
    again = _append_zero_rows(grown, 3)
    assert np.shares_memory(again, grown)
    assert again.shape == (9, 3) and np.array_equal(again[:6], grown) and not again[6:].any()

    interactions = _make_interactions(n_users=30, n_items=10, per_user=3)
    model = RankFM(factors=4).fit(interactions, epochs=1, seed=1)
    for step in range(3):
        new_user = 30 + step
        model.fit_partial(pd.DataFrame({'user_id': [new_user, new_user], 'item_id': [1, 2]}), epochs=1)
    assert isinstance(model._user_index, pd.RangeIndex) and len(model._user_index) == 33
    assert model.v_u.shape == (33, 4) and model.v_u.flags.c_contiguous
    assert model.recommend([32], n_items=2, filter_previous=True).notnull().all().all()


def test_large_weights_trigger_divergence_warning(monkeypatch):
    """weights past the magnitude threshold are reported once per fit instead of being returned silently"""

    import rankfm._rankfm as kernel
    monkeypatch.setattr(kernel, 'LARGE_WEIGHT_THRESHOLD', 0.01)  # any trained model exceeds this

    with pytest.warns(RuntimeWarning, match="diverging") as caught:
        RankFM(factors=4).fit(_make_interactions(), epochs=3, seed=1, n_jobs=1)
    assert len(caught) == 1


def test_merge_pairs_into_csr_matches_full_rebuild():
    """splicing new pairs into the stored history gives the same CSR as rebuilding it from all pairs"""

    rng = np.random.default_rng(6)
    n_items = 37
    for old_users, new_users, n_old, n_new in ((50, 50, 400, 60), (50, 65, 400, 200), (8, 8, 30, 0), (20, 30, 0, 40)):
        old_u = rng.integers(0, old_users, n_old).astype(np.int32)
        old_i = rng.integers(0, n_items, n_old).astype(np.int32)
        new_u = rng.integers(0, new_users, n_new).astype(np.int32)
        new_i = rng.integers(0, n_items, n_new).astype(np.int32)

        indptr, indices = RankFM._pairs_to_csr(old_u, old_i, old_users, n_items)
        merged_ptr, merged_idx = RankFM._merge_pairs_into_csr(indptr, indices, new_u, new_i, new_users, n_items)
        full_ptr, full_idx = RankFM._pairs_to_csr(
            np.concatenate([old_u, new_u]), np.concatenate([old_i, new_i]), new_users, n_items
        )

        assert merged_ptr.dtype == np.int64 and merged_idx.dtype == np.int32
        assert np.array_equal(merged_ptr, full_ptr)
        assert np.array_equal(merged_idx, full_idx)


def test_shuffle_permutation_visits_every_row_exactly_once():
    """the computed per-epoch shuffle is a true permutation, including past the 32-bit row limit"""

    from rankfm._rankfm import _permutation_sample

    for n in list(range(1, 40)) + [1000, 4097, 65536, 100003]:
        order = _permutation_sample(n, 11, np.arange(n))
        assert np.array_equal(np.sort(order), np.arange(n, dtype=np.uint64))

    assert not np.array_equal(_permutation_sample(1000, 1, np.arange(1000)), _permutation_sample(1000, 2, np.arange(1000)))

    n_rows = 6_000_000_000
    positions = np.concatenate([np.arange(50_000), np.arange(n_rows - 50_000, n_rows)])
    order = _permutation_sample(n_rows, 5, positions)
    assert (order < n_rows).all() and len(np.unique(order)) == len(order)
    assert (order >= 2 ** 32).any()


def test_interaction_storage_layouts():
    """training reads the seen-items index directly when it can, and keeps repeated rows separate when it cannot"""

    rng = np.random.default_rng(8)
    n_users, n_items, per_user = 60, 30, 6
    users = np.repeat(np.arange(n_users), per_user)
    items = np.concatenate([np.sort(rng.choice(n_items, per_user, replace=False)) for _ in range(n_users)])
    sorted_pairs = pd.DataFrame({'user_id': users, 'item_id': items})
    weights = rng.uniform(0.5, 1.5, len(users)).astype(np.float32)

    # sorted, unique: the item column is the index, with or without weights (which keep their input order)
    model = RankFM(factors=4).fit(sorted_pairs, epochs=1, seed=1)
    assert model._intx_i is model._ui_indices and model.sample_weight is None
    assert np.array_equal(model.interactions, sorted_pairs.to_numpy())
    reference_index = (model._ui_indptr.copy(), model._ui_indices.copy())

    weighted = RankFM(factors=4).fit(sorted_pairs, sample_weight=weights, epochs=1, seed=1)
    assert weighted._intx_i is weighted._ui_indices
    assert np.array_equal(weighted.sample_weight, weights)

    # unique but shuffled, no weights: training still reads the sorted index
    order = rng.permutation(len(users))
    shuffled = RankFM(factors=4).fit(sorted_pairs.iloc[order], epochs=1, seed=1)
    assert shuffled._intx_i is shuffled._ui_indices
    assert np.array_equal(shuffled._ui_indptr, reference_index[0]) and np.array_equal(shuffled._ui_indices, reference_index[1])
    assert np.array_equal(shuffled.interactions, sorted_pairs.to_numpy())

    # shuffled with weights: pairs and weights stay aligned in input order
    shuffled_weighted = RankFM(factors=4).fit(sorted_pairs.iloc[order], sample_weight=weights[order], epochs=1, seed=1)
    assert np.array_equal(shuffled_weighted.interactions, sorted_pairs.to_numpy()[order])
    assert np.array_equal(shuffled_weighted.sample_weight, weights[order])
    assert np.array_equal(shuffled_weighted._ui_indices, reference_index[1])

    # repeated pairs remain separate training rows, while the index stays deduplicated
    repeated = pd.concat([sorted_pairs, sorted_pairs.iloc[:50]])
    with_repeats = RankFM(factors=4).fit(repeated, epochs=1, seed=1)
    assert len(with_repeats.interactions) == len(repeated)
    assert np.array_equal(with_repeats._ui_indices, reference_index[1])


def test_model_does_not_alias_caller_interaction_arrays():
    """changing the input arrays after fitting must not change the model's seen-items index"""

    users = np.repeat(np.arange(20, dtype=np.int32), 3)
    items = np.tile(np.array([1, 4, 7], dtype=np.int32), 20)
    pairs = np.ascontiguousarray(np.column_stack([users, items]))

    model = RankFM(factors=4).fit(pairs, epochs=1, seed=1)
    index_before = model._ui_indices.copy()
    pairs[:, 1] = 0
    assert np.array_equal(model._ui_indices, index_before)

    sp = pytest.importorskip("scipy.sparse")
    matrix = sp.csr_matrix((np.ones(60, dtype=np.float32), (users, items)), shape=(20, 10))
    sparse_model = RankFM(factors=4).fit(matrix, epochs=1, seed=1)
    assert sparse_model.sample_weight is None
    sparse_index_before = sparse_model._ui_indices.copy()
    matrix.indices[:] = 0
    assert np.array_equal(sparse_model._ui_indices, sparse_index_before)
    assert sparse_index_before[:3].tolist() == [1, 4, 7]


def _dense_feature_dataset(n_users=3000, n_items=200, n_clusters=8, per_user=10, n_dense=6, seed=12):
    """clustered interactions plus a dense real-valued user feature block that identifies each user's cluster"""

    rng = np.random.default_rng(seed)
    u_cluster = rng.integers(0, n_clusters, n_users)
    i_cluster = np.arange(n_items) % n_clusters
    users = np.repeat(np.arange(n_users), per_user)
    in_cluster = rng.integers(0, n_items // n_clusters, len(users)) * n_clusters + u_cluster[users]
    items = np.where(rng.random(len(users)) < 0.85, in_cluster, rng.integers(0, n_items, len(users)))
    centroids = rng.normal(0, 1, (n_clusters, n_dense))
    dense = (centroids[u_cluster] + rng.normal(0, 0.3, (n_users, n_dense))).astype(np.float32)
    dense /= np.abs(dense).max()
    return pd.DataFrame({'user_id': users, 'item_id': items}), dense, u_cluster, i_cluster


def test_hot_row_selection_picks_dense_columns_only():
    """only columns present on at least 1/16 of rows are treated as densely used"""

    from rankfm._rankfm import _select_hot_rows

    column_of_each_entry = np.concatenate([np.full(900, 0), np.full(70, 1), np.full(10, 2), np.full(500, 4)]).astype(np.int32)
    slots, hot = _select_hot_rows(column_of_each_entry, 5, 1000)
    assert hot.tolist() == [0, 1, 4]
    assert slots.tolist() == [0, 1, -1, -1, 2]


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_batched_dense_updates_match_lockfree_quality(n_jobs):
    """batched updates for dense feature columns learn the features as well as per-step updates do"""

    interactions, dense, u_cluster, i_cluster = _dense_feature_dataset()
    train_users = 2700
    train = interactions[interactions['user_id'] < train_users]
    cold_users = np.arange(train_users, 3000)

    precision = {}
    for mode in ('lockfree', 'batched'):
        model = RankFM(factors=12, loss='warp', max_samples=10, learning_rate=0.1, beta=0.01, user_dropout=0.3, dense_updates=mode)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            model.fit(train, user_features=(np.arange(train_users), dense[:train_users]), epochs=8, seed=1, n_jobs=n_jobs)
        recs = model.recommend(cold_users, n_items=5, cold_start='estimate', user_features=(cold_users, dense[train_users:])).to_numpy()
        precision[mode] = float((i_cluster[recs] == u_cluster[cold_users][:, None]).mean())

    assert precision['lockfree'] > 0.6
    assert precision['batched'] > precision['lockfree'] - 0.1


def test_batched_dense_updates_stay_stable_at_high_learning_rate():
    """the rate cap keeps batched updates bounded where an uncapped sum of concurrent batches would overshoot"""

    interactions, dense, _, _ = _dense_feature_dataset()
    model = RankFM(factors=12, loss='warp', learning_rate=0.5, beta=0.01, dense_updates='batched')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        model.fit(interactions, user_features=(np.arange(3000), dense), epochs=6, seed=1, n_jobs=4)
    assert np.abs(model.v_uf).max() < 10.0 and np.abs(model.v_u).max() < 10.0


def test_batched_dense_updates_are_reproducible_and_validated():
    """single-threaded batched training is deterministic, and the option rejects unsupported combinations"""

    interactions, dense, _, _ = _dense_feature_dataset(n_users=600)
    features = (np.arange(600), dense)
    first = RankFM(factors=6, dense_updates='batched').fit(interactions, user_features=features, epochs=2, seed=3, n_jobs=1)
    second = RankFM(factors=6, dense_updates='batched').fit(interactions, user_features=features, epochs=2, seed=3, n_jobs=1)
    assert np.array_equal(first.v_uf, second.v_uf) and np.array_equal(first.v_u, second.v_u)
    assert np.abs(first.v_uf).max() > 0.02  # the dense rows did learn

    with pytest.raises(ValueError):
        RankFM(dense_updates='batched', feature_reg='uniform')
    with pytest.raises(ValueError):
        RankFM(dense_updates='sometimes')
