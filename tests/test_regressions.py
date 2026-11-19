"""regression tests for seeding, CSR construction, small-catalog WARP, CPU detection, and batched inference"""

import warnings

import numpy as np
import pandas as pd
import pytest

import rankfm.rankfm as rankfm_module
from rankfm.rankfm import RankFM, _available_cpus


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


def test_uniform_feature_reg_warns_when_multithreaded():
    """feature_reg='uniform' cannot be honoured on several threads, so say so instead of switching silently"""

    interactions = _make_interactions(n_users=400, per_user=10)
    user_features = pd.DataFrame({'user_id': np.arange(400), 'f0': (np.arange(400) % 2).astype(float)})

    with pytest.warns(RuntimeWarning, match="uniform"):
        RankFM(factors=4, feature_reg='uniform').fit(interactions, user_features=user_features, epochs=1, n_jobs=2)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        RankFM(factors=4, feature_reg='uniform').fit(interactions, user_features=user_features, epochs=1, n_jobs=1)


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


def test_diverging_training_warns():
    """weights that explode while staying finite are reported instead of being returned silently"""

    rng = np.random.default_rng(4)
    n_users, n_items, n_feats = 300, 80, 60
    interactions = _make_interactions(n_users=n_users, n_items=n_items, per_user=8, seed=4)
    feature_values = rng.uniform(0.5, 1.0, size=(n_items, n_feats)) * (rng.random((n_items, n_feats)) < 0.4)
    item_features = (np.arange(n_items), feature_values.astype(np.float32))

    with pytest.warns(RuntimeWarning, match="diverging"):
        RankFM(factors=16, loss='warp', alpha=0.001, beta=0.001, learning_rate=0.5).fit(
            interactions, item_features=item_features, epochs=15, seed=1, n_jobs=1
        )

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        RankFM(factors=16, loss='warp', learning_rate=0.02).fit(
            interactions, item_features=item_features, epochs=15, seed=1, n_jobs=1
        )


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
