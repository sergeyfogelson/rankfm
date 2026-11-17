import numpy as np
import pandas as pd
import pytest

from rankfm.rankfm import RankFM
from rankfm.evaluation import (
    hit_rate,
    reciprocal_rank,
    discounted_cumulative_gain,
    precision,
    recall,
    diversity,
)


def _make_clustered_dataset(seed=42):
    """create a synthetic dataset with two distinct user/item clusters plus sub-preferences"""

    rng = np.random.default_rng(seed)

    # Items 0..5 belong to cluster A (feature [1, 0]); Items 6..11 belong to cluster B (feature [0, 1])
    # Within each cluster, first 3 items are sub-type X, next 3 items are sub-type Y
    item_rows = []
    for item_id in range(12):
        is_a = 1.0 if item_id < 6 else 0.0
        is_b = 1.0 - is_a
        sub_x = 1.0 if (item_id % 6) < 3 else 0.0
        sub_y = 1.0 - sub_x
        item_rows.append((item_id, is_a, is_b, sub_x, sub_y))
    item_features = pd.DataFrame(item_rows, columns=['item_id', 'is_a', 'is_b', 'sub_x', 'sub_y'])

    # Training users 0..19 in cluster A, 20..39 in cluster B
    # User features only expose coarse cluster [seg_a, seg_b], NOT fine-grained sub-type X/Y!
    # That way:
    # - user_features alone identify cluster A vs B
    # - folding in 1-2 interactions further identifies sub-type X vs Y within the cluster!
    user_rows = []
    intx_rows = []
    for u in range(40):
        seg_a = 1.0 if u < 20 else 0.0
        seg_b = 1.0 - seg_a
        user_rows.append((u, seg_a, seg_b))

        prefers_x = (u % 2 == 0)
        if seg_a == 1.0:
            pool = [0, 1, 2] if prefers_x else [3, 4, 5]
        else:
            pool = [6, 7, 8] if prefers_x else [9, 10, 11]

        chosen = rng.choice(pool, size=3, replace=False)
        for item_id in chosen:
            intx_rows.append((u, int(item_id)))

    user_features = pd.DataFrame(user_rows, columns=['user_id', 'seg_a', 'seg_b'])
    interactions = pd.DataFrame(intx_rows, columns=['user_id', 'item_id'])
    return interactions, user_features, item_features


def test_cold_start_user_features_only():
    """unseen users with only user_features receive cluster-appropriate scores & recommendations without mutating model"""

    interactions, user_features, item_features = _make_clustered_dataset()
    np.random.seed(7)
    model = RankFM(factors=8, loss='warp', max_samples=5, alpha=0.01, beta=0.01, learning_rate=0.1, user_dropout=0.25)
    model.fit(interactions, user_features=user_features, item_features=item_features, epochs=30)

    orig_v_u = model.v_u.copy()
    orig_v_i = model.v_i.copy()
    orig_users = model.user_id.copy()

    # Unseen users 100 (cluster A) and 101 (cluster B) with reversed column order to test column alignment
    new_uf = pd.DataFrame([
        (100, 0.0, 1.0),
        (101, 1.0, 0.0),
    ], columns=['user_id', 'seg_b', 'seg_a'])

    recs = model.recommend([100, 101], n_items=4, user_features=new_uf)
    assert recs.shape == (2, 4)
    assert not recs.isnull().any().any()

    # User 100 (seg_a=1) should predominantly get cluster A items (0..5); User 101 (seg_b=1) cluster B items (6..11)
    assert (recs.loc[100] < 6).sum() >= 3
    assert (recs.loc[101] >= 6).sum() >= 3

    # Verify model state was not mutated by stateless inference
    assert np.array_equal(model.v_u, orig_v_u)
    assert np.array_equal(model.v_i, orig_v_i)
    assert np.array_equal(model.user_id.values, orig_users.values)


def test_unseen_users_with_both_features_and_interactions():
    """unseen users with BOTH user_features and recent interactions leverage both feature prior and folded-in residual"""

    interactions, user_features, item_features = _make_clustered_dataset()
    np.random.seed(11)
    model = RankFM(factors=8, loss='warp', max_samples=5, alpha=0.01, beta=0.01, learning_rate=0.1, user_dropout=0.2)
    model.fit(interactions, user_features=user_features, item_features=item_features, epochs=35)

    # Unseen user 200 has cluster A user_features AND interacted with sub-type X items [0, 1]
    # Unseen user 201 has cluster A user_features AND interacted with sub-type Y items [3, 4]
    new_uf = pd.DataFrame([
        (200, 1.0, 0.0),
        (201, 1.0, 0.0),
    ], columns=['user_id', 'seg_a', 'seg_b'])

    new_intx = pd.DataFrame([
        (200, 0), (200, 1),
        (201, 3), (201, 4),
    ], columns=['user_id', 'item_id'])

    # Recommend remaining unseen items with filter_previous=True
    recs = model.recommend(
        [200, 201],
        n_items=2,
        filter_previous=True,
        user_features=new_uf,
        interactions=new_intx,
        fold_in_epochs=25
    )

    # Folded-in items must be filtered out
    assert 0 not in recs.loc[200].values and 1 not in recs.loc[200].values
    assert 3 not in recs.loc[201].values and 4 not in recs.loc[201].values

    # Remaining sub-type X item for user 200 is item 2; remaining sub-type Y item for user 201 is item 5
    assert 2 in recs.loc[200].values
    assert 5 in recs.loc[201].values

    # Pointwise predict() on unseen users with both features and interactions should match top ordering
    pairs = pd.DataFrame([(200, 2), (200, 11), (201, 5), (201, 11)], columns=['user_id', 'item_id'])
    scores = model.predict(pairs, user_features=new_uf, interactions=new_intx, fold_in_epochs=25)
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
    assert scores[0] > scores[1]
    assert scores[2] > scores[3]


def test_mixed_batch_and_cold_start_modes():
    """test batch containing training user, unseen user with data, and completely unknown user across cold_start modes"""

    interactions, user_features, item_features = _make_clustered_dataset()
    np.random.seed(3)
    model = RankFM(factors=4, loss='bpr', learning_rate=0.1, user_dropout=0.1)
    model.fit(interactions, user_features=user_features, item_features=item_features, epochs=15)

    new_uf = pd.DataFrame([(500, 1.0, 0.0)], columns=['user_id', 'seg_a', 'seg_b'])
    new_intx = pd.DataFrame([(500, 0)], columns=['user_id', 'item_id'])
    batch_users = [0, 500, 999]  # 0=trained, 500=unseen w/ features+intx, 999=completely unknown

    # 'nan': user 0 and 500 scored, 999 is NaN
    recs_nan = model.recommend(batch_users, n_items=3, cold_start='nan', user_features=new_uf, interactions=new_intx)
    assert recs_nan.shape == (3, 3)
    assert recs_nan.loc[[0, 500]].notnull().all().all()
    assert recs_nan.loc[999].isnull().all()

    # 'drop': user 999 dropped, 0 and 500 kept
    recs_drop = model.recommend(batch_users, n_items=3, cold_start='drop', user_features=new_uf, interactions=new_intx)
    assert list(recs_drop.index) == [0, 500]
    assert recs_drop.notnull().all().all()

    # 'estimate': even completely unknown user 999 gets item-bias/item-feature baseline recommendations
    recs_est = model.recommend(batch_users, n_items=3, cold_start='estimate', user_features=new_uf, interactions=new_intx)
    assert recs_est.shape == (3, 3)
    assert recs_est.notnull().all().all()

    # Verify unknown user 999 scores under cold_start='estimate' equal exact item_bias (w_i + x_if @ w_if)
    pairs_999 = pd.DataFrame([(999, i) for i in range(12)], columns=['user_id', 'item_id'])
    scores_999 = model.predict(pairs_999, cold_start='estimate')
    expected_bias = model.w_i + model.x_if @ model.w_if
    np.testing.assert_allclose(scores_999, expected_bias, rtol=1e-5, atol=1e-5)


def test_stateful_fit_partial_expands_unseen_users():
    """fit_partial registers unseen users (with freeze_items=True or False) for subsequent standard predict/recommend"""

    interactions, user_features, item_features = _make_clustered_dataset()
    np.random.seed(19)
    model = RankFM(factors=6, loss='warp', max_samples=5, learning_rate=0.1, user_dropout=0.2)
    model.fit(interactions, user_features=user_features, item_features=item_features, epochs=20)

    w_i_before = model.w_i.copy()
    v_i_before = model.v_i.copy()
    v_uf_before = model.v_uf.copy()
    v_if_before = model.v_if.copy()

    # Register unseen users 300 and 301 statefully with freeze_items=True
    partial_uf = pd.DataFrame([
        (300, 1.0, 0.0),
        (301, 0.0, 1.0),
    ], columns=['user_id', 'seg_a', 'seg_b'])
    partial_intx = pd.DataFrame([
        (300, 0), (300, 1),
        (301, 6), (301, 7),
    ], columns=['user_id', 'item_id'])

    model.fit_partial(partial_intx, user_features=partial_uf, epochs=20, freeze_items=True)

    # Item and feature weights must be untouched when freeze_items=True
    np.testing.assert_array_equal(model.w_i, w_i_before)
    np.testing.assert_array_equal(model.v_i, v_i_before)
    np.testing.assert_array_equal(model.v_uf, v_uf_before)
    np.testing.assert_array_equal(model.v_if, v_if_before)

    # Users 300 and 301 are now permanently part of the model
    assert 300 in model.user_to_index and 301 in model.user_to_index
    recs = model.recommend([300, 301], n_items=3, filter_previous=True)
    assert recs.shape == (2, 3)
    assert recs.notnull().all().all()
    assert 0 not in recs.loc[300].values and 1 not in recs.loc[300].values
    assert (recs.loc[300] < 6).sum() >= 2
    assert (recs.loc[301] >= 6).sum() >= 2

    # Also test full joint fit_partial (freeze_items=False) with string user IDs
    str_intx = pd.DataFrame([('u1', 'i1'), ('u1', 'i2'), ('u2', 'i2'), ('u2', 'i3')], columns=['user_id', 'item_id'])
    m_str = RankFM(factors=3)
    m_str.fit(str_intx, epochs=5)
    m_str.fit_partial(pd.DataFrame([('u3', 'i1'), ('u3', 'i3')], columns=['user_id', 'item_id']), epochs=5, freeze_items=False)
    assert 'u3' in m_str.user_to_index
    str_recs = m_str.recommend(['u3'], n_items=1, filter_previous=True)
    assert str_recs.loc['u3', 0] == 'i2'


def test_embed_users_and_similar_users_for_unseen_user():
    """embed_users and similar_users work for unseen users with features and/or interactions"""

    interactions, user_features, item_features = _make_clustered_dataset()
    np.random.seed(23)
    model = RankFM(factors=6, loss='warp', max_samples=5, learning_rate=0.1, user_dropout=0.2)
    model.fit(interactions, user_features=user_features, item_features=item_features, epochs=25)

    new_uf = pd.DataFrame([(777, 1.0, 0.0)], columns=['user_id', 'seg_a', 'seg_b'])
    new_intx = pd.DataFrame([(777, 0), (777, 1)], columns=['user_id', 'item_id'])

    emb = model.embed_users([777], user_features=new_uf, interactions=new_intx, fold_in_epochs=15)
    assert emb.shape == (1, 6)
    assert np.isfinite(emb.values).all()

    sim_users = model.similar_users(777, n_users=5, user_features=new_uf, interactions=new_intx, fold_in_epochs=15)
    assert sim_users.shape == (5,)
    # Cluster A training users are 0..19; majority of nearest neighbors should be in cluster A
    assert (sim_users < 20).sum() >= 4


def test_evaluation_metrics_on_unseen_users():
    """evaluation module metrics evaluate held-out unseen users when user_features and warm-start interactions are supplied"""

    interactions, user_features, item_features = _make_clustered_dataset()
    np.random.seed(29)
    model = RankFM(factors=8, loss='warp', max_samples=5, learning_rate=0.1, user_dropout=0.2)
    model.fit(interactions, user_features=user_features, item_features=item_features, epochs=30)

    test_uf = pd.DataFrame([
        (800, 1.0, 0.0),
        (801, 0.0, 1.0),
    ], columns=['user_id', 'seg_a', 'seg_b'])
    warm_intx = pd.DataFrame([
        (800, 0),
        (801, 6),
    ], columns=['user_id', 'item_id'])
    holdout_intx = pd.DataFrame([
        (800, 1), (800, 2),
        (801, 7), (801, 8),
    ], columns=['user_id', 'item_id'])

    hr = hit_rate(model, holdout_intx, k=3, filter_previous=True, user_features=test_uf, interactions=warm_intx, fold_in_epochs=20)
    rr = reciprocal_rank(model, holdout_intx, k=3, filter_previous=True, user_features=test_uf, interactions=warm_intx, fold_in_epochs=20)
    dcg = discounted_cumulative_gain(model, holdout_intx, k=3, filter_previous=True, user_features=test_uf, interactions=warm_intx, fold_in_epochs=20)
    prec = precision(model, holdout_intx, k=3, filter_previous=True, user_features=test_uf, interactions=warm_intx, fold_in_epochs=20)
    rec = recall(model, holdout_intx, k=3, filter_previous=True, user_features=test_uf, interactions=warm_intx, fold_in_epochs=20)
    div = diversity(model, holdout_intx, k=3, filter_previous=True, user_features=test_uf, interactions=warm_intx, fold_in_epochs=20)

    assert hr == 1.0
    assert rr > 0.5
    assert dcg > 0.5
    assert prec > 0.3
    assert rec >= 0.5
    assert isinstance(div, pd.DataFrame) and div['cnt_users'].sum() == 6
