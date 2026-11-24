"""
rankfm model tuning and evaluation functions
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from rankfm.utils import get_data, require


def _extract_test_user_items(test_interactions):
    """fast construction of {user_id: set(item_ids)} without slow pandas groupby.apply(set)"""

    if isinstance(test_interactions, pd.DataFrame):
        u_list = test_interactions.iloc[:, 0].tolist()
        i_list = test_interactions.iloc[:, 1].tolist()
    else:
        arr = get_data(test_interactions)
        u_list = arr[:, 0].tolist()
        i_list = arr[:, 1].tolist()

    user_items = defaultdict(set)
    for u, i in zip(u_list, i_list):
        user_items[u].add(i)
    return user_items


def _get_eval_recs(
    model,
    test_users,
    k,
    filter_previous,
    user_features=None,
    interactions=None,
    sample_weight=None,
    fold_in_epochs=5,
    cold_start='drop'
):
    """generate top-K evaluation recommendations supporting both seen and unseen test users"""

    return model.recommend(
        users=test_users,
        n_items=k,
        filter_previous=filter_previous,
        cold_start=cold_start,
        user_features=user_features,
        interactions=interactions,
        sample_weight=sample_weight,
        fold_in_epochs=fold_in_epochs
    )


def hit_rate(
    model,
    test_interactions,
    k=10,
    filter_previous=False,
    user_features=None,
    interactions=None,
    sample_weight=None,
    fold_in_epochs=5,
    cold_start='drop'
):
    """evaluate hit-rate (any match) wrt out-of-sample observed interactions"""

    require(model.is_fit, "you must fit the model prior to evaluating hold-out metrics")

    test_user_items = _extract_test_user_items(test_interactions)
    test_users = list(test_user_items.keys())

    test_recs = _get_eval_recs(
        model, test_users, k, filter_previous,
        user_features=user_features, interactions=interactions,
        sample_weight=sample_weight, fold_in_epochs=fold_in_epochs, cold_start=cold_start
    )
    comm_users = test_recs.index.tolist()
    if not comm_users:
        return np.nan

    rec_rows = test_recs.to_numpy().tolist()
    hits = 0
    for u, row in zip(comm_users, rec_rows):
        targets = test_user_items[u]
        for item in row:
            if item in targets:
                hits += 1
                break
    return hits / len(comm_users)


def reciprocal_rank(
    model,
    test_interactions,
    k=10,
    filter_previous=False,
    user_features=None,
    interactions=None,
    sample_weight=None,
    fold_in_epochs=5,
    cold_start='drop'
):
    """evaluate reciprocal rank wrt out-of-sample observed interactions"""

    require(model.is_fit, "you must fit the model prior to evaluating hold-out metrics")

    test_user_items = _extract_test_user_items(test_interactions)
    test_users = list(test_user_items.keys())

    test_recs = _get_eval_recs(
        model, test_users, k, filter_previous,
        user_features=user_features, interactions=interactions,
        sample_weight=sample_weight, fold_in_epochs=fold_in_epochs, cold_start=cold_start
    )
    comm_users = test_recs.index.tolist()
    if not comm_users:
        return np.nan

    rec_rows = test_recs.to_numpy().tolist()
    rr_sum = 0.0
    for u, row in zip(comm_users, rec_rows):
        targets = test_user_items[u]
        for rank_idx, item in enumerate(row):
            if item in targets:
                rr_sum += 1.0 / (rank_idx + 1)
                break
    return rr_sum / len(comm_users)


def discounted_cumulative_gain(
    model,
    test_interactions,
    k=10,
    filter_previous=False,
    user_features=None,
    interactions=None,
    sample_weight=None,
    fold_in_epochs=5,
    cold_start='drop'
):
    """evaluate discounted cumulative gain wrt out-of-sample observed interactions"""

    require(model.is_fit, "you must fit the model prior to evaluating hold-out metrics")

    test_user_items = _extract_test_user_items(test_interactions)
    test_users = list(test_user_items.keys())

    test_recs = _get_eval_recs(
        model, test_users, k, filter_previous,
        user_features=user_features, interactions=interactions,
        sample_weight=sample_weight, fold_in_epochs=fold_in_epochs, cold_start=cold_start
    )
    comm_users = test_recs.index.tolist()
    if not comm_users:
        return np.nan

    discounts = (1.0 / np.log2(np.arange(test_recs.shape[1]) + 2.0)).tolist()
    rec_rows = test_recs.to_numpy().tolist()
    dcg_sum = 0.0
    for u, row in zip(comm_users, rec_rows):
        targets = test_user_items[u]
        for rank_idx, item in enumerate(row):
            if item in targets:
                dcg_sum += discounts[rank_idx]
    return dcg_sum / len(comm_users)


def precision(
    model,
    test_interactions,
    k=10,
    filter_previous=False,
    user_features=None,
    interactions=None,
    sample_weight=None,
    fold_in_epochs=5,
    cold_start='drop'
):
    """evaluate precision wrt out-of-sample observed interactions"""

    require(model.is_fit, "you must fit the model prior to evaluating hold-out metrics")

    test_user_items = _extract_test_user_items(test_interactions)
    test_users = list(test_user_items.keys())

    test_recs = _get_eval_recs(
        model, test_users, k, filter_previous,
        user_features=user_features, interactions=interactions,
        sample_weight=sample_weight, fold_in_epochs=fold_in_epochs, cold_start=cold_start
    )
    comm_users = test_recs.index.tolist()
    if not comm_users:
        return np.nan

    n_cols = test_recs.shape[1]
    rec_rows = test_recs.to_numpy().tolist()
    prec_sum = 0.0
    for u, row in zip(comm_users, rec_rows):
        targets = test_user_items[u]
        matched = sum(1 for item in row if item in targets)
        prec_sum += matched / n_cols
    return prec_sum / len(comm_users)


def recall(
    model,
    test_interactions,
    k=10,
    filter_previous=False,
    user_features=None,
    interactions=None,
    sample_weight=None,
    fold_in_epochs=5,
    cold_start='drop'
):
    """evaluate recall wrt out-of-sample observed interactions"""

    require(model.is_fit, "you must fit the model prior to evaluating hold-out metrics")

    test_user_items = _extract_test_user_items(test_interactions)
    test_users = list(test_user_items.keys())

    test_recs = _get_eval_recs(
        model, test_users, k, filter_previous,
        user_features=user_features, interactions=interactions,
        sample_weight=sample_weight, fold_in_epochs=fold_in_epochs, cold_start=cold_start
    )
    comm_users = test_recs.index.tolist()
    if not comm_users:
        return np.nan

    rec_rows = test_recs.to_numpy().tolist()
    rec_sum = 0.0
    for u, row in zip(comm_users, rec_rows):
        targets = test_user_items[u]
        matched = sum(1 for item in row if item in targets)
        rec_sum += matched / len(targets)
    return rec_sum / len(comm_users)


def diversity(
    model,
    test_interactions,
    k=10,
    filter_previous=False,
    user_features=None,
    interactions=None,
    sample_weight=None,
    fold_in_epochs=5,
    cold_start='drop'
):
    """evaluate the diversity of the model recommendations"""

    require(model.is_fit, "you must fit the model prior to evaluating hold-out metrics")

    if isinstance(test_interactions, pd.DataFrame):
        test_users = pd.unique(test_interactions.iloc[:, 0].to_numpy())
    else:
        test_users = pd.unique(get_data(test_interactions)[:, 0])

    test_recs = _get_eval_recs(
        model, test_users, k, filter_previous,
        user_features=user_features, interactions=interactions,
        sample_weight=sample_weight, fold_in_epochs=fold_in_epochs, cold_start=cold_start
    )
    n_comm = len(test_recs)

    flat_items = test_recs.to_numpy().ravel()
    item_idx = model.item_to_index.index.get_indexer(flat_items)
    counts = np.bincount(item_idx[item_idx >= 0], minlength=len(model.item_id))

    order = np.argsort(-counts, kind='mergesort')
    user_counts = pd.DataFrame({
        'index': model.item_id.values[order],
        'cnt_users': counts[order],
        'pct_users': counts[order] / n_comm if n_comm > 0 else 0.0
    })
    return user_counts
