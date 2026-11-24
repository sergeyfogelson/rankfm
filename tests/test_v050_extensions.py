import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import rankfm
from rankfm import RankFM


def test_v040_regression_and_mode_flag():
    """Default RankFM v0.5.0 preserves exact v0.4.0 mode flag and version metadata"""
    assert rankfm.__version__ == "0.5.0"

    m_default = RankFM(factors=8, loss="warp", max_samples=5)
    assert not m_default._is_v050_mode

    m_cos = RankFM(factors=8, loss="cosine_nce")
    assert m_cos._is_v050_mode and m_cos._use_cosine

    m_hist = RankFM(factors=8, loss="warp", use_history=True)
    assert m_hist._is_v050_mode and not m_hist._use_cosine


def test_asymmetric_history_seq_anchor_event_gates_and_id_free_inductive():
    """Phase 1: Asymmetric P_hist + causal P_seq + typed event gates + user_dropout=1.0 inductive parity"""
    rng = np.random.default_rng(101)
    n_chains = 6
    chain_len = 8  # 48 items partitioned into 6 directed Markov chains: c*8 + 0 -> 1 -> ... -> 7
    n_items = n_chains * chain_len

    rows_intx = []
    rows_prev = []
    rows_hist = []

    user_id = 0
    for _ in range(90):
        for c in range(n_chains):
            base = c * chain_len
            start_step = int(rng.integers(0, 4))
            seq = [base + s for s in range(start_step, start_step + 5)]
            for pos, it in enumerate(seq):
                rows_intx.append((user_id, it))
                rows_prev.append(seq[pos - 1] if pos > 0 else -1)
                # Event kind 0 = main watch, kind 1 = high-intent watchlist on chain items
                rows_hist.append((user_id, it, 1.5, 0))
                if pos == 0:
                    rows_hist.append((user_id, seq[-1], 0.8, 1))
            # Add random cross-chain noise on event kind 2 so g_event learns kind 0/1 > kind 2
            noisy_item = int(rng.integers(0, n_items))
            rows_hist.append((user_id, noisy_item, 0.5, 2))
            user_id += 1

    intx_df = pd.DataFrame(rows_intx, columns=["user_id", "item_id"])
    hist_df = pd.DataFrame(rows_hist, columns=["user_id", "item_id", "weight", "event_type"])

    model = RankFM(
        factors=24,
        loss="cosine_nce",
        max_samples=20,
        temperature=0.12,
        learning_rate=0.12,
        alpha=1e-4,
        beta=1e-3,
        use_history=True,
        use_seq_anchor=True,
        n_event_types=3,
        user_dropout=1.0,
        n_jobs=2,
    )
    model.fit(
        intx_df,
        user_history=hist_df,
        prev_items=np.asarray(rows_prev, dtype=np.int32),
        epochs=24,
        seed=77,
    )

    assert model.p_hist.shape == (n_items, 24)
    assert model.p_seq.shape == (n_items, 24)
    assert model.g_event.shape == (3, 24)
    assert np.all(np.isfinite(model.p_hist))
    assert np.all(np.isfinite(model.p_seq))

    # Pure inductive zero-shot evaluation on brand-new unseen users traversing chains
    cold_users = [10000 + c for c in range(n_chains)]
    cold_hist_rows = []
    cold_last = {}
    expected_next = {}
    for c in range(n_chains):
        u = 10000 + c
        base = c * chain_len
        # User watched step 0, 1, 2 -> next items in chain are {base+3, base+4, ...}
        for step in (0, 1, 2):
            cold_hist_rows.append((u, base + step, 1.5, 0))
        cold_last[u] = base + 2
        expected_next[u] = set(range(base + 3, base + chain_len))

    cold_hist_df = pd.DataFrame(cold_hist_rows, columns=["user_id", "item_id", "weight", "event_type"])
    recs = model.recommend(
        cold_users,
        n_items=4,
        filter_previous=True,
        user_history=cold_hist_df,
        last_items=cold_last,
    )
    assert recs.shape == (n_chains, 4)

    chain_hits = 0
    for c in range(n_chains):
        u = 10000 + c
        got = set(recs.loc[u].to_numpy(dtype=int))
        # Must filter seen items (base+0, base+1, base+2)
        assert len(got.intersection({c * chain_len + s for s in (0, 1, 2)})) == 0
        chain_hits += len(got.intersection(expected_next[u]))
    assert chain_hits >= int(0.75 * n_chains * 4)

    # Verify user_dropout=1.0 pure inductive parity: training user 0 with overridden history == cold user 10000
    u0_override_hist = pd.DataFrame(
        [(0, s, 1.5, 0) for s in (0, 1, 2)],
        columns=["user_id", "item_id", "weight", "event_type"],
    )
    emb_train_override = model.embed_users([0], user_history=u0_override_hist, last_items={0: 2})
    emb_cold = model.embed_users([10000], user_history=cold_hist_df, last_items={10000: 2})
    np.testing.assert_allclose(emb_train_override.to_numpy(), emb_cold.to_numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("loss_name", ["cosine_nce", "simplex"])
def test_cosine_nce_simplex_alias_logq_and_mips_parity(loss_name):
    """Phase 1: Cosine InfoNCE/SimpleX + O(1) Alias popularity sampling + LogQ debiasing + exact MIPS parity"""
    rng = np.random.default_rng(202)
    n_users = 320
    n_items = 100
    n_clusters = 5

    # Strong Zipfian popularity skew within each cluster (first 2 items per cluster are mega-hits, rest are long-tail)
    item_cluster = np.arange(n_items) % n_clusters
    item_rank_in_cluster = np.arange(n_items) // n_clusters
    pop_weights = 1.0 / np.power(item_rank_in_cluster + 1.0, 1.35)

    rows = []
    weights = []
    margins = []
    for u in range(n_users):
        c = u % n_clusters
        cand = np.flatnonzero(item_cluster == c)
        probs = pop_weights[cand] / pop_weights[cand].sum()
        picked = rng.choice(cand, size=7, replace=False, p=probs)
        for it in picked:
            watch_mins = float(rng.uniform(5.0, 90.0))
            rows.append((u, int(it)))
            weights.append(1.0 + 0.2 * np.log1p(watch_mins))
            margins.append(np.log2(watch_mins + 1.0))

    intx_df = pd.DataFrame(rows, columns=["user_id", "item_id"])
    sw = np.asarray(weights, dtype=np.float32)
    msig = np.asarray(margins, dtype=np.float32)

    model = RankFM(
        factors=24,
        loss=loss_name,
        max_samples=25,
        temperature=0.14,
        neg_sampling="popularity",
        neg_pop_exponent=0.75,
        neg_pop_mix=0.65,
        logq_weight=0.75,
        margin=0.03,
        weight_margin_scale=0.02,
        use_history=True,
        learning_rate=0.12,
        n_jobs=4,
    )
    model.fit(intx_df, sample_weight=sw, margin_signal=msig, epochs=15, seed=303)

    # Verify Walker-Vose alias table structure
    assert len(model._alias_prob) == n_items
    assert len(model._alias_idx) == n_items
    assert np.all((model._alias_prob >= 0.0) & (model._alias_prob <= 1.0001))
    assert len(model._log_q) == n_items

    # Verify cluster retrieval accuracy + healthy catalog diversity (> 45% catalog coverage @ top-10)
    recs = model.recommend(np.arange(100), n_items=10, filter_previous=True, n_jobs=2)
    hits = sum(int((item_cluster[recs.loc[u].to_numpy(dtype=int)] == (u % n_clusters)).sum()) for u in range(100))
    assert hits / (100 * 10) > 0.72
    unique_recommended = len(np.unique(recs.to_numpy()))
    assert unique_recommended >= 45

    # Exact parity between predict(), recommend(), and export_mips_embeddings()
    u_ids, u_mips, i_ids, i_mips = model.export_mips_embeddings()
    assert u_mips.shape == (n_users, 25)
    assert i_mips.shape == (n_items, 25)

    test_u = np.repeat(np.arange(15), n_items)
    test_i = np.tile(np.arange(n_items), 15)
    pairs_df = pd.DataFrame({"user_id": u_ids[test_u], "item_id": i_ids[test_i]})
    pred_scores = model.predict(pairs_df)
    mips_scores = (u_mips[:15] @ i_mips.T).ravel()
    np.testing.assert_allclose(pred_scores, mips_scores, rtol=1e-4, atol=1e-4)

    recs_unfiltered = model.recommend(u_ids[:15], n_items=5, filter_previous=False)
    for row_idx in range(15):
        row_scores = mips_scores[row_idx * n_items:(row_idx + 1) * n_items]
        top5_expected = i_ids[np.argsort(-row_scores)[:5]]
        np.testing.assert_array_equal(recs_unfiltered.iloc[row_idx].to_numpy(), top5_expected)


def test_field_gates_and_item_freq_cold_warm_gate():
    """Phase 2: Field-gated feature channels suppress noisy fields and frequency gate boosts 1-shot cold items"""
    rng = np.random.default_rng(404)
    n_users = 280
    n_warm_items = 60
    n_cold_items = 20
    n_items = n_warm_items + n_cold_items
    n_clusters = 4

    u_cluster = np.arange(n_users) % n_clusters
    i_cluster = np.arange(n_items) % n_clusters

    # Item features: 4 clean cluster columns (field 0) + 16 cross-cluster nuisance tag columns (field 1) + 2 dynamic momentum cols (field 2)
    nuisance_if_patterns = rng.normal(0.0, 0.40, size=(10, 16)).astype(np.float32)
    x_if = np.zeros((n_items, 22), dtype=np.float32)
    for i in range(n_items):
        x_if[i, i_cluster[i]] = 1.0
        # Items across different clusters share identical nuisance metadata tags (orthogonal to taste)
        x_if[i, 4:20] = nuisance_if_patterns[(i // n_clusters) % 10]
        x_if[i, 20:] = 0.5 if i < n_warm_items else 0.2

    if_field_ids = np.array([0] * 4 + [1] * 16 + [2] * 2, dtype=np.int32)
    if_dyn_mask = np.array([0] * 20 + [1] * 2, dtype=np.int32)

    # User features: 4 clean cluster columns (field 0) + 8 cross-cluster nuisance columns (field 1)
    nuisance_uf_patterns = rng.normal(0.0, 0.40, size=(10, 8)).astype(np.float32)
    x_uf = np.zeros((n_users, 12), dtype=np.float32)
    for u in range(n_users):
        x_uf[u, u_cluster[u]] = 1.0
        x_uf[u, 4:] = nuisance_uf_patterns[(u // n_clusters) % 10]
    uf_field_ids = np.array([0] * 4 + [1] * 8, dtype=np.int32)

    rows = []
    for u in range(n_users):
        c = u_cluster[u]
        warm_cand = np.flatnonzero((i_cluster[:n_warm_items] == c))
        chosen = rng.choice(warm_cand, size=7, replace=False)
        for it in chosen:
            rows.append((u, int(it)))

    # Give each cold item only a single interaction so v_i is nearly untrained while static X_if[field 0] is clean
    for ci in range(n_warm_items, n_items):
        u_owner = int(i_cluster[ci])
        rows.append((u_owner, ci))

    intx_df = pd.DataFrame(rows, columns=["user_id", "item_id"])
    uf_tuple = (np.arange(n_users), sp.csr_matrix(x_uf))
    if_tuple = (np.arange(n_items), sp.csr_matrix(x_if))

    model = RankFM(
        factors=20,
        loss="cosine_nce",
        max_samples=20,
        temperature=0.15,
        learning_rate=0.12,
        beta=0.01,
        use_history=True,
        use_field_gates=True,
        use_freq_gate=True,
        freq_gate_static_drop=0.8,
        n_jobs=2,
    )
    model.fit(
        intx_df,
        user_features=uf_tuple,
        item_features=if_tuple,
        user_field_ids=uf_field_ids,
        item_field_ids=if_field_ids,
        item_dynamic_mask=if_dyn_mask,
        epochs=18,
        seed=505,
    )

    assert model.gamma_if.shape == (3, 20)
    assert model.gamma_uf.shape == (2, 20)
    assert model.a_gate.shape == (20,)
    assert model.b_gate.shape == (20,)
    # Informative cluster field 0 should have strictly stronger average gate magnitude than noisy field 1
    assert float(model.gamma_if[0].mean()) > float(model.gamma_if[1].mean())

    # Cold items should match held-out cluster users (who never touched cold items) cleanly via gated metadata embeddings
    eval_users_by_c = [np.arange(40, 80)[u_cluster[40:80] == c] for c in range(n_clusters)]
    cluster_scores = np.zeros((n_cold_items, n_clusters), dtype=np.float64)
    for c in range(n_clusters):
        u_c = eval_users_by_c[c]
        pairs_c = pd.DataFrame(
            [(int(u), ci) for ci in range(n_warm_items, n_items) for u in u_c],
            columns=["user_id", "item_id"],
        )
        cluster_scores[:, c] = model.predict(pairs_c).reshape(n_cold_items, len(u_c)).mean(axis=1)
    best_u_cluster = np.argmax(cluster_scores, axis=1)
    cold_acc = float((best_u_cluster == i_cluster[n_warm_items:]).mean())
    assert cold_acc >= 0.75


@pytest.mark.parametrize("agg_temp", [0.0, 0.12])
def test_multi_interest_heads_and_multiprobe_mips_parity(agg_temp):
    """Phase 3: Multi-interest M-head user representations capture bimodal households and preserve multi-probe MIPS"""
    rng = np.random.default_rng(606)
    n_clusters = 6
    items_per_cluster = 15
    n_items = n_clusters * items_per_cluster
    i_cluster = np.arange(n_items) // items_per_cluster

    rows = []
    bimodal_targets = {}
    user_id = 0
    # Unimodal users anchor the 6 clusters cleanly
    for c in range(n_clusters):
        for _ in range(35):
            cand = np.flatnonzero(i_cluster == c)
            for it in rng.choice(cand, size=6, replace=False):
                rows.append((user_id, int(it)))
            user_id += 1

    # Bimodal household users: each consumes from two distinct clusters (c1, c2)
    bimodal_users = []
    for c1 in range(3):
        c2 = c1 + 3
        for _ in range(15):
            cand1 = np.flatnonzero(i_cluster == c1)
            cand2 = np.flatnonzero(i_cluster == c2)
            for it in rng.choice(cand1, size=4, replace=False):
                rows.append((user_id, int(it)))
            for it in rng.choice(cand2, size=4, replace=False):
                rows.append((user_id, int(it)))
            bimodal_users.append(user_id)
            bimodal_targets[user_id] = (c1, c2)
            user_id += 1

    intx_df = pd.DataFrame(rows, columns=["user_id", "item_id"])

    model = RankFM(
        factors=24,
        loss="cosine_nce",
        max_samples=20,
        temperature=0.14,
        learning_rate=0.12,
        use_history=True,
        n_interests=2,
        interest_temperature=0.20,
        head_agg_temperature=agg_temp,
        user_dropout=0.5,
        n_jobs=2,
    )
    model.fit(intx_df, epochs=18, seed=707)

    assert model.c_proto.shape == (2, 24)
    # Prototypes stay unit-normalized and distinct (low cosine overlap)
    proto_cos = abs(float(np.dot(model.c_proto[0], model.c_proto[1])))
    assert proto_cos < 0.50

    recs = model.recommend(bimodal_users, n_items=10, filter_previous=True)
    both_clusters_covered = 0
    for u in bimodal_users:
        c1, c2 = bimodal_targets[u]
        rec_clusters = set(i_cluster[recs.loc[u].to_numpy(dtype=int)])
        if c1 in rec_clusters and c2 in rec_clusters:
            both_clusters_covered += 1
    assert both_clusters_covered / len(bimodal_users) >= 0.75

    # Verify (U, M, F+1) multi-probe MIPS export exact numerical parity with predict()
    u_ids, u_mips, i_ids, i_mips = model.export_mips_embeddings(users=bimodal_users[:8])
    assert u_mips.shape == (8, 2, 25)
    assert i_mips.shape == (n_items, 25)

    head_utilities = np.einsum("umf,if->umi", u_mips, i_mips)
    if agg_temp <= 0.0:
        expected = head_utilities.max(axis=1)
    else:
        mx = head_utilities.max(axis=1, keepdims=True)
        expected = mx[:, 0, :] + agg_temp * np.log(np.exp((head_utilities - mx) / agg_temp).sum(axis=1))

    q_u = np.repeat(u_ids, n_items)
    q_i = np.tile(i_ids, len(u_ids))
    pred = model.predict(pd.DataFrame({"user_id": q_u, "item_id": q_i})).reshape(len(u_ids), n_items)
    np.testing.assert_allclose(pred, expected, rtol=1e-4, atol=1e-4)


def test_inference_addons_sparse_residual_din_attention_and_pairwise_context():
    """Phase 3 add-ons: Orthogonalized EASE residual S + DIN-lite attention + fit_pairwise_context parity"""
    rng = np.random.default_rng(808)
    n_users = 240
    n_items = 64

    rows = []
    # Items 2*p and 2*p+1 form ultra-tight franchise sequel pairs
    for u in range(n_users):
        base_pairs = rng.choice(32, size=4, replace=False)
        for p in base_pairs:
            rows.append((u, int(2 * p)))
            if rng.random() < 0.85:
                rows.append((u, int(2 * p + 1)))

    intx_df = pd.DataFrame(rows, columns=["user_id", "item_id"])

    model = RankFM(
        factors=12,
        loss="cosine_nce",
        max_samples=15,
        temperature=0.15,
        learning_rate=0.12,
        use_history=True,
        sparse_residual_k=12,
        sparse_residual_l2=80.0,
        sparse_residual_weight=0.45,
        target_attention_topk=24,
        target_attention_weight=0.20,
        n_jobs=4,
    )
    model.fit(intx_df, epochs=12, seed=909)

    assert len(model._s_indptr) == n_items + 1
    assert len(model._s_indices) > 0

    # Fit explicit 2D pairwise context ([on_watchlist, recent_click]) via Newton-Raphson IRLS
    ctx_pairs = pd.DataFrame(
        [(u % n_users, (u * 3) % n_items) for u in range(200)],
        columns=["user_id", "item_id"],
    )
    ctx_feats = np.zeros((200, 2), dtype=np.float32)
    ctx_feats[:100, 0] = 1.0  # strong positive watchlist signal
    ctx_feats[100:, 1] = 1.0  # neutral/weak signal
    ctx_labels = np.array([1.0] * 92 + [0.0] * 8 + [0.0] * 85 + [1.0] * 15, dtype=np.float32)

    w_ctx = model.fit_pairwise_context(ctx_pairs, ctx_feats, ctx_labels, l2_reg=5.0)
    assert w_ctx.shape == (2,)
    assert w_ctx[0] > 0.5 and w_ctx[0] > w_ctx[1]

    # Verify exact top-K agreement between predict() and recommend() with ALL 3 add-ons active!
    eval_users = np.arange(12)
    infer_ctx_pairs = pd.DataFrame(
        [(int(u), int((u * 5 + 7) % n_items)) for u in eval_users],
        columns=["user_id", "item_id"],
    )
    infer_ctx_feats = np.column_stack([np.ones(len(eval_users)), np.zeros(len(eval_users))]).astype(np.float32)
    pw_ctx = (infer_ctx_pairs, infer_ctx_feats)

    recs_top5 = model.recommend(eval_users, n_items=5, filter_previous=False, pairwise_context=pw_ctx, n_jobs=2)
    all_items = model._item_index.to_numpy()
    for u in eval_users:
        q_df = pd.DataFrame({"user_id": np.full(n_items, u), "item_id": all_items})
        p_scores = model.predict(q_df, pairwise_context=pw_ctx, n_jobs=2)
        expected_top5 = all_items[np.argsort(-p_scores)[:5]]
        np.testing.assert_array_equal(recs_top5.loc[u].to_numpy(), expected_top5)

    # Streaming recommend_batches() matches recommend() identically
    streamed = pd.concat(
        list(model.recommend_batches(eval_users, batch_size=5, n_items=5, filter_previous=True, pairwise_context=pw_ctx))
    )
    direct = model.recommend(eval_users, n_items=5, filter_previous=True, pairwise_context=pw_ctx)
    pd.testing.assert_frame_equal(streamed, direct)


def test_incremental_fit_partial_on_v050_model():
    """fit_partial expands unseen users cleanly in both freeze_items=True and freeze_items=False modes"""
    rng = np.random.default_rng(1010)
    rows = [(u, int(rng.integers(0, 30))) for u in range(80) for _ in range(5)]
    intx_df = pd.DataFrame(rows, columns=["user_id", "item_id"])

    model = RankFM(
        factors=16,
        loss="cosine_nce",
        max_samples=10,
        use_history=True,
        use_seq_anchor=True,
        n_jobs=2,
    )
    model.fit(intx_df, epochs=8, seed=1111)

    new_intx = pd.DataFrame([(500, 1), (500, 2), (501, 10), (501, 11)], columns=["user_id", "item_id"])
    v_i_before = model.v_i.copy()
    model.fit_partial(new_intx, epochs=5, freeze_items=True, seed=1112)
    np.testing.assert_array_equal(model.v_i, v_i_before)
    assert 500 in model.user_to_index and 501 in model.user_to_index

    recs = model.recommend([500, 501], n_items=4, filter_previous=True)
    assert recs.shape == (2, 4)
    assert recs.notnull().all().all()

    # Full incremental update (freeze_items=False) with another new user
    model.fit_partial(pd.DataFrame([(502, 3), (502, 4)], columns=["user_id", "item_id"]), epochs=4, freeze_items=False, seed=1113)
    assert 502 in model.user_to_index
    recs2 = model.recommend([502], n_items=3, filter_previous=True)
    assert recs2.shape == (1, 3) and recs2.notnull().all().all()


def test_factorized_cross_net_analytic_gradients_and_ndcg_gain():
    """Verify exact analytic gradients (SENet + 2nd-Order FM + DCN-v2) and multiplicative XOR/cross lift"""
    from rankfm import FactorizedCrossNet

    rng = np.random.default_rng(2026)
    B, D, P = 6, 8, 7
    Z = rng.normal(0.0, 1.0, size=(B * D, P)).astype(np.float32)
    gains = np.zeros((B, D), dtype=np.float32)
    gains[:, 0] = 3.0
    gains[:, 2] = 1.5
    valid_mask = np.ones((B, D), dtype=bool)
    valid_mask[:, -1] = False
    idcg10 = np.full(B, 3.95, dtype=np.float64)

    net = FactorizedCrossNet(
        fm_factors=6,
        cross_layers=2,
        cross_rank=4,
        gate_hidden=5,
        gate_scale=0.6,
        anchor_idx=0,
        feature_dropout=0.0,
        fm_l2=0.05,
        dcn_l2=0.05,
        lin_l2=0.05,
        seed=42,
    )
    params = net._init_params(P, rng)
    scores, cache = net._forward_with_cache(Z, params, training=False)
    rk_loss, dy = net.compute_ranking_loss_and_grad_scores(scores.reshape(B, D), gains, valid_mask, idcg10)
    grads, reg_loss = net._backward_from_cache(dy.ravel(), cache, params)

    # Finite-difference check on every parameter tensor (holding LambdaRank |Delta NDCG| weights fixed per Burges 2010)
    # Check pure network Jacobian d(sum(c * y_hat) + reg)/d(param) to 1e-4 precision
    probe = rng.normal(0.0, 1.0, size=scores.shape).astype(np.float32)
    grads_p, _ = net._backward_from_cache(probe, cache, params)
    eps = 1e-3
    for name, arr in params.items():
        flat_a = arr.ravel()
        flat_g = grads_p[name].ravel()
        for idx in range(min(5, len(flat_a))):
            orig = float(flat_a[idx])
            flat_a[idx] = np.float32(orig + eps)
            s_hi, c_hi = net._forward_with_cache(Z, params, training=False)
            _, r_hi = net._backward_from_cache(np.zeros_like(probe), c_hi, params)
            f_hi = float(np.dot(probe.astype(np.float64), s_hi.astype(np.float64))) + r_hi

            flat_a[idx] = np.float32(orig - eps)
            s_lo, c_lo = net._forward_with_cache(Z, params, training=False)
            _, r_lo = net._backward_from_cache(np.zeros_like(probe), c_lo, params)
            f_lo = float(np.dot(probe.astype(np.float64), s_lo.astype(np.float64))) + r_lo
            flat_a[idx] = np.float32(orig)

            fd = (f_hi - f_lo) / (2.0 * eps)
            np.testing.assert_allclose(float(flat_g[idx]), fd, rtol=2e-2, atol=2e-3, err_msg=f"Grad mismatch on {name}[{idx}]")

    # Verify multiplicative interaction learning (target = base + 2.5 * x_user_ctx * x_item_ctx)
    U_syn, D_syn, P_syn = 320, 20, 6
    X_syn = rng.normal(0.0, 1.0, size=(U_syn, D_syn, P_syn)).astype(np.float32)
    # Broadcast user context in col 1
    u_ctx = rng.choice([-1.0, 1.0], size=(U_syn, 1)).astype(np.float32)
    X_syn[:, :, 1] = u_ctx
    true_util = 0.6 * X_syn[:, :, 0] + 2.5 * (X_syn[:, :, 1] * X_syn[:, :, 2])
    top_pos = np.argmax(true_util, axis=1)
    G_syn = np.zeros((U_syn, D_syn), dtype=np.float32)
    G_syn[np.arange(U_syn), top_pos] = 4.0
    vm_syn = np.ones((U_syn, D_syn), dtype=bool)
    idcg_syn = np.full(U_syn, 4.0, dtype=np.float64)

    fit_net = FactorizedCrossNet(
        fm_factors=12, cross_layers=2, cross_rank=8, gate_hidden=8, epochs=18, batch_users=64, learning_rate=0.02, seed=99
    )
    info = fit_net.fit(
        X_syn[:240], G_syn[:240], vm_syn[:240], idcg_syn[:240],
        X_val=X_syn[240:], gains_val=G_syn[240:], valid_mask_val=vm_syn[240:], idcg10_val=idcg_syn[240:],
    )
    assert info["best_valid_ndcg10"] > info["init_valid_ndcg10"] + 0.15

    # Round-trip JSON-safe state dict parity
    reloaded = FactorizedCrossNet.from_dict(fit_net.to_dict())
    np.testing.assert_allclose(fit_net.predict_scores(X_syn[240:]), reloaded.predict_scores(X_syn[240:]), rtol=1e-5, atol=1e-5)


def test_accumulate_pair_counts_omp_parity():
    """Verify multi-threaded Cython _accumulate_pair_counts_omp matches single-thread exact counts"""
    from rankfm._rankfm import _accumulate_pair_counts_omp

    u_indptr = np.array([0, 3, 5, 6], dtype=np.int64)
    items = np.array([2, 5, 7, 5, 2, 4], dtype=np.int32)
    rec = np.array([30.0, 15.0, 5.0, 10.0, 10.0, 2.0], dtype=np.float32)

    r1 = _accumulate_pair_counts_omp(u_indptr, items, rec, 10, n_threads=1)
    r4 = _accumulate_pair_counts_omp(u_indptr, items, rec, 10, n_threads=4)
    for a, b in zip(r1, r4):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
    co, fwd, fwd21, fwd30, tfwd = r4
    # User 0 watched (2@30d, 5@15d, 7@5d); User 1 watched (5@10d, 2@10d same day -> 0.5 each way)
    assert co[2, 5] == 2.0 and co[5, 2] == 2.0
    assert np.isclose(fwd[2, 5], 1.5) and np.isclose(fwd[5, 2], 0.5)
    assert co[4, 4] == 1.0  # 1-item user diagonal preserved


def test_multi_interest_bilinear_warp_stability():
    """Verify multi-interest (M=2, 3) bilinear WARP with multi-source encoders & large weights stays strictly finite"""
    rng = np.random.default_rng(1234)
    n_u, n_i = 1200, 120
    rows = rng.integers(0, n_u, size=6000, dtype=np.int32)
    cols = rng.integers(0, n_i, size=6000, dtype=np.int32)
    ui = sp.csr_matrix((rng.uniform(0.5, 4.0, size=6000).astype(np.float32), (rows, cols)), shape=(n_u, n_i))
    ui.sum_duplicates()

    uh_u = rng.integers(0, n_u, size=12000, dtype=np.int32)
    uh_i = rng.integers(0, n_i, size=12000, dtype=np.int32)
    uh_w = rng.uniform(0.2, 3.0, size=12000).astype(np.float32)
    uh_k = rng.integers(0, 4, size=12000, dtype=np.int32)

    x_uf = rng.uniform(0.0, 2.0, size=(n_u, 10)).astype(np.float32)
    x_if = rng.uniform(0.0, 2.0, size=(n_i, 12)).astype(np.float32)

    for m_heads in (2, 3):
        model = RankFM(
            factors=48,
            loss="warp",
            max_samples=32,
            alpha=0.0012,
            beta=0.0040,
            learning_rate=0.15,
            user_dropout=0.35,
            use_history=True,
            max_history_len=48,
            use_seq_anchor=True,
            n_event_types=4,
            use_field_gates=True,
            n_interests=m_heads,
            interest_temperature=0.35,
            head_agg_temperature=0.12,
            n_jobs=2,
        )
        model.fit(
            ui,
            user_features=(pd.RangeIndex(n_u), x_uf),
            item_features=(pd.RangeIndex(n_i), x_if),
            sample_weight=ui.tocoo().data,
            epochs=4,
            user_history=(uh_u, uh_i, uh_w, uh_k),
        )
        assert np.isfinite(model.w_i).all()
        assert np.isfinite(model.v_i).all()
        assert np.isfinite(model.c_proto).all()


