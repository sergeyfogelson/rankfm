"""Neural-era Factorized Cross-Interaction Network (`FactorizedCrossNet`) for `RankFM v0.5.0`.

Implements a 100% tree-free, pure matrix-factorization & polynomial cross-network architecture
combining three modern Factorization Machine extensions:
  1. `SENet` Dynamic Context/Field Gate (`FiBiNET`, Huang et al. RecSys 2019):
       g = 1.0 + gate_scale * tanh(tanh(z @ W_g1 + b_g1) @ W_g2 + b_g2),   z_tilde = g * z
  2. Exact Rendle 2nd-Order Factorization Machine (`FM2`, Rendle 2010 / `DeepFM`):
       y_fm2 = (0.5 / sqrt(K_fm)) * sum_k [ (z_tilde @ V_fm[:, k])^2 - (z_tilde^2 @ V_fm[:, k]^2) ]
  3. Low-Rank Deep & Cross Network v2 (`DCN-v2`, Wang et al. WWW 2021):
       h_{l+1} = h_l + z_tilde * tanh((h_l @ V_l) @ U_l + b_l),   l = 0 .. L-1
  4. Calibrated Monotone Skip Backbone + Linear Readout:
       y_hat = alpha_skip * z[:, anchor_idx] + h_L @ w_lin + y_fm2

Trained end-to-end in pure NumPy/BLAS via AdamW with Cosine LR decay and exact analytic
ListNet + Burges (2010) LambdaRank |Delta NDCG@k| gradients (zero trees, zero LightGBM).
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np


def _soft_saturate(x: np.ndarray, cap: float = 8.0) -> np.ndarray:
    inv_c = np.float32(1.0 / cap)
    return (x / (np.float32(1.0) + np.abs(x) * inv_c)).astype(np.float32, copy=False)


def _alg_tanh_with_inv_r(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast C-infinity algebraic tanh `u / sqrt(1 + u^2)` (exact derivative `inv_r^3`, zero transcendental `exp`)."""
    inv_r = (np.float32(1.0) / np.sqrt(np.float32(1.0) + u * u)).astype(np.float32, copy=False)
    return (u * inv_r).astype(np.float32, copy=False), inv_r


def _softplus(x: float) -> float:
    if x > 20.0:
        return float(x)
    if x < -20.0:
        return float(math.exp(x))
    return float(math.log1p(math.exp(x)))


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class FactorizedCrossNet:
    """Pure Factorization-Machine (`FM2`) + `SENet` + Low-Rank `DCN-v2` Cross Ranker."""

    def __init__(
        self,
        fm_factors: int = 24,
        cross_layers: int = 2,
        cross_rank: int = 16,
        gate_hidden: int = 16,
        gate_scale: float = 0.65,
        sat_cap: float = 8.0,
        anchor_idx: int | None = 0,
        min_anchor_weight: float = 0.20,
        learning_rate: float = 0.015,
        weight_decay: float = 0.015,
        fm_l2: float = 0.002,
        dcn_l2: float = 0.002,
        lin_l2: float = 0.001,
        listwise_weight: float = 0.35,
        lambdarank_weight: float = 0.65,
        lambdarank_sigma: float = 1.0,
        lambdarank_truncation: int = 25,
        batch_users: int = 512,
        epochs: int = 24,
        patience: int = 5,
        feature_dropout: float = 0.08,
        n_ensemble: int = 1,
        seed: int = 2026050,
    ) -> None:
        self.fm_factors = int(fm_factors)
        self.cross_layers = int(cross_layers)
        self.cross_rank = int(cross_rank)
        self.gate_hidden = int(gate_hidden)
        self.gate_scale = float(gate_scale)
        self.sat_cap = float(sat_cap)
        self.anchor_idx = anchor_idx
        self.min_anchor_weight = float(min_anchor_weight)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.fm_l2 = float(fm_l2)
        self.dcn_l2 = float(dcn_l2)
        self.lin_l2 = float(lin_l2)
        self.listwise_weight = float(listwise_weight)
        self.lambdarank_weight = float(lambdarank_weight)
        self.lambdarank_sigma = float(lambdarank_sigma)
        self.lambdarank_truncation = int(lambdarank_truncation)
        self.batch_users = int(batch_users)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.feature_dropout = float(feature_dropout)
        self.n_ensemble = max(1, int(n_ensemble))
        self.seed = int(seed)

        self.n_features_: int = 0
        self.feature_names_: list[str] = []
        self.mean_: np.ndarray | None = None
        self.inv_std_: np.ndarray | None = None
        self.is_binary_: np.ndarray | None = None
        self.params_: dict[str, np.ndarray] = {}
        self.ensemble_params_: list[dict[str, np.ndarray]] = []
        self.is_fitted: bool = False

    def _init_params(self, n_features: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        P = int(n_features)
        K = self.fm_factors
        R = self.cross_rank
        H = self.gate_hidden
        p: dict[str, np.ndarray] = {}

        if H > 0 and self.gate_scale > 0.0:
            p["W_g1"] = rng.normal(0.0, 0.04 / math.sqrt(max(P, 1)), size=(P, H)).astype(np.float32)
            p["b_g1"] = np.zeros(H, dtype=np.float32)
            p["W_g2"] = rng.normal(0.0, 0.03 / math.sqrt(max(H, 1)), size=(H, P)).astype(np.float32)
            p["b_g2"] = np.zeros(P, dtype=np.float32)

        if K > 0:
            p["V_fm"] = rng.normal(0.0, 0.04 / math.sqrt(max(P, 1)), size=(P, K)).astype(np.float32)

        for l_idx in range(self.cross_layers):
            p[f"V_c{l_idx}"] = rng.normal(0.0, 0.05 / math.sqrt(max(P, 1)), size=(P, R)).astype(np.float32)
            p[f"U_c{l_idx}"] = rng.normal(0.0, 0.04 / math.sqrt(max(R, 1)), size=(R, P)).astype(np.float32)
            p[f"b_c{l_idx}"] = np.zeros(P, dtype=np.float32)

        p["w_lin"] = np.zeros(P, dtype=np.float32)
        if self.anchor_idx is not None and 0 <= self.anchor_idx < P:
            # Initialize softplus(theta_skip) + min_anchor_weight ~= 0.65
            p["theta_skip"] = np.array([math.log(math.expm1(max(0.45, 0.05)))], dtype=np.float32)
        return p

    def standardize_features(
        self,
        X: np.ndarray,
        mean_override: np.ndarray | None = None,
        inv_std_override: np.ndarray | None = None,
    ) -> np.ndarray:
        """Transform raw `(..., P)` features into smooth bounded standardized space."""
        mu = mean_override if mean_override is not None else self.mean_
        istd = inv_std_override if inv_std_override is not None else self.inv_std_
        assert mu is not None and istd is not None
        orig_shape = X.shape
        X2 = np.asarray(X, dtype=np.float32).reshape(-1, orig_shape[-1])
        Z = (X2 - mu[None, :]) * istd[None, :]
        if not np.isfinite(Z).all():
            Z = np.where(np.isfinite(Z), Z, np.float32(0.0))
        Z_sat = _soft_saturate(Z, cap=self.sat_cap)
        return Z_sat.reshape(orig_shape)

    def compute_adapted_domain_stats(
        self,
        X_unlabeled: np.ndarray,
        valid_mask: np.ndarray | None = None,
        momentum: float = 0.90,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unsupervised AdaBN domain adaptation on pre-cutoff unlabeled candidates (zero label access).

        Adapts continuous feature `(mean, inv_std)` to the scoring window's pre-cutoff candidate distribution
        while keeping binary `{0, 1}` indicator columns anchored to their training scale.
        """
        assert self.mean_ is not None and self.inv_std_ is not None
        P = int(self.n_features_)
        flat_X = np.asarray(X_unlabeled, dtype=np.float32).reshape(-1, P)
        if valid_mask is not None:
            flat_v = np.asarray(valid_mask, dtype=bool).reshape(-1)
            valid_X = flat_X[flat_v] if np.any(flat_v) else flat_X
        else:
            valid_X = flat_X
        if valid_X.shape[0] < 64:
            return self.mean_.copy(), self.inv_std_.copy()

        finite_X = np.where(np.isfinite(valid_X), valid_X, 0.0)
        mean_new = finite_X.mean(axis=0, dtype=np.float64).astype(np.float32)
        std_new = finite_X.std(axis=0, dtype=np.float64).astype(np.float32)

        old_std = np.float32(1.0) / np.maximum(self.inv_std_, np.float32(1e-6))
        m = np.float32(np.clip(momentum, 0.0, 1.0))
        eff_mean = self.mean_.copy()
        eff_std = old_std.copy()

        is_bin = self.is_binary_ if (self.is_binary_ is not None and len(self.is_binary_) == P) else np.zeros(P, dtype=bool)
        cont_ok = (~is_bin) & (std_new > np.float32(1e-3))
        eff_mean[cont_ok] = (np.float32(1.0) - m) * self.mean_[cont_ok] + m * mean_new[cont_ok]
        eff_std[cont_ok] = (np.float32(1.0) - m) * old_std[cont_ok] + m * std_new[cont_ok]
        eff_inv_std = (np.float32(1.0) / np.maximum(eff_std, np.float32(1e-4))).astype(np.float32)
        return eff_mean, eff_inv_std

    def _forward_with_cache(
        self,
        Z: np.ndarray,
        params: dict[str, np.ndarray],
        rng: np.random.Generator | None = None,
        training: bool = False,
        need_cache: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Forward pass on standardized `Z (N, P)` returning `(scores (N,), cache)`."""
        N, P = Z.shape
        cache: dict[str, Any] = {"Z": Z} if need_cache else {}

        if training and self.feature_dropout > 0.0 and rng is not None:
            keep_p = np.float32(1.0 - self.feature_dropout)
            drop_mask = (rng.random((1, P), dtype=np.float32) < keep_p).astype(np.float32) / keep_p
            if self.anchor_idx is not None and 0 <= self.anchor_idx < P:
                drop_mask[0, self.anchor_idx] = 1.0
            Z_in = Z * drop_mask
            if need_cache:
                cache["drop_mask"] = drop_mask
        else:
            Z_in = Z
            if need_cache:
                cache["drop_mask"] = None

        # 1. SENet dynamic channel gate (`u / sqrt(1 + u^2)`)
        if self.gate_hidden > 0 and self.gate_scale > 0.0 and "W_g1" in params:
            u_g1 = Z_in @ params["W_g1"] + params["b_g1"][None, :]
            s_g1, inv_r_g1 = _alg_tanh_with_inv_r(u_g1)
            u_g2 = s_g1 @ params["W_g2"] + params["b_g2"][None, :]
            t_g2, inv_r_g2 = _alg_tanh_with_inv_r(u_g2)
            g = np.float32(1.0) + np.float32(self.gate_scale) * t_g2
            Z_tilde = Z_in * g
            if need_cache:
                cache["Z_in"] = Z_in
                cache["s_g1"] = s_g1
                cache["inv_r_g1"] = inv_r_g1
                cache["inv_r_g2"] = inv_r_g2
        else:
            Z_tilde = Z_in
            if need_cache:
                cache["Z_in"] = Z_in

        if need_cache:
            cache["Z_tilde"] = Z_tilde

        # 2. Exact Rendle 2nd-Order Factorization Machine (`FM2`)
        if self.fm_factors > 0 and "V_fm" in params:
            V_fm = params["V_fm"]
            inv_k = np.float32(1.0 / math.sqrt(float(self.fm_factors)))
            Q_fm = Z_tilde @ V_fm
            Z_sq = Z_tilde * Z_tilde
            V_sq = V_fm * V_fm
            y_fm2 = np.float32(0.5) * inv_k * (np.sum(Q_fm * Q_fm, axis=1) - (Z_sq @ np.sum(V_sq, axis=1)))
            if need_cache:
                cache["Q_fm"] = Q_fm
                cache["Z_sq"] = Z_sq
                cache["V_sq"] = V_sq
                cache["inv_k"] = inv_k
        else:
            y_fm2 = np.zeros(N, dtype=np.float32)

        # 3. Low-Rank DCN-v2 Cross Network (`u / sqrt(1 + u^2)`)
        h = Z_tilde
        h_list: list[np.ndarray] = []
        a_list: list[np.ndarray] = []
        t_list: list[np.ndarray] = []
        r_list: list[np.ndarray] = []
        for l_idx in range(self.cross_layers):
            V_c = params[f"V_c{l_idx}"]
            U_c = params[f"U_c{l_idx}"]
            b_c = params[f"b_c{l_idx}"]
            a_l = h @ V_c
            t_l, inv_r_l = _alg_tanh_with_inv_r(a_l @ U_c + b_c[None, :])
            if need_cache:
                h_list.append(h)
                a_list.append(a_l)
                t_list.append(t_l)
                r_list.append(inv_r_l)
            h = h + Z_tilde * t_l

        if need_cache:
            cache["h_list"] = h_list
            cache["a_list"] = a_list
            cache["t_list"] = t_list
            cache["r_list"] = r_list
            cache["h_final"] = h

        y_dcn = h @ params["w_lin"]
        out = y_dcn + y_fm2

        if "theta_skip" in params and self.anchor_idx is not None and 0 <= self.anchor_idx < P:
            alpha_skip = np.float32(_softplus(float(params["theta_skip"][0])) + self.min_anchor_weight)
            out = out + alpha_skip * Z[:, self.anchor_idx]

        return out.astype(np.float32, copy=False), cache

    def _backward_from_cache(
        self,
        dy: np.ndarray,
        cache: dict[str, Any],
        params: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], float]:
        """Exact analytic gradients `dL/dparams` + L2 regularization loss given `dy = dL/dy_hat (N,)`."""
        dy32 = np.asarray(dy, dtype=np.float32)
        dy_col = dy32[:, None]
        Z = cache["Z"]
        Z_in = cache["Z_in"]
        Z_tilde = cache["Z_tilde"]
        grads: dict[str, np.ndarray] = {}
        reg_loss = 0.0

        # Skip anchor gradient
        if "theta_skip" in params and self.anchor_idx is not None:
            sig_th = _sigmoid(float(params["theta_skip"][0]))
            g_skip = float(np.dot(dy32.astype(np.float64), Z[:, self.anchor_idx].astype(np.float64))) * sig_th
            grads["theta_skip"] = np.array([g_skip], dtype=np.float32)

        # Linear readout gradient
        h_final = cache["h_final"]
        w_lin = params["w_lin"]
        grads["w_lin"] = (h_final.T @ dy32) + np.float32(self.lin_l2) * w_lin
        reg_loss += 0.5 * self.lin_l2 * float(np.sum(w_lin.astype(np.float64) ** 2))

        dh = dy_col * w_lin[None, :]
        dZ_tilde = np.zeros_like(Z_tilde)

        # DCN-v2 cross layers backward (l = L-1 .. 0)
        h_list = cache["h_list"]
        a_list = cache["a_list"]
        t_list = cache["t_list"]
        r_list = cache["r_list"]
        for l_idx in range(self.cross_layers - 1, -1, -1):
            V_c = params[f"V_c{l_idx}"]
            U_c = params[f"U_c{l_idx}"]
            h_l = h_list[l_idx]
            a_l = a_list[l_idx]
            t_l = t_list[l_idx]
            inv_r_l = r_list[l_idx]

            dZ_tilde += dh * t_l
            du_l = dh * Z_tilde * (inv_r_l * inv_r_l * inv_r_l)
            grads[f"b_c{l_idx}"] = du_l.sum(axis=0)
            grads[f"U_c{l_idx}"] = (a_l.T @ du_l) + np.float32(self.dcn_l2) * U_c
            da_l = du_l @ U_c.T
            grads[f"V_c{l_idx}"] = (h_l.T @ da_l) + np.float32(self.dcn_l2) * V_c
            dh = dh + (da_l @ V_c.T)
            reg_loss += 0.5 * self.dcn_l2 * float(
                np.sum(U_c.astype(np.float64) ** 2) + np.sum(V_c.astype(np.float64) ** 2)
            )

        dZ_tilde += dh

        # Rendle 2nd-Order FM backward
        if self.fm_factors > 0 and "V_fm" in params:
            V_fm = params["V_fm"]
            Q_fm = cache["Q_fm"]
            Z_sq = cache["Z_sq"]
            V_sq = cache["V_sq"]
            inv_k = cache["inv_k"]
            g_fm_col = inv_k * dy_col
            gQ = g_fm_col * Q_fm  # (N, K)
            sum_z2 = (Z_sq.T @ (inv_k * dy32))[:, None]  # (P, 1)
            grads["V_fm"] = (Z_tilde.T @ gQ - sum_z2 * V_fm) + np.float32(self.fm_l2) * V_fm
            v_sq_row = np.sum(V_sq, axis=1)[None, :]  # (1, P)
            dZ_tilde += (gQ @ V_fm.T) - g_fm_col * (Z_tilde * v_sq_row)
            reg_loss += 0.5 * self.fm_l2 * float(np.sum(V_fm.astype(np.float64) ** 2))

        # SENet dynamic gate backward
        if self.gate_hidden > 0 and self.gate_scale > 0.0 and "W_g1" in params:
            W_g1 = params["W_g1"]
            W_g2 = params["W_g2"]
            s_g1 = cache["s_g1"]
            inv_r_g1 = cache["inv_r_g1"]
            inv_r_g2 = cache["inv_r_g2"]
            dg = dZ_tilde * Z_in
            du_g2 = np.float32(self.gate_scale) * dg * (inv_r_g2 * inv_r_g2 * inv_r_g2)
            grads["b_g2"] = du_g2.sum(axis=0)
            grads["W_g2"] = (s_g1.T @ du_g2) + np.float32(self.dcn_l2) * W_g2
            ds_g1 = du_g2 @ W_g2.T
            du_g1 = ds_g1 * (inv_r_g1 * inv_r_g1 * inv_r_g1)
            grads["b_g1"] = du_g1.sum(axis=0)
            grads["W_g1"] = (Z_in.T @ du_g1) + np.float32(self.dcn_l2) * W_g1
            reg_loss += 0.5 * self.dcn_l2 * float(
                np.sum(W_g1.astype(np.float64) ** 2) + np.sum(W_g2.astype(np.float64) ** 2)
            )

        return grads, reg_loss

    def compute_ranking_loss_and_grad_scores(
        self,
        scores: np.ndarray,
        gains: np.ndarray,
        valid_mask: np.ndarray,
        idcg10: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute hybrid ListNet + Burges (2010) LambdaRank |Delta NDCG| loss and `dL/dscores (B, D)`."""
        B, D = scores.shape
        inv_B = 1.0 / float(max(B, 1))
        grad_s = np.zeros((B, D), dtype=np.float64)
        total_loss = 0.0

        s64 = np.where(valid_mask, scores.astype(np.float64), -1e9)
        g64 = np.where(valid_mask, np.maximum(gains.astype(np.float64), 0.0), 0.0)
        g_sum = g64.sum(axis=1, keepdims=True)
        has_pos = (g_sum.ravel() > 1e-9)
        if not np.any(has_pos):
            return 0.0, grad_s.astype(np.float32)

        # 1. Temperature-1.0 Listwise Plackett-Luce / ListNet term
        if self.listwise_weight > 0.0:
            s_shift = s64 - s64.max(axis=1, keepdims=True)
            exp_s = np.where(valid_mask, np.exp(np.clip(s_shift, -50.0, 0.0)), 0.0)
            probs = exp_s / np.maximum(exp_s.sum(axis=1, keepdims=True), 1e-12)
            q_dist = np.where(g_sum > 1e-9, g64 / np.maximum(g_sum, 1e-12), 0.0)
            u_wt = np.where(
                has_pos,
                0.5 + 0.5 * np.minimum(g_sum.ravel() / np.maximum(idcg10, 1e-6), 2.0),
                0.0,
            )[:, None]
            lw_scale = self.listwise_weight * inv_B
            total_loss += -lw_scale * float(np.sum(u_wt * q_dist * np.log(np.maximum(probs, 1e-15))))
            grad_s += lw_scale * u_wt * (probs - q_dist)

        # 2. Vectorized Burges (2010) Pairwise LambdaRank |Delta NDCG| over positive (u, p) pairs
        if self.lambdarank_weight > 0.0:
            order = np.argsort(-s64, axis=1, kind="stable")
            ranks = np.empty_like(order)
            rows_b = np.arange(B)[:, None]
            ranks[rows_b, order] = np.arange(D)[None, :]

            trunc = self.lambdarank_truncation
            r_idx = np.arange(D, dtype=np.float64)
            base_disc = 1.0 / np.log2(r_idx + 2.0)
            disc_lut = np.where(
                r_idx < 12,
                1.20 * base_disc,
                np.where(r_idx < trunc, 0.85 * base_disc, 0.15 * base_disc),
            )
            item_disc = disc_lut[ranks]  # (B, D)

            pos_u, pos_i = np.nonzero(g64 > 1e-6)
            if len(pos_u) > 0:
                s_p = s64[pos_u, pos_i][:, None]          # (K_pos, 1)
                g_p = g64[pos_u, pos_i][:, None]          # (K_pos, 1)
                d_p = item_disc[pos_u, pos_i][:, None]    # (K_pos, 1)
                idcg_p = np.maximum(idcg10[pos_u], 1e-6)[:, None]

                s_all = s64[pos_u]                        # (K_pos, D)
                g_all = g64[pos_u]                        # (K_pos, D)
                d_all = item_disc[pos_u]                  # (K_pos, D)
                v_all = valid_mask[pos_u]                 # (K_pos, D)

                pair_mask = v_all & (g_p > g_all + 1e-6)
                # Standard DCG gain difference (2^g - 1 style or linear in log-gain g; harness DCG uses g = log2(min+1))
                delta_ndcg = np.where(pair_mask, (g_p - g_all) * np.abs(d_p - d_all) / idcg_p, 0.0)

                sig = self.lambdarank_sigma
                margin = np.clip(sig * (s_p - s_all), -35.0, 35.0)
                # log(1 + exp(-margin))
                softp = np.where(margin > 0.0, np.log1p(np.exp(-margin)), -margin + np.log1p(np.exp(margin)))
                lr_scale = self.lambdarank_weight * inv_B
                total_loss += lr_scale * float(np.sum(delta_ndcg * softp))

                # d/d(s_p) of log(1 + exp(-sig*(s_p - s_j))) = -sig / (1 + exp(margin))
                neg_lam = -lr_scale * sig * delta_ndcg / (1.0 + np.exp(margin))  # (K_pos, D) <= 0
                pos_grad_sum = neg_lam.sum(axis=1)                                # (K_pos,)
                np.add.at(grad_s, (pos_u, pos_i), pos_grad_sum)
                # Scatter -neg_lam to (pos_u, 0..D-1)
                np.add.at(grad_s, pos_u, -neg_lam)

        return total_loss, grad_s.astype(np.float32)

    def fit(
        self,
        X_train: np.ndarray,
        gains_train: np.ndarray,
        valid_mask_train: np.ndarray,
        idcg10_train: np.ndarray,
        X_val: np.ndarray | None = None,
        gains_val: np.ndarray | None = None,
        valid_mask_val: np.ndarray | None = None,
        idcg10_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        verbose: bool = False,
        overwrite_x: bool = False,
    ) -> dict[str, Any]:
        """Fit `FactorizedCrossNet` on `(U, D, P)` candidate tensors with early stopping on exact `NDCG@10`."""
        t0 = time.time()
        U_tr, D_tr, P = X_train.shape
        self.n_features_ = int(P)
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"f_{j}" for j in range(P)]

        # Streaming chunked standardization stats & binary indicator mask (zero multi-GB copies!)
        sum64 = np.zeros(P, dtype=np.float64)
        sumsq64 = np.zeros(P, dtype=np.float64)
        total_cnt = 0
        is_bin = np.ones(P, dtype=bool)
        step_st = 4096
        for s in range(0, U_tr, step_st):
            e = min(s + step_st, U_tr)
            xb = X_train[s:e].reshape(-1, P)
            vb = valid_mask_train[s:e].reshape(-1)
            sub_x = xb[vb] if np.any(vb) else xb
            if len(sub_x) == 0:
                continue
            fin_b = np.where(np.isfinite(sub_x), sub_x, np.float32(0.0))
            if s == 0:
                chk = fin_b[: min(len(fin_b), 131072)]
                is_bin &= np.all((np.abs(chk) < 1e-5) | (np.abs(chk - 1.0) < 1e-5), axis=0)
            f64 = fin_b.astype(np.float64)
            sum64 += f64.sum(axis=0)
            sumsq64 += np.square(f64).sum(axis=0)
            total_cnt += len(fin_b)

        denom_cnt = float(max(total_cnt, 1))
        mean64 = sum64 / denom_cnt
        var64 = np.maximum(sumsq64 / denom_cnt - np.square(mean64), 0.0)
        std64 = np.sqrt(var64)
        self.mean_ = mean64.astype(np.float32)
        self.inv_std_ = (1.0 / np.maximum(std64, 1e-4)).astype(np.float32)
        self.is_binary_ = is_bin

        if overwrite_x and X_train.dtype == np.float32:
            for s in range(0, U_tr, step_st):
                e = min(s + step_st, U_tr)
                X_train[s:e] = self.standardize_features(X_train[s:e])
            Z_tr = X_train
        else:
            Z_tr = self.standardize_features(X_train)
        Z_va = self.standardize_features(X_val) if X_val is not None else None

        disc10 = (1.0 / np.log2(np.arange(min(10, D_tr), dtype=np.float64) + 2.0))[None, :]

        def _predict_z_split(Z_split: np.ndarray, p_list: list[dict[str, np.ndarray]]) -> np.ndarray:
            U_s, D_s, _ = Z_split.shape
            preds = np.zeros((U_s, D_s), dtype=np.float32)
            inv_m = np.float32(1.0 / float(len(p_list)))
            step_u = 4096
            for s in range(0, U_s, step_u):
                e = min(s + step_u, U_s)
                z_flat = Z_split[s:e].reshape((e - s) * D_s, P)
                acc_y = np.zeros((e - s) * D_s, dtype=np.float32)
                for p_eval in p_list:
                    y_b, _ = self._forward_with_cache(z_flat, p_eval, training=False, need_cache=False)
                    acc_y += inv_m * y_b
                preds[s:e] = acc_y.reshape(e - s, D_s)
            return preds

        def _eval_ndcg10(
            Z_split: np.ndarray,
            g_split: np.ndarray,
            vm_split: np.ndarray,
            idcg_split: np.ndarray,
            p_list: list[dict[str, np.ndarray]],
        ) -> float:
            U_s, D_s, _ = Z_split.shape
            k10 = min(10, D_s)
            preds = np.where(vm_split, _predict_z_split(Z_split, p_list), -np.inf)
            top10 = np.argpartition(-preds, k10 - 1, axis=1)[:, :k10] if k10 < D_s else np.broadcast_to(np.arange(D_s), (U_s, D_s)).copy()
            sub_p = np.take_along_axis(preds, top10, axis=1)
            ord10 = np.argsort(-sub_p, axis=1, kind="stable")
            top10 = np.take_along_axis(top10, ord10, axis=1)
            got_g = np.take_along_axis(g_split, top10, axis=1).astype(np.float64)
            return float(np.mean(np.sum(got_g * disc10[:, :k10], axis=1) / np.maximum(idcg_split, 1e-12)))

        n_batches = max(1, (U_tr + self.batch_users - 1) // self.batch_users)
        total_steps = max(1, self.epochs * n_batches)
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

        trained_members: list[dict[str, np.ndarray]] = []
        best_epochs: list[int] = []
        init_val_ndcg = -1.0

        for m_idx in range(self.n_ensemble):
            rng = np.random.default_rng(self.seed + m_idx * 1009)
            params = self._init_params(P, rng)
            m_adam = {k: np.zeros_like(v, dtype=np.float32) for k, v in params.items()}
            v_adam = {k: np.zeros_like(v, dtype=np.float32) for k, v in params.items()}

            cur_init_ndcg = (
                _eval_ndcg10(Z_va, gains_val, valid_mask_val, idcg10_val, [params])
                if (Z_va is not None and gains_val is not None and valid_mask_val is not None and idcg10_val is not None)
                else -1.0
            )
            if m_idx == 0:
                init_val_ndcg = cur_init_ndcg
            best_m_ndcg = cur_init_ndcg
            best_m_params = {k: v.copy() for k, v in params.items()}
            best_m_epoch = 0
            no_improve = 0
            global_step = 0

            for epoch in range(1, self.epochs + 1):
                perm = rng.permutation(U_tr)
                ep_loss = 0.0
                for b_start in range(0, U_tr, self.batch_users):
                    b_end = min(b_start + self.batch_users, U_tr)
                    b_idx = perm[b_start:b_end]
                    B_cur = len(b_idx)
                    global_step += 1

                    # Cosine annealing learning rate with 1-epoch linear warmup
                    warmup_steps = n_batches
                    if global_step <= warmup_steps:
                        lr_t = self.learning_rate * (float(global_step) / float(max(warmup_steps, 1)))
                    else:
                        prog = float(global_step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
                        lr_t = self.learning_rate * (0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0))))

                    Z_b = Z_tr[b_idx].reshape(B_cur * D_tr, P)
                    y_b, cache = self._forward_with_cache(Z_b, params, rng=rng, training=True)
                    rk_loss, grad_scores = self.compute_ranking_loss_and_grad_scores(
                        y_b.reshape(B_cur, D_tr),
                        gains_train[b_idx],
                        valid_mask_train[b_idx],
                        idcg10_train[b_idx],
                    )
                    grads, reg_loss = self._backward_from_cache(grad_scores.ravel(), cache, params)
                    ep_loss += (rk_loss + reg_loss) * (B_cur / float(U_tr))

                    bc1 = 1.0 - (beta1 ** global_step)
                    bc2 = 1.0 - (beta2 ** global_step)
                    step_alpha = np.float32(lr_t / bc1)
                    inv_bc2 = np.float32(1.0 / bc2)

                    for k, p_arr in params.items():
                        g_arr = np.clip(grads[k], -5.0, 5.0)
                        m_k = m_adam[k]
                        v_k = v_adam[k]
                        m_k *= np.float32(beta1)
                        m_k += np.float32(1.0 - beta1) * g_arr
                        v_k *= np.float32(beta2)
                        v_k += np.float32(1.0 - beta2) * (g_arr * g_arr)
                        denom = np.sqrt(v_k * inv_bc2) + np.float32(eps_adam)
                        if self.weight_decay > 0.0 and not k.startswith("b_") and k != "theta_skip":
                            p_arr *= np.float32(1.0 - lr_t * self.weight_decay)
                        p_arr -= step_alpha * (m_k / denom)

                if Z_va is not None and gains_val is not None and valid_mask_val is not None and idcg10_val is not None:
                    val_ndcg = _eval_ndcg10(Z_va, gains_val, valid_mask_val, idcg10_val, [params])
                    if verbose:
                        print(
                            f"[FactorizedCrossNet m={m_idx}] epoch={epoch:02d} loss={ep_loss:.5f} val_ndcg@10={val_ndcg:.5f}",
                            flush=True,
                        )
                    if val_ndcg > best_m_ndcg + 1e-6:
                        best_m_ndcg = val_ndcg
                        best_m_params = {k: v.copy() for k, v in params.items()}
                        best_m_epoch = epoch
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= self.patience:
                            break
                else:
                    best_m_params = {k: v.copy() for k, v in params.items()}
                    best_m_epoch = epoch

            trained_members.append(best_m_params)
            best_epochs.append(best_m_epoch)

        self.ensemble_params_ = trained_members
        self.params_ = trained_members[0]
        self.is_fitted = True
        best_epoch = max(best_epochs) if best_epochs else 0
        best_val_ndcg = (
            _eval_ndcg10(Z_va, gains_val, valid_mask_val, idcg10_val, self.ensemble_params_)
            if (Z_va is not None and gains_val is not None and valid_mask_val is not None and idcg10_val is not None)
            else init_val_ndcg
        )
        fit_sec = time.time() - t0

        # Effective linear + cross channel attribution averaged over ensemble members for diagnostics
        w_lin = np.mean([m_p["w_lin"] for m_p in self.ensemble_params_], axis=0).astype(np.float32)
        avg_skip = float(
            np.mean(
                [
                    _softplus(float(m_p["theta_skip"][0])) + self.min_anchor_weight
                    for m_p in self.ensemble_params_
                    if "theta_skip" in m_p
                ]
                or [0.0]
            )
        )
        if self.anchor_idx is not None and any("theta_skip" in m_p for m_p in self.ensemble_params_):
            w_lin[self.anchor_idx] += np.float32(avg_skip)
        fm_norm = (
            np.mean([np.linalg.norm(m_p["V_fm"], axis=1) for m_p in self.ensemble_params_], axis=0).astype(np.float32)
            if "V_fm" in self.params_
            else np.zeros(P, dtype=np.float32)
        )
        imp = np.abs(w_lin) + np.float32(0.5) * fm_norm
        top_idx = np.argsort(-imp)[:15]
        top_features = {self.feature_names_[int(i)]: round(float(imp[int(i)]), 5) for i in top_idx}

        return {
            "n_features": P,
            "fm_factors": self.fm_factors,
            "cross_layers": self.cross_layers,
            "cross_rank": self.cross_rank,
            "gate_hidden": self.gate_hidden,
            "n_ensemble": len(self.ensemble_params_),
            "best_epoch": int(best_epoch),
            "member_best_epochs": [int(e) for e in best_epochs],
            "init_valid_ndcg10": round(float(init_val_ndcg), 6),
            "best_valid_ndcg10": round(float(best_val_ndcg), 6),
            "anchor_skip_weight": round(avg_skip, 5),
            "fit_seconds": round(fit_sec, 2),
            "top_feature_importances": top_features,
        }

    def predict_scores(
        self,
        X: np.ndarray,
        batch_rows: int = 65536,
        mean_override: np.ndarray | None = None,
        inv_std_override: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict `(...,)` scores in `float32` for candidate feature matrix `X (..., P)`."""
        assert self.is_fitted and self.mean_ is not None and self.inv_std_ is not None
        members = self.ensemble_params_ if self.ensemble_params_ else [self.params_]
        inv_m = np.float32(1.0 / float(len(members)))
        orig_shape = X.shape[:-1]
        P = X.shape[-1]
        X_flat = np.asarray(X, dtype=np.float32).reshape(-1, P)
        N = X_flat.shape[0]
        out = np.empty(N, dtype=np.float32)
        for s in range(0, N, batch_rows):
            e = min(s + batch_rows, N)
            Z_b = self.standardize_features(
                X_flat[s:e],
                mean_override=mean_override,
                inv_std_override=inv_std_override,
            )
            if len(members) == 1:
                out[s:e], _ = self._forward_with_cache(Z_b, members[0], training=False, need_cache=False)
            else:
                acc = np.zeros(e - s, dtype=np.float32)
                for m_p in members:
                    y_m, _ = self._forward_with_cache(Z_b, m_p, training=False, need_cache=False)
                    acc += inv_m * y_m
                out[s:e] = acc
        return out.reshape(orig_shape)

    def to_dict(self) -> dict[str, Any]:
        """Serialize `FactorizedCrossNet` hyperparameters, scalers, and learned factor matrices."""
        assert self.is_fitted and self.mean_ is not None and self.inv_std_ is not None
        members = self.ensemble_params_ if self.ensemble_params_ else [self.params_]
        return {
            "config": {
                "fm_factors": self.fm_factors,
                "cross_layers": self.cross_layers,
                "cross_rank": self.cross_rank,
                "gate_hidden": self.gate_hidden,
                "gate_scale": self.gate_scale,
                "sat_cap": self.sat_cap,
                "anchor_idx": self.anchor_idx,
                "min_anchor_weight": self.min_anchor_weight,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "fm_l2": self.fm_l2,
                "dcn_l2": self.dcn_l2,
                "lin_l2": self.lin_l2,
                "listwise_weight": self.listwise_weight,
                "lambdarank_weight": self.lambdarank_weight,
                "lambdarank_sigma": self.lambdarank_sigma,
                "lambdarank_truncation": self.lambdarank_truncation,
                "batch_users": self.batch_users,
                "epochs": self.epochs,
                "patience": self.patience,
                "feature_dropout": self.feature_dropout,
                "n_ensemble": len(members),
                "seed": self.seed,
            },
            "n_features": self.n_features_,
            "feature_names": list(self.feature_names_),
            "mean": self.mean_.tolist(),
            "inv_std": self.inv_std_.tolist(),
            "is_binary": self.is_binary_.tolist() if self.is_binary_ is not None else [False] * self.n_features_,
            "params": {k: v.tolist() for k, v in self.params_.items()},
            "ensemble_params": [{k: v.tolist() for k, v in m_p.items()} for m_p in members],
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "FactorizedCrossNet":
        obj = cls(**state["config"])
        obj.n_features_ = int(state["n_features"])
        obj.feature_names_ = list(state["feature_names"])
        obj.mean_ = np.asarray(state["mean"], dtype=np.float32)
        obj.inv_std_ = np.asarray(state["inv_std"], dtype=np.float32)
        obj.is_binary_ = (
            np.asarray(state["is_binary"], dtype=bool)
            if "is_binary" in state
            else np.zeros(obj.n_features_, dtype=bool)
        )
        obj.params_ = {k: np.asarray(v, dtype=np.float32) for k, v in state["params"].items()}
        if "ensemble_params" in state and state["ensemble_params"]:
            obj.ensemble_params_ = [
                {k: np.asarray(v, dtype=np.float32) for k, v in m_d.items()} for m_d in state["ensemble_params"]
            ]
        else:
            obj.ensemble_params_ = [obj.params_]
        obj.is_fitted = True
        return obj
