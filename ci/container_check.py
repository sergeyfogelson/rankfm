"""end-to-end check of RankFM inside a container: CPU detection, BLAS capping, training, incremental fit, inference

Run it under a CPU limit and tell it what to expect, for example:
    docker run --rm --cpus=2 -e EXPECT_CPUS=2 rankfm
It exits non-zero if any check fails and prints one RESULT line of JSON at the end.
"""

import json
import os
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

import rankfm.rankfm as rankfm_module
from rankfm.rankfm import RankFM, _available_cpus, _blas_thread_cap

result = {}


def check(condition, message):
    if not condition:
        print("FAILED: " + message)
        print("RESULT " + json.dumps(result))
        sys.exit(1)
    print("ok: " + message)


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


# ---- 1. environment and CPU detection
expected_cpus = int(os.environ['EXPECT_CPUS']) if os.environ.get('EXPECT_CPUS') else None
affinity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
result.update(
    machine=platform.machine(),
    system=platform.system(),
    python=platform.python_version(),
    numpy=np.__version__,
    pandas=pd.__version__,
    host_cpu_count=os.cpu_count(),
    affinity_cpus=affinity,
    cgroup_cpu_max=read('/sys/fs/cgroup/cpu.max'),
    available_cpus=_available_cpus(),
    n_jobs_all=RankFM(n_jobs=-1)._resolve_n_jobs(),
)
print(json.dumps(result, indent=2))
if expected_cpus is not None:
    check(result['available_cpus'] == expected_cpus, "available CPUs == {} (host reports {})".format(expected_cpus, os.cpu_count()))
    check(result['n_jobs_all'] == expected_cpus, "n_jobs=-1 resolves to {}".format(expected_cpus))

# ---- 2. BLAS thread capping
try:
    from threadpoolctl import threadpool_info
    blas_before = [(lib.get('internal_api'), lib.get('num_threads')) for lib in threadpool_info() if lib.get('user_api') == 'blas']
    with _blas_thread_cap():
        blas_capped = [(lib.get('internal_api'), lib.get('num_threads')) for lib in threadpool_info() if lib.get('user_api') == 'blas']
    result.update(blas_threads_default=blas_before, blas_threads_capped=blas_capped)
    print("BLAS threads by default: {}  inside recommend(): {}".format(blas_before, blas_capped))
    if platform.system() == 'Linux':  # macOS wheels use Accelerate, which threadpoolctl cannot control
        check(len(blas_capped) > 0, "a controllable BLAS library is loaded")
    check(all(n <= result['available_cpus'] for _, n in blas_capped), "BLAS is held to the available CPUs inside recommend()")
except ImportError:
    check(False, "threadpoolctl is installed")

# ---- 3. synthetic clustered data with sparse user and item features
rng = np.random.default_rng(0)
n_users, n_items, n_clusters, per_user = 60000, 4000, 20, 15
u_cluster = rng.integers(0, n_clusters, n_users)
i_cluster = rng.integers(0, n_clusters, n_items)
order = np.argsort(i_cluster, kind='stable')
bounds = np.searchsorted(i_cluster[order], np.arange(n_clusters + 1))
users = np.repeat(np.arange(n_users), per_user)
cl = u_cluster[users]
in_cluster = order[bounds[cl] + (rng.random(len(users)) * (bounds[cl + 1] - bounds[cl])).astype(np.int64)]
items = np.where(rng.random(len(users)) < 0.8, in_cluster, rng.integers(0, n_items, len(users)))
interactions = sp.csr_matrix((np.ones(len(users), dtype=np.float32), (users, items)), shape=(n_users, n_items))
interactions.sum_duplicates()
interactions.data[:] = 1.0


def one_hot_features(n_rows, clusters, n_cols, n_active):
    cols = (clusters[:, None] * (n_cols // n_clusters) + rng.integers(0, n_cols // n_clusters, (n_rows, n_active))).ravel()
    mat = sp.csr_matrix((np.ones(n_rows * n_active, dtype=np.float32), (np.repeat(np.arange(n_rows), n_active), cols)), shape=(n_rows, n_cols))
    mat.sum_duplicates()
    mat.data[:] = 1.0
    return mat


user_features = one_hot_features(n_users, u_cluster, 2000, 6)
item_features = one_hot_features(n_items, i_cluster, 2000, 20)
n_samples = interactions.nnz
eval_users = np.arange(0, n_users, 30)


def precision(model, n_jobs):
    recs = model.recommend(eval_users, n_items=10, filter_previous=True, n_jobs=n_jobs).to_numpy()
    return float((i_cluster[recs] == u_cluster[eval_users][:, None]).mean())


kernel_seconds = {}
original_fit = rankfm_module._fit_csr


def timed_fit(*args, **kwargs):
    start = time.perf_counter()
    original_fit(*args, **kwargs)
    kernel_seconds['last'] = time.perf_counter() - start


rankfm_module._fit_csr = timed_fit

# ---- 4. training: quality, stability, and throughput by thread count
EPOCHS = 4
throughput = {}
with warnings.catch_warnings():
    warnings.simplefilter('error')  # the divergence warning must not fire
    for n_jobs in (1, -1):
        model = RankFM(factors=32, loss='warp', max_samples=10, learning_rate=0.1, n_jobs=n_jobs)
        model.fit(interactions, user_features=user_features, item_features=item_features, epochs=EPOCHS, seed=1)
        throughput[n_jobs] = EPOCHS * n_samples / kernel_seconds['last'] / 1e6
        quality = precision(model, n_jobs)
        largest = float(max(np.abs(model.v_u).max(), np.abs(model.v_i).max(), np.abs(model.v_uf).max(), np.abs(model.v_if).max()))
        print("n_jobs={:>2}: {:.2f}M samples/s, in-cluster@10 {:.3f}, largest weight {:.2f}".format(n_jobs, throughput[n_jobs], quality, largest))
        check(quality > 0.9, "n_jobs={} reaches in-cluster precision > 0.9".format(n_jobs))
        check(largest < 50.0, "n_jobs={} weights stay bounded".format(n_jobs))
result.update(samples_per_sec_1_thread_m=round(throughput[1], 2), samples_per_sec_all_threads_m=round(throughput[-1], 2))

# ---- 5. reproducibility single-threaded
small = interactions[:5000]
first = RankFM(factors=8).fit(small, epochs=2, seed=7, n_jobs=1)
second = RankFM(factors=8).fit(small, epochs=2, seed=7, n_jobs=1)
check(np.array_equal(first.v_i, second.v_i), "same seed gives bit-identical single-threaded fits")

# ---- 6. uniform regularization on all threads
with warnings.catch_warnings():
    warnings.simplefilter('error')
    uniform = RankFM(factors=16, loss='warp', learning_rate=0.1, feature_reg='uniform', n_jobs=-1)
    uniform.fit(interactions, user_features=user_features, item_features=item_features, epochs=EPOCHS, seed=2)
check(precision(uniform, -1) > 0.9, "feature_reg='uniform' trains on all available threads")

# ---- 6b. dense feature columns with batched updates on all threads
dense_centroids = rng.normal(0, 1, (n_clusters, 8))
dense_user_features = (dense_centroids[u_cluster] + rng.normal(0, 0.4, (n_users, 8))).astype(np.float32)
dense_user_features /= np.abs(dense_user_features).max()
with warnings.catch_warnings():
    warnings.simplefilter('error')
    for mode in ('lockfree', 'batched'):
        dense_model = RankFM(factors=16, loss='warp', max_samples=10, learning_rate=0.1, beta=0.01, n_jobs=-1, dense_updates=mode)
        dense_model.fit(interactions, user_features=(np.arange(n_users), dense_user_features), epochs=EPOCHS, seed=3)
        rate = EPOCHS * n_samples / kernel_seconds['last'] / 1e6
        quality = precision(dense_model, -1)
        result['dense_{}_samples_per_sec_m'.format(mode)] = round(rate, 2)
        print("dense features, dense_updates={}: {:.2f}M samples/s, in-cluster@10 {:.3f}".format(mode, rate, quality))
        check(quality > 0.9, "dense_updates='{}' reaches in-cluster precision > 0.9".format(mode))

# ---- 7. incremental fit with new users, then batched inference for them
new_users = np.arange(n_users, n_users + 200)
new_cluster = rng.integers(0, n_clusters, len(new_users))
new_rows = np.repeat(new_users, 5)
new_cl = np.repeat(new_cluster, 5)
new_items = order[bounds[new_cl] + (rng.random(len(new_rows)) * (bounds[new_cl + 1] - bounds[new_cl])).astype(np.int64)]
start = time.perf_counter()
model.fit_partial(
    pd.DataFrame({'user_id': new_rows, 'item_id': new_items}),
    user_features=(new_users, one_hot_features(len(new_users), new_cluster, 2000, 6)),
    epochs=5, freeze_items=True,
)
result['fit_partial_ms'] = round((time.perf_counter() - start) * 1e3, 1)
check(isinstance(model._user_index, pd.RangeIndex) and len(model._user_index) == n_users + 200, "fit_partial registered 200 new users")

batches = list(model.recommend_batches(new_users, batch_size=64, n_items=10, filter_previous=True))
recs = pd.concat(batches).to_numpy()
new_quality = float((i_cluster[recs] == new_cluster[:, None]).mean())
print("new users after fit_partial: in-cluster@10 {:.3f} ({} batches, {:.1f} ms)".format(new_quality, len(batches), result['fit_partial_ms']))
check(new_quality > 0.7, "new users get in-cluster recommendations after fit_partial")

print("RESULT " + json.dumps(result))
