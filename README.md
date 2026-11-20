# RankFM

[![CI](https://github.com/sergeyfogelson/rankfm/actions/workflows/ci.yml/badge.svg)](https://github.com/sergeyfogelson/rankfm/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **This repository is a fork of [etlundquist/rankfm](https://github.com/etlundquist/rankfm), created and written by Eric Lundquist.**
> The model, the original Cython implementation, the evaluation module, the examples and the documentation are his work, released under the GNU GPL v3. This fork is maintained by Sergey Fogelson and has been modified since 17 November 2026; see [Attribution and Changes from the Original](#attribution-and-changes-from-the-original). The `rankfm` package on PyPI and the Read the Docs site are the original project's, not this fork's.

RankFM is a python implementation of the general [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) model class adapted for collaborative filtering recommendation/ranking problems with implicit feedback user/item interaction data. It uses [Bayesian Personalized Ranking (BPR)](https://arxiv.org/pdf/1205.2618.pdf) and a variant of [Weighted Approximate-Rank Pairwise (WARP)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.587.3946&rep=rep1&type=pdf) loss to learn model weights via Stochastic Gradient Descent (SGD). It can (optionally) incorporate sample weights and user/item auxiliary features to augment the main interaction data.

The core (training, prediction, recommendation) methods are written in [Cython](https://cython.org/), making it possible to scale to millions of user/item interactions. Designed for ease-of-use, RankFM accepts both `pd.DataFrame` and `np.ndarray` inputs - you do not have to convert your data to `scipy.sparse` matrices or re-map user/item identifiers prior to use. RankFM internally maps all user/item identifiers to zero-based integer indexes, but always converts its output back to the original user/item identifiers from your data, which can be arbitrary (non-zero-based, non-consecutive) integers or even strings.

In addition to the familiar `fit()`, `predict()`, `recommend()` methods, RankFM includes additional utilities `similiar_users()` and `similar_items()` to find the most similar users/items to a given user/item based on latent factor space embeddings. A number of popular recommendation/ranking evaluation metric functions have been included in the separate `evaluation` module to streamline model tuning and validation.

* see the **Quickstart** section below to get started with the basic functionality
* see the `/examples` folder for more in-depth jupyter notebook walkthroughs with several popular open-source data sets
* see the original project's [Online Documentation](https://rankfm.readthedocs.io/en/latest/) for the main model class and separate evaluation module (it describes version 0.2.5 and does not cover the additions in this fork)
* see Eric Lundquist's [Medium Article](https://towardsdatascience.com/factorization-machines-for-item-recommendation-with-implicit-feedback-data-5655a7c749db) for contextual motivation and a detailed mathematical description of the algorithm

---
### Attribution and Changes from the Original

RankFM was created by **Eric Lundquist**. The original project lives at [github.com/etlundquist/rankfm](https://github.com/etlundquist/rankfm), is published on PyPI as [`rankfm`](https://pypi.org/project/rankfm/), and its last release is 0.2.5 (July 2020). Its complete commit history is preserved in this repository, and every file here remains under the GNU General Public License v3 (see `LICENSE`).

This fork starts from the original's commit `c3e9395` and has been modified by Sergey Fogelson since 17 November 2026. The modifications, by release:

* **0.4.0** (November 2026): multi-threaded training, sparse (CSR) user and item features, inference for unseen users (cold-start estimation and fold-in), batched and streaming recommendation, export of embeddings for nearest-neighbour search
* **0.4.1**: reproducible seeding, container-aware thread counts, step normalization for side features, multi-threaded `feature_reg='uniform'`, faster `fit_partial`, input validation that survives `python -O`
* **0.4.2**: corrected regularization and gradient computation when training with side features, a Dockerfile, and continuous integration on Linux, Windows and macOS
* **0.4.5**: no fixed limit on interactions per fit, about 8 bytes of memory per interaction instead of 20, and optional batched updates for densely used feature columns (`dense_updates='batched'`)

The public API of the original (`RankFM.fit`, `fit_partial`, `predict`, `recommend`, `similar_items`, `similar_users`, and the `rankfm.evaluation` functions) is kept, with new optional arguments. Models trained with this fork are not numerically identical to models trained with the original.

---
### Dependencies
* Python 3.10+
* Cython >= 3.0 (build time only)
* threadpoolctl (optional - keeps BLAS threads within the CPUs available to the process)
* numpy >= 1.15
* pandas >= 0.24

### Installation

#### Prerequisites

To install RankFM's C extensions you will need the [GNU Compiler Collection (GCC)](https://gcc.gnu.org/). Check to see whether you already have it installed:
```
gcc --version
```

If you don't have it already you can easily install it using [Homebrew](https://brew.sh/) on OSX or your default linux package manager:
```
# OSX
brew install gcc

# linux
sudo yum install gcc

# ensure [gcc] has been installed correctly and is on the system PATH
gcc --version
```

#### Package Installation

Install this fork directly from GitHub:
```
pip install git+https://github.com/sergeyfogelson/rankfm.git#egg=rankfm
```
`pip install rankfm` installs the original upstream package from PyPI, which does not include the multi-threaded training, sparse feature, unseen-user, or streaming inference work in this repository.

#### Performance and Deployment Notes

* **Threads in containers**: `n_jobs=-1` uses the CPUs actually available to the process (CPU affinity and cgroup quota), not the host core count, so it is safe under a Kubernetes / Cloud Run CPU limit. With `threadpoolctl` installed, the BLAS call inside `recommend()` is held to the same limit; without it, set `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` yourself.
* **Instruction set**: build with `RANKFM_MARCH=x86-64-v3 pip install .` to enable AVX2 + FMA on any x86 CPU from roughly 2015 onward. Avoid `RANKFM_ARCH_NATIVE=1` when building an image or wheel on one machine to run on another: it targets the build machine's CPU and fails with an illegal-instruction error elsewhere.
* **Reproducibility**: `fit(seed=...)` fixes both weight initialization and sampling. Results are bit-identical with `n_jobs=1`; multi-threaded training is lock-free, so repeated runs differ slightly.
* **Dense feature columns**: a feature that is non-zero for a large share of users (or items) has its weights written by every thread on nearly every step, so with the default `dense_updates='lockfree'` training stops getting faster beyond about 4-8 threads (model quality is unaffected). `dense_updates='batched'` has each thread collect the updates for those columns and apply them once every 64 steps, scaled down when `learning_rate * threads * 64` exceeds a stability limit. On a dataset with 8 dense user features that took 16-thread training from 11M to 35M samples/s with the same accuracy. The limit was calibrated on synthetic data with up to 16 threads, so compare both settings on your own data before relying on it. Not available with `feature_reg='uniform'`.
* **Learning rate with side features**: every active feature on a row adds to the effective step size, so raw SGD steps diverge once rows carry more than a few features. `normalize_steps=True` (the default) scales each update, gradient and L2 decay together, by the feature mass of the rows it touches. Training with user and/or item features is then stable over at least the same range of `learning_rate` as a model without features. It is not a guarantee, and training warns when weights grow past 1e3. Pass `normalize_steps=False` for unscaled updates.
* **Item features and speed**: candidate negatives are scored from a cache of effective item embeddings, but the positive item and the chosen negative are recomputed from the current weights before each gradient step. A gradient taken from the cache alone keeps pushing pairs that are already ranked correctly. The recomputation costs roughly 30-45% of training throughput at 20 active features per item, and nothing without item features.
* **Feature regularization**: `feature_reg='uniform'` works with any number of threads.
* **Incremental updates**: `fit_partial()` places new interactions by binary search and reuses spare capacity when users are added, so a small batch costs one block copy of the stored history and one scan of the weights, not a re-sort. Sequential integer user IDs keep constant-time lookups; with arbitrary IDs the ID index is rebuilt whenever a batch adds users.
* **Input checks**: NaN or infinite feature values and sample weights are rejected before training starts. Sparse feature matrices are copied, never sorted in place or referenced afterwards.
* **Containers and CI**: the `Dockerfile` builds a Linux image (`docker build --build-arg RANKFM_MARCH=x86-64-v3 -t rankfm .`), and `ci/container_check.py` is an end-to-end check meant to run under a CPU limit (`docker run --rm --cpus=2 -e EXPECT_CPUS=2 rankfm`). GitHub Actions runs the tests on Linux (Python 3.10-3.13), Windows and macOS, and the container check on x86 Linux.
* **Scale and memory**: there is no fixed limit on the number of interactions per `fit()` call; each epoch visits them in a shuffled order that is computed, not stored. The model holds 8 bytes per interaction when the input has no repeated (user, item) pairs and either carries no weights or is already sorted by user then item, plus 4 bytes with weights. Other inputs (repeated pairs, or unsorted pairs with weights) take about 12-16 bytes. Repeated pairs are kept as separate training rows, as before.
* **Aggregating event logs**: for large event data, train on one weighted row per (user, item) rather than one row per event. A BigQuery sketch (always filter the partition column):

  ```sql
  SELECT user_id, item_id, LN(1 + COUNT(*)) AS weight
  FROM `my-project.my_dataset.events`
  WHERE event_date BETWEEN '2026-01-01' AND '2026-03-31'
  GROUP BY user_id, item_id
  ```

  Pass `weight` as `sample_weight`. Damp raw counts or watch time with a log as above: a weight multiplies the step size, so a weight of 500 behaves like a 500 times larger learning rate for that row.

It's highly recommended that you use an [Anaconda](https://www.anaconda.com/) base environment to ensure that all core numpy C extensions and linear algebra libraries have been installed and configured correctly. Anaconda: it just works.

### Quickstart
Let's work through a simple example of fitting a model, generating recommendations, evaluating performance, and assessing some item-item similarities. The data we'll be using here may already be somewhat familiar: you know it, you love it, it's the [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)!

Let's first look at the required shape of the interaction data:

| user_id | item_id |
|---------|---------|
| 3       | 233     |
| 5       | 377     |
| 8       | 610     |

It has just two columns: a `user_id` and an `item_id` (you can name these fields whatever you want or use a numpy array instead). Notice that there is no `rating` column - this library is for **implicit feedback** data (e.g. watches, page views, purchases, clicks) as opposed to **explicit feedback** data (e.g. 1-5 ratings, thumbs up/down). Implicit feedback is far more common in real-world recommendation contexts and doesn't suffer from the [missing-not-at-random problem](https://resources.bibblio.org/hubfs/share/2018-01-24-RecSysLDN-Ravelin.pdf) of pure explicit feedback approaches.

Now let's import the library, initialize our model, and fit on the training data:
```python
from rankfm.rankfm import RankFM
model = RankFM(factors=20, loss='warp', max_samples=20, alpha=0.01, sigma=0.1, learning_rate=0.1, learning_schedule='invscaling')
model.fit(interactions_train, epochs=20, verbose=True)
# NOTE: this takes about 30 seconds for 750,000 interactions on my 2.3 GHz i5 8GB RAM MacBook
```
If you set `verbose=True` the model will print the current epoch number as well as the epoch's log-likelihood during training. This can be useful to gauge both computational speed and training gains by epoch. If the log likelihood is not increasing then try upping the `learning_rate` or lowering the (`alpha`, `beta`) regularization strength terms. If the log likelihood is starting to bounce up and down try lowering the `learning_rate` or using `learning_schedule='invscaling'` to decrease the learning rate over time. If you run into overflow errors then decrease the feature and/or sample-weight magnitudes and try upping `beta`, especially if you have a small number of dense user-features and/or item-features. Selecting `BPR` loss will lead to faster training times, but `WARP` loss typically yields superior model performance.

Now let's generate some user-item model scores from the validation data:
```python
valid_scores = model.predict(interactions_valid, cold_start='nan')
```
this will produce an array of real-valued model scores generated using the Factorization Machines model equation. You can interpret it as a measure of the predicted utility of item (i) for user (u). The `cold_start='nan'` option can be used to set scores to `np.nan` for user/item pairs not found in the training data, or `cold_start='drop'` can be specified to drop those pairs so the results contain no missing values.

Now let's generate our topN recommended movies for each user:
```python
valid_recs = model.recommend(valid_users, n_items=10, filter_previous=True, cold_start='drop')
```
The input should be a `pd.Series`, `np.ndarray` or `list` of `user_id` values. You can use `filter_previous=True` to prevent generating recommendations that include any items observed by the user in the training data, which could be useful depending on your application context. The result will be a `pd.DataFrame` where `user_id` values will be the index and the rows will be each user's top recommended items in descending order (best item is in column 0):

|   |    0|    1|    2|    3|    4|    5|    6|    7|   8|    9|
|---|-----|-----|-----|-----|-----|-----|-----|-----|----|-----|
|3  | 2396| 1265|  357|   34| 2858| 3175|    1| 2028|  17|  356|
|5  |  608| 1617| 1610| 3418|  590|  474|  858|  377| 924| 1036|
|8  |  589| 1036| 2571| 2028| 2000| 1220| 1197|  110| 780| 1954|

Now let's see how the model is performing wrt the included validation metrics evaluated on the hold-out data:
```python
from rankfm.evaluation import hit_rate, reciprocal_rank, discounted_cumulative_gain, precision, recall

valid_hit_rate = hit_rate(model, interactions_valid, k=10)
valid_reciprocal_rank = reciprocal_rank(model, interactions_valid, k=10)
valid_dcg = discounted_cumulative_gain(model, interactions_valid, k=10)
valid_precision = precision(model, interactions_valid, k=10)
valid_recall = recall(model, interactions_valid, k=10)
```
```
hit_rate: 0.796
reciprocal_rank: 0.339
dcg: 0.734
precision: 0.159
recall: 0.077
```
[That's a Bingo!](https://www.youtube.com/watch?v=q5pESPQpXxE)

Now let's find the most similar other movies for a few movies based on their embedding representations in latent factor space:
```python
# Terminator 2: Judgment Day (1991)
model.similar_items(589, n_items=10)
```
```
2571                       Matrix, The (1999)
1527                Fifth Element, The (1997)
2916                      Total Recall (1990)
3527                          Predator (1987)
780             Independence Day (ID4) (1996)
1909    X-Files: Fight the Future, The (1998)
733                          Rock, The (1996)
1376     Star Trek IV: The Voyage Home (1986)
480                      Jurassic Park (1993)
1200                            Aliens (1986)
```
[I hope you like explosions...](https://www.youtube.com/watch?v=uENYMZNzg9w)

```python
# Being John Malkovich (1999)
model.similar_items(2997, n_items=10)
```
```
2599           Election (1999)
3174    Man on the Moon (1999)
2858    American Beauty (1999)
3317        Wonder Boys (2000)
223              Clerks (1994)
3897      Almost Famous (2000)
2395           Rushmore (1998)
2502       Office Space (1999)
2908     Boys Don't Cry (1999)
3481      High Fidelity (2000)
```
[Let's get weird...](https://www.youtube.com/watch?v=lIpev8JXJHQ&t=5s)
