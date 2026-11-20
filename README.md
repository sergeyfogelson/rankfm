# RankFM

[![PyPI version](https://badge.fury.io/py/rankfm.svg)](https://badge.fury.io/py/rankfm)
[![CircleCI](https://circleci.com/gh/etlundquist/rankfm.svg?style=shield)](https://circleci.com/gh/etlundquist/rankfm)
[![Documentation Status](https://readthedocs.org/projects/rankfm/badge/?version=latest)](https://rankfm.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

RankFM is a python implementation of the general [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) model class adapted for collaborative filtering recommendation/ranking problems with implicit feedback user/item interaction data. It uses [Bayesian Personalized Ranking (BPR)](https://arxiv.org/pdf/1205.2618.pdf) and a variant of [Weighted Approximate-Rank Pairwise (WARP)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.587.3946&rep=rep1&type=pdf) loss to learn model weights via Stochastic Gradient Descent (SGD). It can (optionally) incorporate sample weights and user/item auxiliary features to augment the main interaction data.

The core (training, prediction, recommendation) methods are written in [Cython](https://cython.org/), making it possible to scale to millions of user/item interactions. Designed for ease-of-use, RankFM accepts both `pd.DataFrame` and `np.ndarray` inputs - you do not have to convert your data to `scipy.sparse` matrices or re-map user/item identifiers prior to use. RankFM internally maps all user/item identifiers to zero-based integer indexes, but always converts its output back to the original user/item identifiers from your data, which can be arbitrary (non-zero-based, non-consecutive) integers or even strings.

In addition to the familiar `fit()`, `predict()`, `recommend()` methods, RankFM includes additional utilities `similiar_users()` and `similar_items()` to find the most similar users/items to a given user/item based on latent factor space embeddings. A number of popular recommendation/ranking evaluation metric functions have been included in the separate `evaluation` module to streamline model tuning and validation.

* see the **Quickstart** section below to get started with the basic functionality
* see the `/examples` folder for more in-depth jupyter notebook walkthroughs with several popular open-source data sets
* see the [Online Documentation](https://rankfm.readthedocs.io/en/latest/) for more comprehensive documentation on the main model class and separate evaluation module
* see the [Medium Article](https://towardsdatascience.com/factorization-machines-for-item-recommendation-with-implicit-feedback-data-5655a7c749db) for contextual motivation and a detailed mathematical description of the algorithm

---
### Dependencies
* Python 3.9+
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
* **Dense feature columns**: a feature that is non-zero for most users (or most items) is written by every thread on nearly every step. Model quality is unaffected, but training stops getting faster beyond about 4-8 threads. One-hot or bucketed features scale to all cores.
* **Learning rate with side features**: every active feature on a row adds to the effective step size, so raw SGD steps diverge once rows carry more than a few features. `normalize_steps=True` (the default) divides each step by the feature mass of the rows it touches, which brings the stable range of `learning_rate` back to roughly that of a model without features. It is not a guarantee: a rate that diverges without features still diverges, and training warns when weights grow past 1e3. Pass `normalize_steps=False` for the unscaled updates of 0.4.0 and earlier.
* **Feature regularization**: `feature_reg='uniform'` works with any number of threads.
* **Incremental updates**: `fit_partial()` places new interactions by binary search and reuses spare capacity when users are added, so a small batch costs one block copy of the stored history and one scan of the weights, not a re-sort. Sequential integer user IDs keep constant-time lookups; with arbitrary IDs the ID index is rebuilt whenever a batch adds users.
* **Input checks**: NaN or infinite feature values and sample weights are rejected before training starts. Sparse feature matrices are copied, never sorted in place or referenced afterwards.
* **Size limits**: a single `fit()` call accepts up to 2,147,483,647 interactions and holds roughly 20 bytes per interaction in memory.

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
