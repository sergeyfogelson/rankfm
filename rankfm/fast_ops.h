#ifndef RANKFM_FAST_OPS_H
#define RANKFM_FAST_OPS_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <malloc.h>
#include <process.h>
#include <windows.h>
#else
#include <pthread.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define RFM_RESTRICT __restrict__
#define RFM_INLINE static inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define RFM_RESTRICT __restrict
#define RFM_INLINE static __forceinline
#else
#define RFM_RESTRICT
#define RFM_INLINE static inline
#endif

/* ---------------------------------------------------------
 * 1. 64-Byte Cache-Line Aligned Scratch Memory Allocators
 * --------------------------------------------------------- */

static inline void *rfm_aligned_malloc(size_t nbytes) {
    size_t padded = ((nbytes == 0 ? 64 : nbytes) + 63u) & ~((size_t)63u);
#ifdef _WIN32
    return _aligned_malloc(padded, 64);
#else
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, padded) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

static inline void rfm_aligned_free(void *ptr) {
    if (ptr == NULL) {
        return;
    }
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

RFM_INLINE int64_t rfm_atomic_fetch_add_i64(int64_t *ptr, int64_t val) {
#ifdef _WIN32
    return (int64_t)InterlockedExchangeAdd64((volatile LONG64 *)ptr, (LONG64)val);
#else
    return __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED);
#endif
}

/* ---------------------------------------------------------
 * 2. Thread-Local Lock-Free xoshiro256** PRNG + Lemire Draw
 * --------------------------------------------------------- */

typedef struct {
    uint64_t s[4];
} rfm_rng_t;

RFM_INLINE uint64_t rfm_rotl64(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

RFM_INLINE uint64_t rfm_splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline void rfm_rng_seed(rfm_rng_t *rng, uint64_t seed, uint64_t stream) {
    uint64_t sm = seed ^ (stream * 0xD1B54A32D192ED03ULL + 0x9E3779B97F4A7C15ULL);
    rng->s[0] = rfm_splitmix64(&sm);
    rng->s[1] = rfm_splitmix64(&sm);
    rng->s[2] = rfm_splitmix64(&sm);
    rng->s[3] = rfm_splitmix64(&sm);
}

RFM_INLINE uint64_t rfm_rng_u64(rfm_rng_t *rng) {
    const uint64_t result = rfm_rotl64(rng->s[1] * 5ULL, 7) * 9ULL;
    const uint64_t t = rng->s[1] << 17;
    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3] = rfm_rotl64(rng->s[3], 45);
    return result;
}

RFM_INLINE uint32_t rfm_rng_u32(rfm_rng_t *rng) {
    return (uint32_t)(rfm_rng_u64(rng) >> 32);
}

/* Divisionless bounded draw in [0, bound) (Lemire, 2019) */
RFM_INLINE uint32_t rfm_rng_bounded(rfm_rng_t *rng, uint32_t bound) {
    return (uint32_t)(((uint64_t)rfm_rng_u32(rng) * (uint64_t)bound) >> 32);
}

RFM_INLINE float rfm_rng_float01(rfm_rng_t *rng) {
    return (float)(rfm_rng_u32(rng) >> 8) * (1.0f / 16777216.0f);
}

static inline void rfm_shuffle_int32(int32_t *arr, int64_t n, rfm_rng_t *rng) {
    int64_t i, j;
    int32_t tmp;
    for (i = n - 1; i > 0; i--) {
        if (i < 0x7FFFFFFFLL) {
            j = (int64_t)rfm_rng_bounded(rng, (uint32_t)(i + 1));
        } else {
            j = (int64_t)(rfm_rng_u64(rng) % (uint64_t)(i + 1));
        }
        tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

/* ---------------------------------------------------------
 * 3. 4-Way Unrolled SIMD-Friendly Vector Math Kernels
 * --------------------------------------------------------- */

RFM_INLINE float rfm_dot(int F, const float * RFM_RESTRICT a, const float * RFM_RESTRICT b) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int f = 0;
    const int limit16 = F - 15;
    for (; f < limit16; f += 16) {
        s0 += a[f + 0] * b[f + 0] + a[f + 4] * b[f + 4] + a[f + 8] * b[f + 8] + a[f + 12] * b[f + 12];
        s1 += a[f + 1] * b[f + 1] + a[f + 5] * b[f + 5] + a[f + 9] * b[f + 9] + a[f + 13] * b[f + 13];
        s2 += a[f + 2] * b[f + 2] + a[f + 6] * b[f + 6] + a[f + 10] * b[f + 10] + a[f + 14] * b[f + 14];
        s3 += a[f + 3] * b[f + 3] + a[f + 7] * b[f + 7] + a[f + 11] * b[f + 11] + a[f + 15] * b[f + 15];
    }
    const int limit4 = F - 3;
    for (; f < limit4; f += 4) {
        s0 += a[f + 0] * b[f + 0];
        s1 += a[f + 1] * b[f + 1];
        s2 += a[f + 2] * b[f + 2];
        s3 += a[f + 3] * b[f + 3];
    }
    float tail = 0.0f;
    for (; f < F; f++) {
        tail += a[f] * b[f];
    }
    return ((s0 + s1) + (s2 + s3)) + tail;
}

RFM_INLINE void rfm_vec_zero(int F, float * RFM_RESTRICT dst) {
    memset(dst, 0, (size_t)F * sizeof(float));
}

RFM_INLINE void rfm_vec_copy(int F, const float * RFM_RESTRICT src, float * RFM_RESTRICT dst) {
    memcpy(dst, src, (size_t)F * sizeof(float));
}

RFM_INLINE void rfm_vec_add(int F, const float * RFM_RESTRICT a, const float * RFM_RESTRICT b, float * RFM_RESTRICT out) {
    int f = 0;
    for (; f <= F - 4; f += 4) {
        out[f + 0] = a[f + 0] + b[f + 0];
        out[f + 1] = a[f + 1] + b[f + 1];
        out[f + 2] = a[f + 2] + b[f + 2];
        out[f + 3] = a[f + 3] + b[f + 3];
    }
    for (; f < F; f++) {
        out[f] = a[f] + b[f];
    }
}

RFM_INLINE void rfm_vec_sub(int F, const float * RFM_RESTRICT a, const float * RFM_RESTRICT b, float * RFM_RESTRICT out) {
    int f = 0;
    for (; f <= F - 4; f += 4) {
        out[f + 0] = a[f + 0] - b[f + 0];
        out[f + 1] = a[f + 1] - b[f + 1];
        out[f + 2] = a[f + 2] - b[f + 2];
        out[f + 3] = a[f + 3] - b[f + 3];
    }
    for (; f < F; f++) {
        out[f] = a[f] - b[f];
    }
}

RFM_INLINE void rfm_vec_axpy(int F, float alpha, const float * RFM_RESTRICT x, float * RFM_RESTRICT y) {
    int f = 0;
    for (; f <= F - 4; f += 4) {
        y[f + 0] += alpha * x[f + 0];
        y[f + 1] += alpha * x[f + 1];
        y[f + 2] += alpha * x[f + 2];
        y[f + 3] += alpha * x[f + 3];
    }
    for (; f < F; f++) {
        y[f] += alpha * x[f];
    }
}

RFM_INLINE void rfm_vec_reg_axpy(int F, float keep, float alpha, const float * RFM_RESTRICT x, float * RFM_RESTRICT y) {
    int f = 0;
    for (; f <= F - 4; f += 4) {
        y[f + 0] = y[f + 0] * keep + alpha * x[f + 0];
        y[f + 1] = y[f + 1] * keep + alpha * x[f + 1];
        y[f + 2] = y[f + 2] * keep + alpha * x[f + 2];
        y[f + 3] = y[f + 3] * keep + alpha * x[f + 3];
    }
    for (; f < F; f++) {
        y[f] = y[f] * keep + alpha * x[f];
    }
}

RFM_INLINE void rfm_vec_scale(int F, float keep, float * RFM_RESTRICT y) {
    int f = 0;
    for (; f <= F - 4; f += 4) {
        y[f + 0] *= keep;
        y[f + 1] *= keep;
        y[f + 2] *= keep;
        y[f + 3] *= keep;
    }
    for (; f < F; f++) {
        y[f] *= keep;
    }
}

RFM_INLINE void rfm_dual_item_update(
    int F,
    float keep,
    float step,
    const float * RFM_RESTRICT u_emb,
    float * RFM_RESTRICT vi,
    float * RFM_RESTRICT vj,
    float * RFM_RESTRICT veff_i,
    float * RFM_RESTRICT veff_j,
    int update_veff
) {
    int f;
    if (update_veff) {
        for (f = 0; f < F; f++) {
            const float uf = u_emb[f];
            const float old_i = vi[f];
            const float new_i = old_i * keep + step * uf;
            vi[f] = new_i;
            veff_i[f] += (new_i - old_i);

            const float old_j = vj[f];
            const float new_j = old_j * keep - step * uf;
            vj[f] = new_j;
            veff_j[f] += (new_j - old_j);
        }
    } else {
        for (f = 0; f < F; f++) {
            const float uf = u_emb[f];
            vi[f] = vi[f] * keep + step * uf;
            vj[f] = vj[f] * keep - step * uf;
        }
    }
}

/* ---------------------------------------------------------
 * 4. Zero-Dependency Reusable Barrier Thread Pool
 * --------------------------------------------------------- */

typedef void (*rfm_worker_fn)(int tid, int n_threads, void *ctx);

#ifdef _WIN32

typedef struct {
    int n_threads;
} rfm_pool_t;

typedef struct {
    rfm_worker_fn fn;
    void *ctx;
    int tid;
    int n_threads;
    int started;
} rfm_win_arg_t;

static unsigned __stdcall rfm_win_entry(void *raw) {
    rfm_win_arg_t *a = (rfm_win_arg_t *)raw;
    a->fn(a->tid, a->n_threads, a->ctx);
    return 0;
}

static inline rfm_pool_t *rfm_pool_create(int n_threads) {
    if (n_threads <= 1) return NULL;
    rfm_pool_t *p = (rfm_pool_t *)malloc(sizeof(rfm_pool_t));
    if (p != NULL) p->n_threads = n_threads;
    return p;
}

static inline void rfm_pool_run(rfm_pool_t *pool, rfm_worker_fn fn, void *ctx) {
    if (pool == NULL || pool->n_threads <= 1) {
        fn(0, 1, ctx);
        return;
    }
    int nt = pool->n_threads;
    if (nt > MAXIMUM_WAIT_OBJECTS + 1) nt = MAXIMUM_WAIT_OBJECTS + 1;
    HANDLE *handles = (HANDLE *)malloc((size_t)(nt - 1) * sizeof(HANDLE));
    rfm_win_arg_t *args = (rfm_win_arg_t *)malloc((size_t)(nt - 1) * sizeof(rfm_win_arg_t));
    if (handles == NULL || args == NULL) {
        free(handles);
        free(args);
        fn(0, 1, ctx);
        return;
    }
    int spawned = 0;
    for (int t = 1; t < nt; t++) {
        args[t - 1].fn = fn;
        args[t - 1].ctx = ctx;
        args[t - 1].tid = t;
        args[t - 1].n_threads = nt;
        uintptr_t h = _beginthreadex(NULL, 0, rfm_win_entry, &args[t - 1], 0, NULL);
        args[t - 1].started = (h != 0);
        if (h != 0) handles[spawned++] = (HANDLE)h;
    }
    fn(0, nt, ctx);
    /* Workers split the input by thread id, so a thread that failed to start would leave its share
       unprocessed. Run those shares here on the calling thread instead. */
    for (int t = 1; t < nt; t++) {
        if (!args[t - 1].started) fn(t, nt, ctx);
    }
    if (spawned > 0) {
        WaitForMultipleObjects((DWORD)spawned, handles, TRUE, INFINITE);
        for (int i = 0; i < spawned; i++) CloseHandle(handles[i]);
    }
    free(handles);
    free(args);
}

static inline void rfm_pool_destroy(rfm_pool_t *pool) {
    free(pool);
}

#else /* POSIX pthreads (macOS & Linux) */

typedef struct rfm_pool_s rfm_pool_t;

typedef struct {
    rfm_pool_t *pool;
    int tid;
} rfm_pthread_arg_t;

struct rfm_pool_s {
    int n_threads;
    pthread_t *threads;
    rfm_pthread_arg_t *args;
    pthread_mutex_t mtx;
    pthread_cond_t cv_start;
    pthread_cond_t cv_done;
    uint64_t gen;
    int active_workers;
    int shutdown;
    rfm_worker_fn fn;
    void *ctx;
};

static void *rfm_pthread_worker(void *raw_arg) {
    rfm_pthread_arg_t *arg = (rfm_pthread_arg_t *)raw_arg;
    rfm_pool_t *pool = arg->pool;
    const int tid = arg->tid;
    uint64_t my_gen = 0;

    for (;;) {
        pthread_mutex_lock(&pool->mtx);
        while (!pool->shutdown && pool->gen == my_gen) {
            pthread_cond_wait(&pool->cv_start, &pool->mtx);
        }
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mtx);
            break;
        }
        my_gen = pool->gen;
        rfm_worker_fn fn = pool->fn;
        void *ctx = pool->ctx;
        const int nt = pool->n_threads;
        pthread_mutex_unlock(&pool->mtx);

        fn(tid, nt, ctx);

        pthread_mutex_lock(&pool->mtx);
        pool->active_workers--;
        if (pool->active_workers == 0) {
            pthread_cond_signal(&pool->cv_done);
        }
        pthread_mutex_unlock(&pool->mtx);
    }
    return NULL;
}

static inline rfm_pool_t *rfm_pool_create(int n_threads) {
    if (n_threads <= 1) {
        return NULL;
    }
    rfm_pool_t *pool = (rfm_pool_t *)calloc(1, sizeof(rfm_pool_t));
    if (pool == NULL) {
        return NULL;
    }
    pool->n_threads = n_threads;
    pool->threads = (pthread_t *)calloc((size_t)(n_threads - 1), sizeof(pthread_t));
    pool->args = (rfm_pthread_arg_t *)calloc((size_t)(n_threads - 1), sizeof(rfm_pthread_arg_t));
    if (pool->threads == NULL || pool->args == NULL) {
        free(pool->threads);
        free(pool->args);
        free(pool);
        return NULL;
    }
    pthread_mutex_init(&pool->mtx, NULL);
    pthread_cond_init(&pool->cv_start, NULL);
    pthread_cond_init(&pool->cv_done, NULL);

    int created = 0;
    for (int t = 1; t < n_threads; t++) {
        pool->args[t - 1].pool = pool;
        pool->args[t - 1].tid = t;
        if (pthread_create(&pool->threads[t - 1], NULL, rfm_pthread_worker, &pool->args[t - 1]) == 0) {
            created++;
        } else {
            break;
        }
    }
    if (created < n_threads - 1) {
        pthread_mutex_lock(&pool->mtx);
        pool->shutdown = 1;
        pthread_cond_broadcast(&pool->cv_start);
        pthread_mutex_unlock(&pool->mtx);
        for (int i = 0; i < created; i++) {
            pthread_join(pool->threads[i], NULL);
        }
        pthread_mutex_destroy(&pool->mtx);
        pthread_cond_destroy(&pool->cv_start);
        pthread_cond_destroy(&pool->cv_done);
        free(pool->threads);
        free(pool->args);
        free(pool);
        return NULL;
    }
    return pool;
}

static inline void rfm_pool_run(rfm_pool_t *pool, rfm_worker_fn fn, void *ctx) {
    if (pool == NULL || pool->n_threads <= 1) {
        fn(0, 1, ctx);
        return;
    }
    pthread_mutex_lock(&pool->mtx);
    pool->fn = fn;
    pool->ctx = ctx;
    pool->active_workers = pool->n_threads - 1;
    pool->gen++;
    pthread_cond_broadcast(&pool->cv_start);
    pthread_mutex_unlock(&pool->mtx);

    fn(0, pool->n_threads, ctx);

    pthread_mutex_lock(&pool->mtx);
    while (pool->active_workers > 0) {
        pthread_cond_wait(&pool->cv_done, &pool->mtx);
    }
    pthread_mutex_unlock(&pool->mtx);
}

static inline void rfm_pool_destroy(rfm_pool_t *pool) {
    if (pool == NULL) {
        return;
    }
    pthread_mutex_lock(&pool->mtx);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->cv_start);
    pthread_mutex_unlock(&pool->mtx);

    for (int i = 0; i < pool->n_threads - 1; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    pthread_mutex_destroy(&pool->mtx);
    pthread_cond_destroy(&pool->cv_start);
    pthread_cond_destroy(&pool->cv_done);
    free(pool->threads);
    free(pool->args);
    free(pool);
}

#endif /* _WIN32 */

#endif /* RANKFM_FAST_OPS_H */
