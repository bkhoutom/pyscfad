/* Test-only observer: run the original GNU OpenMP team and its original work. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>

typedef void (*parallel_fn)(void (*)(void *), void *, unsigned, unsigned);
static int active;
static int counts[2];

struct observed_work {
    void (*function)(void *);
    void *data;
    int (*get_dynamic)(void);
};

static void observe_worker(void *data)
{
    struct observed_work *work = data;
    const int dynamic = work->get_dynamic() != 0;
    __atomic_fetch_add(&counts[dynamic], 1, __ATOMIC_RELAXED);
    work->function(work->data);
}

void GOMP_parallel(void (*function)(void *), void *data,
                   unsigned threads, unsigned flags)
{
    parallel_fn original = (parallel_fn)dlsym(RTLD_NEXT, "GOMP_parallel");
    if (original == NULL) abort();
    if (!active) {
        original(function, data, threads, flags);
        return;
    }
    struct observed_work work = {
        function, data, (int (*)(void))dlsym(RTLD_NEXT, "omp_get_dynamic")
    };
    if (work.get_dynamic == NULL) abort();
    original(observe_worker, &work, threads, flags);
}

void policy_probe_start(void)
{
    counts[0] = counts[1] = 0;
    active = 1;
}

void policy_probe_stop(void)
{
    active = 0;
}

int policy_probe_count(int dynamic)
{
    return counts[dynamic != 0];
}
