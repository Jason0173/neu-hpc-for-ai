# Week 1: Multithreaded Matrix Multiplication

`matrix_st_updated.c` multiplies matrices with a single-threaded baseline and with a pthreads version that splits the output rows across worker threads.

```bash
gcc -O2 -pthread matrix_st_updated.c -o matmul -lm && ./matmul
```

## Correctness

Seven test cases, including 1×1, row and column vectors, non-square shapes and random matrices. In every case the multithreaded result matches both the expected values and the single-threaded result.

## Performance

Wall-clock time from one run on my machine (the numbers vary by CPU):

| Threads | 100×100 | 300×300 | 500×500 | Speedup (500×500) |
|---|---|---|---|---|
| 1 | 0.005 s | 0.081 s | 0.426 s | 1.00× |
| 4 | 0.002 s | 0.026 s | 0.123 s | 3.47× |
| 16 | 0.001 s | 0.019 s | 0.093 s | 4.58× |
| 32 | 0.002 s | 0.017 s | 0.088 s | 4.83× |
| 64 | 0.002 s | 0.017 s | 0.090 s | 4.72× |
| 128 | 0.002 s | 0.019 s | 0.109 s | 3.90× |

Speedup levels off after about 16 threads. It then drops at 128 threads, most likely because thread creation and scheduling overhead outweigh the extra parallelism once each thread has only a few rows. Parallel efficiency falls from 0.87 at 4 threads to 0.03 at 128.
