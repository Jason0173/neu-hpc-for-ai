chapter_3
1.
a.
__global__ void matrixMultiplyKernel1(float* A_d, float* B_d, float* C_d, int m, int n) {
    // one output row per thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n) {
        for (int col = 0; col < m; ++col) {
            float res = 0;
            for (int k = 0; k < m; ++k) {
                res += A_d[row * m + k] * B_d[col + k * m];
            }
            C_d[row * m + col] = res;
        }
    }
}

b.
__global__ void matrixMultiplyKernel2(float* A_d, float* B_d, float* C_d, int m, int n) {
    // one output column per thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < m) {
        for (int row = 0; row < m; ++row) {
            float res = 0;
            for (int k = 0; k < m; ++k) {
                res += A_d[row * m + k] * B_d[col + k * m];
            }
            C_d[row * m + col] = res;
        }
    }
}

c.
These approaches are most useful when the matrix is too large to fit entirely into memory.

For a square matrix (where rows = columns), both methods perform equally well.

If the output matrix has more rows than columns (n > m), Solution A is the better choice.

If the output matrix has more columns than rows (m > n), Solution B works best.

2.
__global__ void matvec_kernel(const float* __restrict__ B,
                              const float* __restrict__ C,
                              float* __restrict__ A,
                              int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // one thread per output element 
    if (row >= n) return;

    float acc = 0.0f;
    int base = row * n;
    for (int i = 0; i < n; ++i) {
        acc += B[base + i] * C[i];   // dot product 
    }
    A[row] = acc;
}

3.
a.
16 * 32 = 512
b.
19 * 5 * 512 = 48,640
c.
gridDim.x = ceil(N/16) = ceil(300/16) = 19
gridDim.y = ceil(M/32) = ceil(150/32) = 5
Blocks in grid = 19 * 5 = 95
d.
45,000

4.
a.
20 * 400 + 10 = 8010
b.
10 *500 + 20 = 5020

5.
zmn + mx + y = 5 * 400 * 500 + 400 * 10 + 20 = 1004020

chapter_4
1.
a. 128/32 = 4
b. 8*4 = 32
c.
i. 24
ii. 8 * 2 = 16
iii. 100%
iv. 8/32 = 25%
v. 75%
d.
i. 32
ii. 32
iii. 16/32 = 50%
e.
i. 3
ii. 2


2.
512 * 4 = 2048

3.
1

4.
Sum waits = (3.0−2.0)+(3.0−2.3)+…+(3.0−2.9) = 4.1 μs
Percentage = 4.1 / 24.0 ≈ 0.1708 → 17.1%.

5
No.
One shouldn’t assume that all threads in a warp execute with the same execution timing, so this could still cause them to have sync problems.

6.
c

7.
All of them are possible,
a) 50%
b) 50%
c) 50%
d) 100%
e) 100%

8.
a. Yes.
b. Not enough block slots.
c. Not enough registers.

9.
I don’t believe that exact setup ran as stated. Either they ran on a GPU with 1024 threads/block support, or they actually used 16*16 (or similar) blocks, or they changed the work mapping so one thread computes multiple outputs.

chapter_5
1.
No. why: no reuse

2.
M (8x8), tiles 2x2                    N (8x8), tiles 2x2
+--+--+--+--+                         +--+--+--+--+
|##|##|##|##| <- row 0..1             |##|##|##|##|
+--+--+--+--+                         +--+--+--+--+
|##|##|##|##| <- row 2..3             |##|##|##|##|
+--+--+--+--+                         +--+--+--+--+
|##|##|##|##| <- row 4..5             |##|##|##|##|
+--+--+--+--+                         +--+--+--+--+
|##|##|##|##| <- row 6..7             |##|##|##|##|
+--+--+--+--+                         +--+--+--+--+
T=2: Loads = 1024 / 2 = 512 → 2× reduction

M (8x8), tiles 4x4                    N (8x8), tiles 4x4
+----+----+                           +----+----+
|####|####| <- row 0..3               |####|####|
+----+----+                           +----+----+
|####|####| <- row 4..7               |####|####|
+----+----+                           +----+----+
T=4: Loads = 1024 / 4 = 256 → 4× reduction

3.
If one forget the first one, invalid elements may be accessed when computing the partial matmul. If one forgets the second one, a thread further down the execution stream might overwrite a value in shared memory that a thread performing the partial matmul computation might need, thus corrputing the result.

4.
Even if capacity isn’t a problem, shared memory is valuable because it is visible to all threads in a block, while registers are private to each thread. That means only shared memory allows inter-thread cooperation and data reuse, which is essential for reducing global memory traffic.

5.
32x reduction

6.
512000 versions.

7.
1,000

8.
a: N times
b: N/T times

9.
a) memory-bound
b) compute-bound

10.
a) Only BLOCK_WIDTH == 1 executes correctly without synchronization.

b) The root cause is missing barrier synchronization when threads read each other’s shared memory values. Adding __syncthreads() (after the write and before the read) fixes the code for all block sizes (1…20).

11.
a.
1024, one per thread. In registers.

b.
Also 1024, for the same reason. In registers because of the constant access pattern (we assume the footnote in page 101 applies here).

c.
1 per block, so 8. In shared memory.

d.
1 per block, so 8. In shared memory.

e.
y_s + b_s = 1 * 4 + 128 * 4 = 516 bytes. (assuming 32 bit floats)

f.
Each thread reads from global memory 5 elements: 4 from a (line 7) and 1 from b (line 12). It performs 10 floating point operations (lines 14, 15): 5 prods and 5 sums. So, flop to global memory access ratio is 10/5 = 2 ops per byte.

12.
a. The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared memory/SM.
I'm not sure whether "4 KB of shared memory/SM" refers to total memory used by all blocks in the SM, or the memory used by a single block. I'll assume the latter. In this case, the limiting factor is shared memory: all 32 blocks of 64 threads could be scheduled on the SM, but would use 32 * 4 KB = 128 KB of shared memory, which is more than the 96 KB available.

b. The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of shared memory/SM.
With the same assumption as before, there is no limiting factor and the kernel can achieve full occupancy.