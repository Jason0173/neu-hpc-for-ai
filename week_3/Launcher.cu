void gemm_op(const float* dA, int mA, int kA, char opA,
             const float* dB, int kB, int nB, char opB,
             float* dC, int m, int n, float alpha, float beta,
             cudaStream_t stream = 0)
{
    // After op:
    // op(A) is (m x K), op(B) is (K x n)
    int m_after = (opA == 'N') ? mA : kA;
    int K_A     = (opA == 'N') ? kA : mA;
    int K_B     = (opB == 'N') ? kB : nB;
    int n_after = (opB == 'N') ? nB : kB;

    // Basic shape checks (debug-time)
    // assert(m_after == m && n_after == n && K_A == K_B);

    int K = K_A;

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);

    gemm_op_inplace<<<grid, block, 0, stream>>>(
        dA, mA, kA, opA,
        dB, kB, nB, opB,
        dC, m, n, K,
        alpha, beta);
}
