#pragma once

#include <stdint.h>

#ifdef __CUDACC__         // Only defined when compiled with nvcc
#include <cuda_runtime.h> // Provides __threadfence / atomicAdd / atomicSub etc.
#endif

// Task types: GEMM0 / GEMM1 / COMBINE
// Task types: FFN two matrix multiplications + combine step
enum TaskType : uint8_t {
    TASK_GEMM0  = 0,
    TASK_GEMM1  = 1,
    TASK_COMBINE = 2
};

// Generic Task: describes computation for one tile
// Offsets are in float elements, relative to symmetric layout base
struct Task {
    uint64_t a_offset;   // input tile A
    uint64_t b_offset;   // weight tile B
    uint64_t c_offset;   // output tile C
    uint64_t d_offset;   // optional bias/residual tile D

    uint32_t m, n, k;    // GEMM tile shape: [m x k] * [k x n] -> [m x n]

    uint16_t peer;       // remote PE id (if data stored on another PE)
    uint16_t expert;     // local expert id (for debug / routing)
    uint16_t round;      // ROUND_DISPATCH / ROUND_COMBINE
    uint8_t  type;       // TaskType

    uint8_t  pad[5];     // padding to 40 bytes (not strictly needed)
};

// Very simple lock-free ring buffer for Tasks
// Single producer / multiple consumer ring task queue
struct TaskQueue {
    Task     *buffer;    // [capacity] in symmetric/global memory
    uint32_t capacity;   // must be power-of-2
    uint32_t *head;      // consumer index (device global)
    uint32_t *tail;      // producer index (device/host)
};

// Device-side task enqueue: device-side enqueue
__device__ __forceinline__ bool tq_push(TaskQueue q, const Task& t) {
    uint32_t pos = *q.tail;
    if (pos >= q.capacity) return false;

    q.buffer[pos] = t;
    (*q.tail)++;

    __threadfence();    // Ensure task write is visible to other GPU threads (visibility fence)
    return true;
}

// Device-side task dequeue: device-side dequeue
__device__ __forceinline__ bool tq_pop(TaskQueue q, Task& out) {
    // Atomically increment head, "grab" a task index from queue
    uint32_t pos = atomicAdd(q.head, 1);

    if (pos >= *q.tail) {
        // Grabbed too many, rollback
        atomicSub(q.head, 1);
        return false;
    }

    out = q.buffer[pos];
    return true;
}

