#pragma once

#include "flash_dmoe_layout.h"
#include "flash_dmoe_task.h"

#ifdef __CUDACC__
extern "C"
__global__ void flash_dmoe_kernel(
        float *L_base,           // symmetric layout base pointer
        SymLayoutConfig cfg,     // layout config (by value, POD)
        Task *task_buffer,       // Task buffer
        uint32_t *q_head,        // queue head pointer
        uint32_t *q_tail,        // queue tail pointer
        uint32_t q_capacity,     // queue capacity
        uint32_t total_tasks_est // Used by scheduler to determine when to end
);
#endif
