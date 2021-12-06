// #ifndef RWALKCU_H_
// #define RWALKCU_H_

#include <stdio.h>
#include <assert.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

// decleration
extern int64_t* dev_global_walk;
extern int64_t* dev_node_idx;
extern float* dev_timestamp;
extern int64_t* dev_start_idx;

extern int64_t *start_idx_host;
extern int64_t *node_idx_host;
extern float *timestamp_host;
extern int64_t *random_walk_host;


extern int dev_count;


// extern cudaDeviceProp prop;


// void __global__ singleRandomWalk(int num_of_node, int num_of_walk, int max_walk_length, int64_t* node_idx, float* timestamp, int64_t* dev_start_idx, int64_t* rand_walk);

void cuda_rwalk(int max_walk_length, int num_walks_per_node, int64_t num_nodes, int64_t num_edges, unsigned long long random_number);

// #endif /* RWALKCU_H_ */