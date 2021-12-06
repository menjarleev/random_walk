#include "rwalk.cuh"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <limits>

int64_t* dev_global_walk;
int64_t* dev_node_idx;
float* dev_timestamp;
int64_t* dev_start_idx;

int threadBlockSize;
cudaDeviceProp prop;


// assert(err == cudaSuccess);

#define cudaCheck(err) { \
	if (err != cudaSuccess) { \
		printf("CUDA error: %s: %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
	} \
}

// rand_walk -> [num_of_node, num_of_walk, max_walk_length]
void __global__ singleRandomWalk(int num_of_node, int num_of_walk, int max_walk_length, int64_t* node_idx, float* timestamp, int64_t* start_idx, int64_t* rand_walk, unsigned long long rnumber){
    // assuming grid = 1
    int64_t i =  (blockDim.x * blockIdx.x) + threadIdx.x;
    if(i >= num_of_node * num_of_walk){
        return;
    }

    int64_t src_node = i / (int64_t) num_of_walk;
    float curr_timestamp = .0f;
    rand_walk[i * max_walk_length + 0] = src_node;

    // printf("start : %lld ; end : %lld; src_node: %lld; num_of_walk : %d; max_walk_length: %d; i : %lld\n", (long long int)start, (long long int)end, (long long int)src_node, num_of_walk, max_walk_length, (long long int)i);
    int64_t start;
    int64_t end;

    int walk_cnt;
    for(walk_cnt = 1; walk_cnt < max_walk_length; walk_cnt ++){
        // ! can be improved
        start = start_idx[src_node];
        end = start_idx[src_node + 1];
        // printf("start: %lld end: %lld\n", (long long int) start, (long long int)end);

        // control divergence
        // range should be [start, end)
        if(start < end){
            float* valid_timestamp = (float*) malloc((end - start) * sizeof(float));
            int64_t* valid_node = (int64_t*) malloc((end - start) * sizeof(int64_t));
            int idx = 0;
            // float cdf[end - start];
            float max_timestamp = timestamp[start];
            float min_timestamp = timestamp[start];
            // ! parallizable
            for(int64_t j = start; j < end; j ++){
                // printf("idx: %lld, timestamp: %f node_idx: %lld\n", (long long int) j, timestamp[j], (long long int) node_idx);
                if(timestamp[j] > curr_timestamp){
                    valid_node[idx] = node_idx[j];
                    valid_timestamp[idx++] = timestamp[j];
                }
                max_timestamp = max(max_timestamp, timestamp[j]);
                min_timestamp = min(min_timestamp, timestamp[j]);
            }
            // printf("idx: %d\n", idx);
            if(!idx){
                free(valid_node);
                free(valid_timestamp);
                break;
            }

            // every timestamp is the same
            if(max_timestamp - min_timestamp >= - 0.0000001 && max_timestamp - min_timestamp <= 0.0000001){
                printf("valid node: %lld\n", (long long int)valid_node[0]);
                rand_walk[i * max_walk_length + walk_cnt] = valid_node[0];
                src_node = valid_node[0];
                curr_timestamp = valid_timestamp[0];
                free(valid_node);
                free(valid_timestamp);
                continue;
            }

            float* cdf = (float*) malloc(idx * sizeof(float));

            // ! need to determine how to get prob
            float prob = rnumber * 1.0 / ULLONG_MAX;

            // refresh rnumber
            rnumber = rnumber * (unsigned long long)25214903917 + 11;
            bool fall_through = true;

            // ! reduction tree here (kernel in kernel)
            float denom = .0f;
            for(int j = 0; j < idx; j ++){
                printf("J: %d\n", j);
                cdf[j] =  expf((valid_timestamp[j] - curr_timestamp) / (max_timestamp - min_timestamp));
                denom += cdf[j];
            }
            float curr_cdf = .0f,  next_cdf = .0f;
            for(int j = 0; j < idx; j ++){
                next_cdf += cdf[j] / denom;
                if(prob >= curr_cdf && prob <= next_cdf){
                    printf("valid node: %lld\n", (long long int)valid_node[j]);
                    rand_walk[i * max_walk_length + walk_cnt] = valid_node[j];
                    src_node = valid_node[j];
                    curr_timestamp = valid_timestamp[j];
                    fall_through = false;
                    break;
                }
                curr_cdf = next_cdf;
            }

            // fall through should never happen
            if(fall_through){
                // printf("valid node: %lld\n", (long long int)valid_node[0]);
                rand_walk[i * max_walk_length + walk_cnt] = valid_node[0];
                src_node = valid_node[0];
                curr_timestamp = valid_timestamp[0];
            }

            free(valid_node);
            free(valid_timestamp);
            free(cdf);
        }
        else{
            break;
        }
    }

    if(walk_cnt < max_walk_length){
        // signal the rest is invalid and there is no descending node
        rand_walk[i * max_walk_length + walk_cnt] = -1;
    }
}




void cuda_rwalk(int max_walk_length, int num_walks_per_node, int64_t num_nodes, int64_t num_edges, unsigned long long random_number){

    size_t free_memory;
    size_t total_memory;

    cudaCheck(cudaMemGetInfo(&free_memory, &total_memory));
    // printf("free memory : %zu ; total memory : %zu\n", free_memory, total_memory);

    // malloc GPU memory
    cudaCheck(cudaMalloc((void **)&dev_start_idx, sizeof(int64_t) * (num_nodes + 1)));
    cudaCheck(cudaMalloc((void **)&dev_node_idx, sizeof(int64_t) * num_edges));
    cudaCheck(cudaMalloc((void **)&dev_timestamp, sizeof(float) * num_edges));
    cudaCheck(cudaMalloc((void **)&dev_global_walk, sizeof(int64_t) * num_nodes * max_walk_length * num_walks_per_node));

    // memcpy
    cudaCheck(cudaMemcpy(dev_start_idx, start_idx_host, sizeof(int64_t) * (num_nodes + 1), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_node_idx, node_idx_host, sizeof(int64_t) * num_edges, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_timestamp, timestamp_host, sizeof(float) * num_edges, cudaMemcpyHostToDevice));

    // cudaGetDeviceCount(&dev_count);
    // for(int i = 0; i < dev_count; i ++){
    //     printf("total_global_Mem: %zu\n shared_mem_per_block: %zu\n max_threads_per_block: %d\n max_thread_dim: %d\n max_grid_size: %d",
    //     prop.totalGlobalMem, prop.sharedMemPerBlock, prop.maxThreadsPerBlock, prop.maxThreadsDim, prop.maxGridSize);

    // }
    cudaGetDeviceProperties(&prop, 0);
    threadBlockSize = prop.maxThreadsPerBlock;
    // start training
    int grid_size = (num_nodes * num_walks_per_node - 1 ) / 32 + 1;
    // printf("grid_size: %d\n", grid_size);
    dim3 gridDim(grid_size);
    dim3 blockDim(32);
    // ?? header file
    singleRandomWalk<<<gridDim, blockDim>>>(num_nodes, num_walks_per_node, max_walk_length, dev_node_idx, dev_timestamp, dev_start_idx, dev_global_walk, random_number);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    // cudaCheck(cudaMemGetInfo(&free_memory, &total_memory));
    // printf("free memory : %zu ; total memory : %zu\n", free_memory, total_memory);

    // get result
    cudaDeviceSynchronize();
    cudaCheck(cudaMemcpy(random_walk_host, dev_global_walk, sizeof(int64_t) * num_nodes * max_walk_length * num_walks_per_node, cudaMemcpyDeviceToHost));

    // clean arrays
    cudaCheck(cudaFree(dev_start_idx));
    cudaCheck(cudaFree(dev_node_idx));
    cudaCheck(cudaFree(dev_timestamp));
    cudaCheck(cudaFree(dev_global_walk));
}

