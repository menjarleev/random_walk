#include "rwalk.cuh"
#include <stdio.h>
#include <assert.h>
#include <limits>

void __global__ multipleRandomWalk(int num_of_node, int num_of_walk, int max_walk_length, int64_t* node_idx, float* timestamp, int64_t* start_idx, int64_t* rand_walk, unsigned long long rnumber){
  // assuming grid = 1
  int64_t src_node =  (blockDim.x * blockIdx.x) + threadIdx.x;
  if(src_node >= num_of_node){
      return;
  }

  for (int k = 0; k < num_of_walk; k++)
  {
    int i = src_node * num_of_walk + k;

    float curr_timestamp = .0f;
    rand_walk[i * max_walk_length + 0] = src_node;

    int walk_cnt;
    for(walk_cnt = 1; walk_cnt < max_walk_length; walk_cnt ++){
        int64_t start = start_idx[src_node];
        int64_t end = start_idx[src_node];

        // control divergence
        // range should be [start, end)
        if(start < end){
            float* cdf = (float*) malloc((end - start) * sizeof(float));
            int64_t* valid_node_idx = (int64_t*) malloc((end - start) * sizeof(int64_t));
            int idx = 0;
            int valid_neighbor_cnt = 0;
            // float cdf[end - start];
            float max_timestamp = timestamp[start];
            float min_timestamp = timestamp[start];
            // ! parallizable
            for(int j = start; j < end; j ++){
                if(timestamp[j] > curr_timestamp){
                    valid_neighbor_cnt ++;
                    valid_node_idx[idx] = j;
                    cdf[idx++] = timestamp[j];
                }
                max_timestamp = max(curr_timestamp, timestamp[j]);
                min_timestamp = min(min_timestamp, timestamp[j]);
            }
            if(!valid_neighbor_cnt){
                free(valid_node_idx);
                free(cdf);
                break;
            }

            // every timestamp is the same
            if(max_timestamp - min_timestamp >= - 0.0000001 && max_timestamp - min_timestamp <= 0.0000001){
                rand_walk[i * max_walk_length + walk_cnt] = valid_node_idx[0];
                src_node = valid_node_idx[0];
                curr_timestamp = cdf[0];
                free(valid_node_idx);
                free(cdf);
                continue;
            }

            // ! need to determine how to get prob
            float prob = rnumber * 1.0 / ULLONG_MAX;

            // refresh rnumber
            rnumber = rnumber * (unsigned long long)25214903917 + 11;
            bool fall_through = true;

            // ! reduction tree here (kernel in kernel)
            float denom = .0f;
            for(int j = 0; j < valid_neighbor_cnt; j ++){
                cdf[j] =  expf((cdf[j] - curr_timestamp) / (max_timestamp - min_timestamp));
                denom += cdf[j];
            }
            float curr_cdf = 0,  next_cdf = .0f;
            for(int j = 0; j < valid_neighbor_cnt; j ++){
                next_cdf += cdf[j] / denom;
                if(prob >= curr_cdf && prob < next_cdf){
                    src_node = node_idx[start + j];
                    curr_timestamp = timestamp[start + j];
                    fall_through = false;
                    break;
                }
                curr_cdf = next_cdf;
            }

            // fall through should never happen
            if(fall_through){
                rand_walk[i * max_walk_length + walk_cnt] = valid_node_idx[0];
                src_node = valid_node_idx[0];
                curr_timestamp = cdf[0];
            }

            free(valid_node_idx);
            free(cdf);
        }
    }

    if(walk_cnt < max_walk_length){
        // signal the rest is invalid and there is no descending node
        rand_walk[i * max_walk_length + walk_cnt] = -1;
    }
  }
}

void __global__ singleRandomWalk(int num_of_node, int num_of_walk, int max_walk_length, int64_t* node_idx, float* timestamp, int64_t* start_idx, int64_t* rand_walk, unsigned long long rnumber){
  // assuming grid = 1
  int64_t i =  (blockDim.x * blockIdx.x) + threadIdx.x;
  if(i >= num_of_node * num_of_walk){
      return;
  }
  int64_t src_node = i / num_of_walk;
  float curr_timestamp = .0f;
  rand_walk[i * max_walk_length + 0] = src_node;

  int walk_cnt;
  for(walk_cnt = 1; walk_cnt < max_walk_length; walk_cnt ++){
      int64_t start = start_idx[src_node];
      int64_t end = start_idx[src_node];

      // control divergence
      // range should be [start, end)
      if(start < end){
          float* cdf = (float*) malloc((end - start) * sizeof(float));
          int64_t* valid_node_idx = (int64_t*) malloc((end - start) * sizeof(int64_t));
          int idx = 0;
          // float cdf[end - start];
          float max_timestamp = timestamp[start];
          float min_timestamp = timestamp[start];
          // ! parallizable
          for(int j = start; j < end; j ++){
              if(timestamp[j] > curr_timestamp){
                  valid_node_idx[idx] = j;
                  cdf[idx++] = timestamp[j];
              }
              max_timestamp = max(curr_timestamp, timestamp[j]);
              min_timestamp = min(min_timestamp, timestamp[j]);
          }
          if(!idx){
              free(valid_node_idx);
              free(cdf);
              break;
          }

          // every timestamp is the same
          if(max_timestamp - min_timestamp >= - 0.0000001 && max_timestamp - min_timestamp <= 0.0000001){
              rand_walk[i * max_walk_length + walk_cnt] = valid_node_idx[0];
              src_node = valid_node_idx[0];
              curr_timestamp = cdf[0];
              free(valid_node_idx);
              free(cdf);
              continue;
          }

          // ! need to determine how to get prob
          float prob = rnumber * 1.0 / ULLONG_MAX;

          // refresh rnumber
          rnumber = rnumber * (unsigned long long)25214903917 + 11;
          bool fall_through = true;

          // ! reduction tree here (kernel in kernel)
          float denom = .0f;
          for(int j = 0; j < idx; j ++){
              cdf[j] =  expf((cdf[j] - curr_timestamp) / (max_timestamp - min_timestamp));
              denom += cdf[j];
          }
          float curr_cdf = 0,  next_cdf = .0f;
          for(int j = 0; j < idx; j ++){
              next_cdf += cdf[j] / denom;
              if(prob >= curr_cdf && prob < next_cdf){
                  src_node = node_idx[start + j];
                  curr_timestamp = timestamp[start + j];
                  fall_through = false;
                  break;
              }
              curr_cdf = next_cdf;
          }

          // fall through should never happen
          if(fall_through){
              rand_walk[i * max_walk_length + walk_cnt] = valid_node_idx[0];
              src_node = valid_node_idx[0];
              curr_timestamp = cdf[0];
          }

          free(valid_node_idx);
          free(cdf);
      }
  }

  if(walk_cnt < max_walk_length){
      // signal the rest is invalid and there is no descending node
      rand_walk[i * max_walk_length + walk_cnt] = -1;
  }
}

__global__ pageRank(float* weight, int num_nodes, int iteration, float constant)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i > num_nodes)
  {
    return;
  }
  int start = start_idx[i];
  int end = start_idx[i + 1];
  // float coeff_unlinked = constant / num_nodes;
  float coeff_linked = (1.0f - constant) / (end - start);
  while (iteration-- > 0)
  {
    float sum_weight_linked = 0f;
    for (int j = start; j < end; j++)
    {
      sum_weight_linked += weight[node_idx[j]];
    }
    float new_weight = sum_weight_linked * coeff_linked + constant;
    __syncthread();
    weight[i] = new_weight;
    __syncthread();
  }
}