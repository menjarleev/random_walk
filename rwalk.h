/*
 * Temporal random walk.
 */

#include <mutex>
std::mutex m_screen;
#include "rwalk.cuh"
#include <limits.h>
#include <assert.h>

int64_t *start_idx_host;
int64_t *node_idx_host;
float *timestamp_host;
int64_t *random_walk_host;

// int64_t * p_scan_list;
// int64_t * v_list;
// float * w_list;

// long int num_of_nodes = 0;
// long long int num_of_edges = 0;

// NodeID *global_walk;

/*
  Helper function to sort two temporal edges (TNode)
  in the increasing order of their timestamps
*/
bool SortByTime(TNode n1, TNode n2)
{
  return (n1.second < n2.second);
}

/*
  Filter a node's edges that have timestamps larger
  than the incoming edge (i.e., src_time)
*/
TempNodeVector FilterEdgesPostTime(
  const WGraph &g,
  NodeID src_node,
  WeightT src_time)
{
  TempNodeVector filtered_edges;
  for(auto v : g.out_neigh(src_node)) {
    if(v.w > src_time)
      filtered_edges.push_back(std::make_pair(v, v.w));
  }
  // Sort edges by timestamp?
  //std::sort(filtered_edges.begin(), filtered_edges.end(), SortByTime);
  return filtered_edges;
}

/*
  Random number generator
*/
double RandomNumberGenerator()
{
    static std::uniform_real_distribution<double> uid(0,1);
    return uid(rng);
}

/*
  Function that returns a random neighbor to
  walk based on the probability distribution
*/
int FindNeighborIdx(DoubleVector prob_dist)
{
  /*
  Algorithm: get a random number between 0 and 1
  calculate the CDF of probTimestamps and compare
  against random number
  */
  double curCDF = 0, nextCDF = 0;
  int cnt = 0;
  double random_number = RandomNumberGenerator();
  for(auto it : prob_dist) {
      nextCDF += it;
      if(nextCDF >= random_number && curCDF <= random_number) {
          return cnt;
      }  else {
          curCDF = nextCDF;
          cnt++;
    }
  }
  // Ideally, it should never hit this point
  // The only time when it hits this is when the timestamps of all
  //     outgoing edges are same, in which case, just randomly
  //     selecting one edge out of all (as done below) should work.
  return rand() % prob_dist.size();
}

/*
  Function to compute the immediate next neighbor to walk
  Builds a probability distribution based on the time difference
  of incoming edge and outgoing edges
*/
bool GetNeighborToWalk(
  const WGraph &g,
  NodeID src_node,
  WeightT src_time,
  TNode& next_neighbor)
{
  int neighborhood_size = g.out_degree(src_node);
  if(neighborhood_size == 0) {
    return false;
  } else {
    TempNodeVector filtered_edges = FilterEdgesPostTime(g, src_node, src_time);
    if(filtered_edges.empty()) {
      return false;
    }
    if(filtered_edges.size() == 1) {
      next_neighbor = filtered_edges[0];
      return true;
    } else {
      DoubleVector prob_dist;
      WeightT time_boundary_diff;
      time_boundary_diff = g.TimeBoundsDelta(src_node);
      if(time_boundary_diff == 0)
      {
        next_neighbor = filtered_edges[rand() % filtered_edges.size()];
        return true;
      } else {
        // TODO: parallelism?
        for(auto it : filtered_edges) {
          prob_dist.push_back(exp((float)(it.second-src_time)/time_boundary_diff));
        }
        double exp_sum = std::accumulate(prob_dist.begin(), prob_dist.end(), 0.0);
        for (uint32_t i = 0; i < prob_dist.size(); ++i) {
          prob_dist[i] = prob_dist[i] / exp_sum;
        }
        int neighbor_index = FindNeighborIdx(prob_dist);
        next_neighbor = filtered_edges[neighbor_index];
        return true;
      }
    }
  }
}

/*
  Function to compute a random walk from a node.
  It recursively calls itself until either the maximum
  walk count has reached or if a node has no latent outgoing edges.
  GetNeighborToWalk() function is used find the immediate
  next neighbor to walk.
*/
bool compute_walk_from_a_node(
  const WGraph& g,
  NodeID src_node,
  WeightT prev_time_stamp,
  TNode& next_neighbor_ret,
  int max_walk_length,
  NodeID *local_array,
  int32_t pos)
{
  TNode next_neighbor;
  if(g.out_degree(src_node) != 0 &&
    GetNeighborToWalk(g, src_node, prev_time_stamp, next_neighbor)) {
    local_array[pos] = next_neighbor.first;
    next_neighbor_ret = next_neighbor;
    return true;
  }
  return false;
}

/*
  Function to return timestamp to initiate a random walk
  from a source node.
  Currently it always returns 0, can be modified in the future.
*/
WeightT GetInitialTime(const WGraph &g, NodeID src_node)
{
  // TODO: modify this function later
  return (WeightT) 0;
}

/*
  Write random walk to a file
*/
void WriteWalkToAFile(
  NodeID* global_walk,
  int num_nodes,
  int max_walk,
  int num_walks_per_node,
  std::string walk_filename)
{
  std::ofstream random_walk_file(walk_filename);
  for(int w_n = 0; w_n < num_walks_per_node; ++w_n) {
    for(NodeID iter = 0; iter < num_nodes; iter++) {
      NodeID *local_walk =
        global_walk +
        ( iter * max_walk * num_walks_per_node ) +
        ( w_n * max_walk );
      for (int i = 0; i < max_walk; i++) {
          if (local_walk[i] == -1)
            break;
          random_walk_file << local_walk[i] << " ";
      }
      random_walk_file << "\n";
    }
  }
  random_walk_file.close();
}

/*
  Function that iterates over all vertices in graph,
  and calls compute_walk_from_a_node() function.
  Each random walk from a node is stored in local_walk,
  which is pushed to global_walk that stores all random walks
*/
void compute_random_walk(
  const WGraph &g,
  int max_walk_length,
  int num_walks_per_node,
  std::string walk_filename) {
  std::cout << "Computing random walk for " << g.num_nodes() << " nodes and "
      << g.num_edges() << " edges." << std::endl;
  max_walk_length++;
  NodeID *global_walk = new NodeID[g.num_nodes() * max_walk_length * num_walks_per_node];
  Timer t;
  t.Start();
  for(int w_n = 0; w_n < num_walks_per_node; ++w_n) {
    std::cout << "walk number: " << w_n << std::endl;
    for(NodeID i = 0; i < g.num_nodes(); ++i) {
      NodeID *local_walk =
        global_walk +
        ( i * max_walk_length * num_walks_per_node ) +
        ( w_n * max_walk_length );
      local_walk[0] = i;
      WeightT prev_time_stamp = 0;
      NodeID next_neighbor = i;
      TNode next_neighbor_ret;
      int walk_cnt;
      for(walk_cnt = 1; walk_cnt < max_walk_length; ++walk_cnt) {
        bool cont = compute_walk_from_a_node(
          g,
          next_neighbor,
          prev_time_stamp,
          next_neighbor_ret,
          max_walk_length,
          local_walk,
          walk_cnt
        );
        if(!cont) break;
        next_neighbor = next_neighbor_ret.first;
        prev_time_stamp = next_neighbor_ret.second;
      }
      if (walk_cnt != max_walk_length)
          local_walk[walk_cnt] = -1;
    }
  }
  t.Stop();
  PrintStep("[TimingStat] Random walk time (s):", t.Seconds());
  WriteWalkToAFile(global_walk, g.num_nodes(),
    max_walk_length, num_walks_per_node, walk_filename);
  delete[] global_walk;
}


// num_of_node, num_of_walk, max_walk_length,
// node_idx : |E| array, node index -> int64_t*
// timestamp : |E| array -> float*
// start_idx : |V| + 1array, for starting index in the above two
//              arrays of each node -> int64_t*
void GPU_random_walk(
    const WGraph &g,
    int max_walk_length,
    int num_walks_per_node,
    std::string walk_filename)
{
    // flatten the graph into 2 arrays
    // pair<> not supported in GPU
    int64_t num_nodes = g.num_nodes();
    int64_t num_edges = g.num_edges();
    // printf("num_nodes : %d ; num_edges : %d； num_walks_per_node: %d; max_walk_length: %d\n", num_nodes, num_edges, num_walks_per_node, max_walk_length);

    // define the array
    start_idx_host = new int64_t[num_nodes + 1];
    node_idx_host = new int64_t[num_edges];
    timestamp_host = new float[num_edges];
    max_walk_length++;
    random_walk_host = new int64_t[num_nodes * max_walk_length * num_walks_per_node];

    //prepare input data
    // flatten graph into arrays
    start_idx_host[0] = 0;

    for (NodeID i = 0; i < num_nodes; i++)
    {
        int64_t out_edge_number = g.out_degree(i);
        start_idx_host[i + 1] = start_idx_host[i] + out_edge_number;
        int64_t j = 0;
        for (auto out_node : g.out_neigh(i))
        {
            int64_t idx = start_idx_host[i] + j;
            node_idx_host[idx] = out_node.v;
            timestamp_host[idx] = out_node.w;
            j++;
        }
        assert(j == out_edge_number);
    }

    Timer t;
    t.Start();
    cuda_rwalk(max_walk_length, num_walks_per_node, num_nodes, num_edges, (unsigned long long) (RandomNumberGenerator() * 1.0 * ULLONG_MAX));
    t.Stop();

    PrintStep("[TimingStat] Random walk time (s):", t.Seconds());
    WriteWalkToAFile(random_walk_host, num_nodes, max_walk_length, num_walks_per_node, walk_filename);

    delete[] start_idx_host;
    delete[] node_idx_host;
    delete[] timestamp_host;
    delete[] random_walk_host;
}

// void GPU_random_walk_original(
//   const WGraph &g,
//   int max_walk_length,
//   int num_walks_per_node,
//   std::string walk_filename) {

//   num_of_nodes = g.num_nodes();
//   num_of_edges = g.num_edges();

//   p_scan_list = new int64_t [num_of_nodes + 1];
//   p_scan_list[0] = 0;
//   v_list = new int64_t[num_of_edges];
//   w_list = new float[num_of_edges];

//   std::cout << "Generating parallel scan list and data structures for GPU..." << std::endl;
//   for(NodeID i = 0; i < num_of_nodes; ++i) {
//     p_scan_list[i + 1] = p_scan_list[i] + g.out_degree(i);
//     int cnt = 0;
//     for(auto v: g.out_neigh(i)){
//       int64_t idx = p_scan_list[i] + cnt;
//       v_list[idx] = v.v;
//       w_list[idx] = v.w;
//       cnt++;
//     }
//   }

//   std::cout << "Computing random walk for " << g.num_nodes() << " nodes and "
//       << g.num_edges() << " edges." << std::endl;
//   max_walk_length++;
//   global_walk = new NodeID[g.num_nodes() * max_walk_length * num_walks_per_node];
//   // NodeID * host_global_walk = new NodeID[g.num_nodes() * max_walk_length * num_walks_per_node];

//   Timer t;
//   t.Start();
//   initializeGPU_rwalk(max_walk_length, num_walks_per_node);
//   TrainGPU_rwalk(max_walk_length, num_walks_per_node, (unsigned long long) (RandomNumberGenerator() * 1.0 * ULLONG_MAX));

//   cleanUpGPU_rwalk();
//   t.Stop();
//   PrintStep("[TimingStat] Random walk time (s):", t.Seconds());
//   WriteWalkToAFile(global_walk, g.num_nodes(), max_walk_length, num_walks_per_node, walk_filename);
//   delete[] global_walk;

//   delete[] p_scan_list;
//   delete[] v_list;
//   delete[] w_list;
// }