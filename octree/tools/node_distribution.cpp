#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <omp.h>

#include "cmd_flags.h"
#include "octree.h"
#include "filenames.h"
#include "math_functions.h"

#include "eigen3/Eigen/Dense"

using std::vector;
using std::string;
using std::cout;
using std::endl;
using cflags::Require;
const float kPI = 3.14159265f;

DEFINE_string(filenames, kRequired, "", "The octree list");


int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: node_distribution.exe");
    return 0;
  }



  // category path
  string list_path = FLAGS_filenames;
  vector<string> all_files;
  get_all_filenames(all_files, list_path);

  printf("number octree %d\n", all_files.size());



  // counter number of threads in use
  int max_num_threads = 32;
  Eigen::VectorXf thread_counter = VectorXf::Zero(max_num_threads);

  // depth X num threads matrix
  Eigen::VectorXf num_files = Eigen::VectorXf::Ones(max_num_threads);
  Eigen::MatrixXf distrib = Eigen::MatrixXf::Zero(9, max_num_threads);

  #pragma omp parallel for
  for (int i = 0; i < all_files.size(); i++) {

    Octree octree;
    octree.read_octree(all_files[i]);
    
    int depth = octree.info().depth();
    Eigen::VectorXf count = Eigen::VectorXf::Zero(depth+1);

    //std::cout << "d: " << depth << " -> " << all_files[i] << std::endl;

    // count relative node distrib for single octree
    for (int d = 0; d <= depth; d++) {
        count(d) = octree.info().node_num(d);
    }
    count /= float(count.sum()); // add relative 

    // write result into shared count
    int t = omp_get_thread_num();
    distrib.block(0, t, depth+1, 1) += count;
    num_files(t)++;
    thread_counter(t) = 1;

    if (int(num_files(t)) % 100 == 0) printf("thread %d -> %d\n", t, int(num_files(t)));
    
  }

  int num_threads_used = thread_counter.sum();

  // divide distrib by number of files processed
  for (int r = 0; r < distrib.rows(); r++) {
    distrib.block(r, 0, 1, max_num_threads) = distrib.row(r).array() / num_files.transpose().array();
  }

  // divide by number of threads - doesnt sum to one because init num_fules with ones
  Eigen::VectorXf result = distrib.rowwise().sum() / num_threads_used;
  printf("sanity check: percentages sum to %f\n", result.sum());
  result /= result.sum(); 
  printf("post normalize: percentages sum to %f\n\n", result.sum());


  printf("relative number of nodes distribution accress depth levels:\n");
  for (int d = 0; d < distrib.rows(); d++) {
    printf("depth %d: %f\n", d, result(d));
  }


  return 0;
}

