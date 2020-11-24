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
DEFINE_string(filename_out, kRequired, "", "fname output");


int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: node_distribution.exe");
    return 0;
  }

  ofstream outfile(FLAGS_filename_out, ios::app);
  if (!outfile) {
    cout << "Error: Can not open the output file:" << FLAGS_filename_out << endl;
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
  Eigen::VectorXf num_files = Eigen::VectorXf::Zero(max_num_threads);
  Eigen::MatrixXf distrib = Eigen::MatrixXf::Zero(9, max_num_threads);

  #pragma omp parallel for
  for (int i = 0; i < all_files.size(); i++) {

    Octree octree;
    bool loaded = octree.read_octree(all_files[i]);
    if (!loaded) {
      printf("couldnt load %s /n", all_files[i].c_str());
      continue;
    }

    int depth = octree.info().depth();
    Eigen::VectorXf count = Eigen::VectorXf::Zero(depth+1);

    // count relative node distrib for single octree
    for (int d = 0; d <= depth; d++) {
      count(d) = octree.info().node_num(d);
    }
    //count /= float(count.sum()); // store relative 

    // write result into shared count
    int t = omp_get_thread_num();
    distrib.block(0, t, depth+1, 1) += count;
    num_files(t)++;
    thread_counter(t) = 1;

    // progress bar
    if (int(num_files(t)) % 100 == 0) printf("thread %d -> %d\n", t, int(num_files(t)));
  }
  Eigen::VectorXf result = distrib.rowwise().sum() / num_files.sum();

  for (int d = 0; d < distrib.rows(); d++) {
    outfile << "d=" << d << "\t";
  } outfile << std::endl;

  for (int d = 0; d < distrib.rows(); d++) {
    outfile << result(d) << "\t";
  } outfile << std::endl;

  result /= result.sum();

  for (int d = 0; d < distrib.rows(); d++) {
    outfile << result(d)*100 << "%\t";
  } outfile << std::endl;


  outfile.close();

  return 0;
}

