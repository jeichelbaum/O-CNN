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
  vector<std::ostringstream> octree_stats(max_num_threads);


  #pragma omp parallel for
  for (int i = 0; i < all_files.size(); i++) {

    Octree octree;
    bool loaded = octree.read_octree(all_files[i]);
    if (!loaded) {
      printf("couldnt load %s /n", all_files[i].c_str());
      continue;
    }

    // get curvature stats
    vector<float> stats;
    octree.avg_max_curvature(stats);
    
    // write curvature to string stream
    int t = omp_get_thread_num();
    octree_stats[t] << all_files[i].c_str() << " ";
    for (int c = 0; c < stats.size(); c++) {
        octree_stats[t] << stats[c] << " ";
    }
    octree_stats[t] << std::endl;
  }


  // write all string streams to out file
  for (int d = 0; d < max_num_threads; d++) {
    outfile << octree_stats[d].str();
  }
  outfile.close();

  return 0;
}

