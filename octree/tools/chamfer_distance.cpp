#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <ctime>
#include <nanoflann.hpp>

#include "points.h"
#include "filenames.h"
#include "cmd_flags.h"
#include "math_functions.h"

using namespace std;

DEFINE_string(path_a, kRequired, "", "The file path of a");
DEFINE_string(path_b, kRequired, "", "The file path of b");
DEFINE_string(filename_out, kRequired, "", "The output filename");
DEFINE_int(pt_num, kOptional, -1, "The point num");
DEFINE_string(category, kOptional, "", "The output filename");
DEFINE_int(rescale_depth, kOptional, -1, "rescale depth");


template < class VectorType, int DIM = 3, typename num_t = float,
    class Distance = nanoflann::metric_L2, typename IndexType = size_t >
class KDTreeVectorOfVectorsAdaptor {
 public:
  typedef KDTreeVectorOfVectorsAdaptor<VectorType, DIM, num_t, Distance> self_t;
  typedef typename Distance::template traits<num_t, self_t>::distance_t metric_t;
  typedef nanoflann::KDTreeSingleIndexAdaptor< metric_t, self_t, DIM, IndexType>  index_t;

  //! The kd-tree index for the user to call its methods as usual with any other FLANN index.
  index_t* index_;

 private:
  const VectorType &data_;

 public:
  // Constructor: takes a const ref to the vector of vectors object with the data points
  KDTreeVectorOfVectorsAdaptor(const VectorType &mat, const int leaf_max_sz = 10)
    : data_(mat) {
    assert(mat.info().pt_num() != 0);
    index_ = new index_t(DIM, *this, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_sz));
    index_->buildIndex();
  }

  ~KDTreeVectorOfVectorsAdaptor() {
    if (index_) delete index_;
  }

  // Query for the num_closest closest points to a given point .
  inline void query(const num_t *query_point, const size_t num_closest, IndexType *out_indices,
      num_t *out_distances_sq, const int nChecks_IGNORED = 10) const {
    nanoflann::KNNResultSet<num_t, IndexType> resultSet(num_closest);
    resultSet.init(out_indices, out_distances_sq);
    index_->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
  }

  // Query for the closest points to a given point (entered as query_point[0:dim-1]).
  inline void closest(const num_t *query_point, IndexType *out_index, num_t *out_distance_sq) const {
    query(query_point, 1, out_index, out_distance_sq);
  }

  // Interface expected by KDTreeSingleIndexAdaptor
  const self_t & derived() const { return *this; }
  self_t & derived() { return *this; }

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const {
    return data_.info().pt_num();
  }

  // Returns the j'th component of the i'th point in the class:
  inline num_t kdtree_get_pt(const size_t i, int j) const {
    return data_.ptr(PointsInfo::kPoint)[i * DIM + j];
  }

  // Return false to default to a standard bbox computation loop.
  template <class BBOX>
  bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }
};

void closet_pts(vector<size_t>& idx, vector<float>& distance,
    const Points& pts, const Points& des) {
  // build a kdtree

  KDTreeVectorOfVectorsAdaptor<Points> kdtree(des);

  // kdtree search
  size_t num = pts.info().pt_num();

  distance.resize(num);
  idx.resize(num);
  const float* pt = pts.ptr(PointsInfo::kPoint);
  #pragma omp parallel for
  for (int i = 0; i < num; ++i) {
    kdtree.closest(pt + 3 * i, idx.data() + i, distance.data() + i);
  }
}

void rescale_autoencoder_output(Points input, Points output, int depth) 
{
    float radius_ = 0;
    float center_[3] = {0,0,0};
    bounding_sphere(radius_, center_, input.points(), input.info().pt_num());

    // centralize 
    float trans[3] = { -center_[0], -center_[1], -center_[2] };

    // bounding box
    float bbmin[] = { -radius_, -radius_, -radius_ };
    float bbmax[] = { radius_, radius_, radius_ };
    float max_width = bbmax[0] - bbmin[0];
    if (max_width == 0.0f) max_width = 1.0e-10f;

    // rescale
    float mul = 1 / (float(1 << depth) / max_width);
    output.uniform_scale(mul);  

    // translate
    for ( int i = 0; i < 3; i++) {
      bbmin[i] -= trans[i];
    }
    output.translate(bbmin);
}


std::default_random_engine generator_(static_cast<unsigned int>(time(nullptr)));

void downsample(Points& point_cloud, int target_num) {
  int npt = point_cloud.info().pt_num();
  if (npt < target_num) return;

  std::bernoulli_distribution distribution_(float(target_num) / float(npt));
  vector<float> pt, normal;
  pt.reserve(npt);
  normal.reserve(npt);
  const float* ptr_pt = point_cloud.ptr(PointsInfo::kPoint);
  const float* ptr_nm = point_cloud.ptr(PointsInfo::kNormal);
  for (int i = 0; i < npt; ++i) {
    if (distribution_(generator_)) {
      for (int c = 0; c < 3; ++c) {
        pt.push_back(ptr_pt[i * 3 + c]);
        if (ptr_nm != nullptr) {
          normal.push_back(ptr_nm[i * 3 + c]);
        }
      }
    }
  }

  if (normal.size() == 0) {
    normal.assign(pt.size(), sqrtf(3.0f) / 3.0f);
  }

  point_cloud.set_points(pt, normal);
}

#if 1

int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: update_octree.exe");
    return 0;
  }

  ofstream outfile(FLAGS_filename_out, ios::app);
  if (!outfile) {
    cout << "Error: Can not open the output file:" << FLAGS_filename_out << endl;
    return 0;
  }

  vector<string> all_files_a, all_files_b;
  get_all_filenames(all_files_a, FLAGS_path_a);
  get_all_filenames(all_files_b, FLAGS_path_b);
  std::cout << "File number a: " << all_files_a.size() << std::endl;
  std::cout << "File number b: " << all_files_a.size() << std::endl;

  for (int i = 0; i < all_files_a.size(); ++i) {
    // load points
    string filename_a = all_files_a[i];
    string filename_b = all_files_b[i];
    Points point_cloud_a, point_cloud_b;

    bool loaded_a = point_cloud_a.read_points(filename_a);
    bool loaded_b = point_cloud_b.read_points(filename_b);
    if (!loaded_a || !loaded_b) {
      printf("\t\tpts empty -> skipping %d\n", i);
      continue;
    }

    if (FLAGS_rescale_depth > 0) {
      rescale_autoencoder_output(point_cloud_a, point_cloud_b, FLAGS_rescale_depth);
    }

    // random downsample points
    if (FLAGS_pt_num > 0) {
      downsample(point_cloud_a, FLAGS_pt_num);
      downsample(point_cloud_b, FLAGS_pt_num);

      point_cloud_a.write_points(FLAGS_filename_out + "_a.points");
      point_cloud_b.write_points(FLAGS_filename_out + "_b.points");
    }

    // distance from a to b
    vector<size_t> idx;
    vector<float> distance;
    closet_pts(idx, distance, point_cloud_a, point_cloud_b);


    float avg_ab = 0;
    for (auto& d : distance) { avg_ab += d; }
    avg_ab /= (float)distance.size();

    // distance from b to a
    closet_pts(idx, distance, point_cloud_b, point_cloud_a);

    float avg_ba = 0;
    for (auto& d : distance) { avg_ba += d; }
    avg_ba /= (float)distance.size();


    // output to file
    string filename = extract_filename(filename_a); // no suffix
    cout << "Processing: " + filename + "\n";
    outfile << FLAGS_category << ", " << filename << ", "
        << extract_filename(filename_b) << ", "
        << point_cloud_a.info().pt_num() << ", "
        << point_cloud_b.info().pt_num() << ", "
        << avg_ab << ", " << avg_ba << ", "
        << avg_ab + avg_ba << endl;
  }

  outfile.close();

  return 0;
}

#else

int main(int argc, char* argv[]) {

  // load points
  vector<float> feature{ 1, 1, 1 };
  vector<float> pt_src{ 2.5, 2.5, 2.5, 1, 1, 0, -1, 0, 0 };
  vector<float> pt_des{ 0, 0, 0, 1, 1, 1, 2, 2, 2 };
  Points point_cloud_a, point_cloud_b;
  point_cloud_a.set_points(pt_src, vector<float>(), feature);
  point_cloud_b.set_points(pt_des, vector<float>(), feature);

  // cloest distance
  vector<size_t> idx;
  vector<float> distance; // squared distance
  closet_pts(idx, distance, point_cloud_a, point_cloud_b);

  // average
  float avg_ab = 0;
  for (auto& d : distance) { avg_ab += d; }
  avg_ab /= (float)distance.size();

  // output
  string filename = extract_filename(FLAGS_filename_src);
  cout << filename + to_string(avg_ab) << endl;

  return 0;
}

#endif