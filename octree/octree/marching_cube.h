#ifndef MARCHIING_CUBE_H_
#define MARCHIING_CUBE_H_

#include <vector>
#include <eigen3/Eigen/Dense>

#include "polynomial.h"

using std::vector;

class MarchingCube {
 public:
  MarchingCube() : fval_(nullptr), iso_value_(0), left_btm_(nullptr), vtx_id_(0) {}
  MarchingCube(const float* fval, float iso_val, const float* left_btm, int vid);
  void set(const float* fval, float iso_val, const float* left_btm, int vid);
  void contouring(vector<float>& vtx, vector<int>& face) const;
  unsigned int compute_cube_case() const;

 private:
  inline int btwhere(int x) const;
  inline void interpolation(float* pt, const float* pt1, const float* pt2,
      const float f1, const float f2) const;


 protected:
  const float* fval_;
  float iso_value_;
  const float* left_btm_;
  int vtx_id_;

 public:
  static const int edge_table_[256];
  static const int tri_table_[256][16];
  static const int edge_vert_[12][2];
  static const float corner_[8][3];
};


// two convenient interfaces
void marching_cube_octree(vector<float>& V, vector<int>& F, const vector<float>& pts,
    const vector<float>& pts_ref, const vector<float>& normals);

// sample multiple surfels at same octree depth
void triquad_marchingcube(vector<float>& V, vector<int>& F, const vector<float>& cell_bases, float cell_size, 
    const vector<float>& surfel_centers, const vector<float>& surfel_coefs, const int n_subdivision);
        
void marching_cube_octree_implicit(vector<float>& V, vector<int>& F, const vector<float>& pts,
const vector<float>& pts_ref, const vector<float>& normals, const vector<float>& coefs, const int n_subdivision);
void intersect_cube(vector<float>& V, const float* pt, const float* pt_base,
    const float* normal);

#endif // MARCHIING_CUBE_H_