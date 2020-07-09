#ifndef POLYNOMIAL_H_
#define POLYNOMIAL_H_

#include <vector>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "points.h"
#include "octree_parser.h"
#include "marching_cube.h"
#include "chamfer_dist.h"

using std::vector;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using Eigen::MatrixXf;


class polynomial {
 public:

  static int num_points(OctreeParser* octree, const vector<int>& children_depth, int cstart, int cend, 
    const vector<OctreeParser::uint32>& unique_idx, const vector<OctreeParser::uint32>& sorted_idx);
  static Vector3f avg_point(OctreeParser* octree, const vector<int>& children_depth, int cstart, int cend, 
    const vector<float>& pst_scaled, const int num_points, const vector<OctreeParser::uint32>& unique_idx, const vector<OctreeParser::uint32>& sorted_idx);
  static Vector3f avg_normal(OctreeParser* octree, const vector<int>& children_depth, int cstart, int cend, 
    const float* normals, const int num_normals, const vector<OctreeParser::uint32>& unique_idx, const vector<OctreeParser::uint32>& sorted_idx);

  // -------------- MATH UTILS

  static MatrixXf calc_rotation_matrix(Vector3f norm1);
  static MatrixXf biquad(float u, float v);
  static MatrixXf triquad(Vector3f p);

  static MatrixXf biquad_approximation(const vector<float>& pts_scaled, const vector<OctreeParser::uint32>& sorted_idx, 
    int jstart, int jend, MatrixXf R, Vector3f plane_center, float support_radius);

  static MatrixXf biquad_approximation(OctreeParser* octree, const vector<int>& children_depth, int cstart, int cend, 
    const vector<float>& pts_scaled, float scale_factor, const vector<OctreeParser::uint32>& unique_idx, const vector<OctreeParser::uint32>& sorted_idx, 
    MatrixXf R, Vector3f plane_center, float support_radius);

  static float biquad_approximation_error(OctreeParser* octree, const vector<int>& children_depth, int cstart, int cend, 
    const vector<float>& pts_scaled, const int num_points, float scale_factor, const vector<OctreeParser::uint32>& unique_idx, const vector<OctreeParser::uint32>& sorted_idx, 
    MatrixXf R, Vector3f plane_center, MatrixXf coef, float support_radius);

  static float biquad_approximation_chamfer_dist(OctreeParser* octree, const vector<int>& children_depth, int cstart, int cend, 
    const vector<float>& pts_scaled, const int num_points, float scale_factor, const vector<OctreeParser::uint32>& unique_idx, const vector<OctreeParser::uint32>& sorted_idx, 
    Vector3f node_center, Vector3f plane_center, Vector3f plane_normal, MatrixXf coef, float support_radius);

  static float fval_biquad(float u, float v, MatrixXf c);
  static float fval_triquad(Vector3f p, Vector3f plane_center, MatrixXf c);
  static float taubin_distance_biquad(float u, float v, MatrixXf c);

  static Vector3f uv2xyz(Vector2f uv, Vector3f plane_center, MatrixXf R, MatrixXf c);
  static Vector3f uv2norm(Vector2f uv, Vector3f pc, MatrixXf R, MatrixXf c);
  static MatrixXf biquad2triquad(Vector3f plane_center, MatrixXf R, MatrixXf c, float range);
};


#endif // POLYNOMIAL_H_