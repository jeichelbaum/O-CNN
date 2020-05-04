#ifndef POLYNOMIAL_H_
#define POLYNOMIAL_H_

#include <vector>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "octree_parser.h"

using std::vector;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using Eigen::MatrixXf;


class polynomial {
 public:

  static MatrixXf calc_rotation_matrix(Vector3f norm1);
  static MatrixXf biquad(float u, float v);
  static MatrixXf triquad(Vector3f p);

  static MatrixXf biquad_approximation(const vector<float>& pts_scaled, const vector<OctreeParser::uint32>& sorted_idx, 
    int jstart, int jend, MatrixXf R, Vector3f plane_center);

  static float fval_biquad(float u, float v, MatrixXf c);
  static float fval_triquad(Vector3f p, Vector3f plane_center, MatrixXf c);

  static Vector3f uv2xyz(Vector2f uv, Vector3f plane_center, MatrixXf R, MatrixXf c);
  static Vector3f uv2norm(Vector2f uv, Vector3f pc, MatrixXf R, MatrixXf c);
  static MatrixXf biquad2triquad(Vector3f plane_center, MatrixXf R, MatrixXf c, float range);
};


#endif // POLYNOMIAL_H_