#include "marching_cube.h"


inline int MarchingCube::btwhere(int x) const {
  float f = (unsigned int)x;
  return ((*(unsigned int*)(&f)) >> 23) - 127;
}

inline void MarchingCube::interpolation(float* pt, const float* pt1,
    const float* pt2, const float f1, const float f2) const {
  float df = f2 - f1;
  if (df == 0) df += 1.0e-10f;
  float t = -f1 / df;
  for (int i = 0; i < 3; ++i) {
    pt[i] = pt1[i] + t * (pt2[i] - pt1[i]);
  }
}

MarchingCube::MarchingCube(const float* fval, float iso_val, const float* left_btm,
    int vid) {
  set(fval, iso_val, left_btm, vid);
}

void MarchingCube::set(const float* fval, float iso_val, const float* left_btm,
    int vid) {
  fval_ = fval;
  iso_value_ = iso_val;
  left_btm_ = left_btm;
  vtx_id_ = vid;
}

unsigned int MarchingCube::compute_cube_case() const {
  const unsigned int mask[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
  unsigned int cube_case = 0;
  for (int i = 0; i < 8; ++i) {
    if (fval_[i] < iso_value_) cube_case |= mask[i];
  }
  return cube_case;
}

void MarchingCube::contouring(vector<float>& vtx, vector<int>& face) const {
  // compute cube cases
  unsigned int cube_case = compute_cube_case();

  // generate vtx
  int vid[12], id = vtx_id_;
  int edge_mask = edge_table_[cube_case];
  while (edge_mask != 0) {
    int pos = btwhere(edge_mask & (-edge_mask));

    // set vertex id
    vid[pos] = id++;

    // calc points
    float pti[3];
    int v1 = edge_vert_[pos][0];
    int v2 = edge_vert_[pos][1];
    interpolation(pti, corner_[v1], corner_[v2], fval_[v1], fval_[v2]);
    for (int j = 0; j < 3; ++j) {
      float p = pti[j] + left_btm_[j];
      vtx.push_back(p);
    }

    edge_mask &= edge_mask - 1;
  }

  // generate triangle
  const int* tri = tri_table_[cube_case];
  for (int i = 0; i < 16; ++i) {
    if (tri[i] == -1) break;
    face.push_back(vid[tri[i]]);
  }
}


void intersect_cube(vector<float>& V, const float* pt, const float* pt_base,
    const float* normal) {
  // compute f_val
  float fval[8] = { 0 };
  for (int k = 0; k < 8; ++k) {
    for (int j = 0; j < 3; ++j) {
      fval[k] += (MarchingCube::corner_[k][j] + pt_base[j] - pt[j]) * normal[j];
    }
  }

  // marching cube
  V.clear();
  vector<int> F;
  MarchingCube mcube(fval, 0, pt_base, 0);
  mcube.contouring(V, F);
}


void marching_cube_octree(vector<float>& V, vector<int>& F, const vector<float>& pts,
    const vector<float>& pts_ref, const vector<float>& normals) {
  int num = pts.size() / 3;
  V.clear(); F.clear();
  for (int i = 0; i < num; ++i) {
    // get point and normal
    int ix3 = i * 3;
    float pt[3], pt_ref[3], normal[3];
    for (int j = 0; j < 3; ++j) {
      pt_ref[j] = pts_ref[ix3 + j];      // the reference point
      pt[j] = pts[ix3 + j] - pt_ref[j];  // the local displacement
      normal[j] = normals[ix3 + j];
    }

    // compute f_val
    float fval[8] = {0};
    for (int k = 0; k < 8; ++k) {
      for (int j = 0; j < 3; ++j) {
        fval[k] += (MarchingCube::corner_[k][j] - pt[j]) * normal[j];
      }
    }

    // marching cube
    int vid = V.size() / 3;
    MarchingCube mcube(fval, 0, pt_ref, vid);
    mcube.contouring(V, F);
  }
}


void marching_cube_octree_implicit(vector<float>& V, vector<int>& F, const vector<float>& pts,
    const vector<float>& pts_ref, const vector<float>& normals, const vector<float>& coefs, const int n_subdivision) {
  int num = pts.size() / 3;
  V.clear(); F.clear();
  for (int i = 0; i < num; ++i) {
    // get point and normal
    int ix3 = i * 3;
    float pt[3], pt_ref[3], normal[3], c[6];
    for (int j = 0; j < 3; ++j) {
      pt_ref[j] = pts_ref[ix3 + j];       // global node start point
      pt[j] = pts[ix3 + j] - pt_ref[j];   // plane center in local coordinates 
      normal[j] = normals[ix3 + j];       // plane normal
    }
    for (int j = 0; j < 6; ++j) { c[j] = coefs[i*6 + j]; }  // slim coefficients

    Eigen::Vector3f plane_center(pt[0], pt[1], pt[2]);
    Eigen::Vector3f plane_normal(normal[0], normal[1], normal[2]);
    Eigen::MatrixXf R = polynomial::calc_rotation_matrix(plane_normal);
    Eigen::MatrixXf coef(6,1);
    coef << c[0],c[1],c[2],c[3],c[4],c[5]; 

    auto coefs3 = polynomial::biquad2triquad(plane_center, R, coef, 1);

    float pt_ref_sub[3] = {0};
    float step = 1.0 / float(n_subdivision);

    Eigen::Vector3f ray_dir = {1, 0, 0};

    // compute function value for corners
    for (int x = 0; x < n_subdivision; x++) {
        pt_ref_sub[0] = x * step;
        for (int y = 0; y < n_subdivision; y++) {
            pt_ref_sub[1] = y * step;
            for (int z = 0; z < n_subdivision; z++) {
                pt_ref_sub[2] = z * step;

                Eigen::Vector3f p;
                float fval[8] = {0};
                for (int k = 0; k < 8; ++k) {
                    for (int j = 0; j < 3; ++j) {
                        p(j) = MarchingCube::corner_[k][j] * step + pt_ref_sub[j];
                    }

                    // calcualate function value
                    fval[k] = polynomial::fval_triquad(p, plane_center, coefs3);
                }

                // marching cube
                int vid = V.size() / 3;
                float ref[3] = {0,0,0};
                MarchingCube mcube(fval, 0, ref, vid);
                mcube.contouring(V, F);
                for (int i = vid; i < V.size() / 3; i++) {
                    for (int c = 0; c < 3; c++) { V[i*3+c] = V[i*3+c]*step + pt_ref_sub[c] + pt_ref[c]; }
                }
            }   
        }   
    }


  }
}
