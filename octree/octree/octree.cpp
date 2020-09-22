#include "octree.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <queue>
#include <random>
#include <ctime>
#include <cstring>
#include <cmath>

#include "math_functions.h"
#include "octree_nn.h"
#include "marching_cube.h"


void Octree::set_octree(const Octree& octree_in) {
  buffer_ = octree_in.buffer();
  this->set_cpu(buffer_.data());
}

void Octree::set_octree(vector<char>& data) {
  buffer_.swap(data);
  this->set_cpu(buffer_.data());
}

void Octree::set_octree(const char* data, const int sz) {
  resize_octree(sz);
  memcpy(buffer_.data(), data, sz);
  this->set_cpu(buffer_.data());
}

void Octree::resize_octree(const int sz) {
  buffer_.resize(sz);
  this->set_cpu(buffer_.data());
}

bool Octree::read_octree(const string& filename) {
  std::ifstream infile(filename, std::ios::binary);
  if (!infile) return false;

  infile.seekg(0, infile.end);
  size_t len = infile.tellg();
  infile.seekg(0, infile.beg);
  if (len < sizeof(OctreeInfo)) {
    // the file should at least contain a OctreeInfo structure
    infile.close();
    return false;
  }

  buffer_.resize(len);
  infile.read(buffer_.data(), len);
  this->set_cpu(buffer_.data());

  infile.close();
  return true;
}

bool Octree::write_octree(const string& filename) const {
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) return false;
  outfile.write(buffer_.data(), buffer_.size());
  outfile.close();
  return true;
}

std::string Octree::get_binary_string() const {
  return std::string(buffer_.cbegin(), buffer_.cend());
}

void Octree::build(const OctreeInfo& octree_info, Points& point_cloud) {
  auto start = system_clock::now();

  // init
  clear(octree_info.depth());
  oct_info_ = octree_info;
  info_ = &oct_info_;

  // preprocess, get key and sort
  vector<float> pts_scaled;
  normalize_pts(pts_scaled, point_cloud);
  vector<uintk> node_keys, sorted_idx;
  sort_keys(node_keys, sorted_idx, pts_scaled);
  vector<uintk> unique_idx;
  unique_key(node_keys, unique_idx);

  // build octree structure
  build_structure(node_keys);

  // set nnum_[], nnum_cum_[], nnum_nempty_[] and ptr_dis_[]
  calc_node_num();

  covered_depth_nodes(); // weird displace but also computes idx_d needed for point attribution

  // average the signal for the last octree layer
  calc_signal_implicit(point_cloud, pts_scaled, sorted_idx, unique_idx);
  avg_signal_implicit(point_cloud, pts_scaled, sorted_idx, unique_idx);

  //read_signal("/home/jeri/dev/implicit_ocnn/testwrite.exoct");

  // average the signal for the other octree layers
  if (oct_info_.locations(OctreeInfo::kFeature) == -1) {
    covered_depth_nodes();

    /*bool has_normal = point_cloud.info().has_property(PointsInfo::kNormal);
    bool calc_norm_err = oct_info_.is_adaptive() && has_normal;
    bool calc_dist_err = oct_info_.is_adaptive() && oct_info_.has_displace() && has_normal;
    calc_signal(calc_norm_err, calc_dist_err, pts_scaled, sorted_idx, unique_idx);*/
  }

  // generate split label
  if (oct_info_.has_property(OctreeInfo::kSplit)) {
    calc_split_label();
  }

  // extrapolate node feature
  /*if (oct_info_.extrapolate() && oct_info_.locations(OctreeInfo::kFeature) == -1) {
    extrapolate_signal();
  }*/

  // serialization
  serialize();

  trim_octree();

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start).count(); 
  printf("total time %f\n", float(duration) / 1000000.0);
}

void Octree::clear(int depth) {
  keys_.clear();
  children_.clear();
  displacement_.clear();
  split_labels_.clear();
  avg_normals_.clear();
  avg_features_.clear();
  avg_pts_.clear();
  avg_labels_.clear();
  max_label_ = 0;
  buffer_.clear();
  info_ = nullptr;
  dnum_.clear();
  didx_.clear();
  normal_err_.clear();
  distance_err_.clear();

  if (depth == 0) return;
  keys_.resize(depth + 1);
  children_.resize(depth + 1);
  displacement_.resize(depth + 1);
  split_labels_.resize(depth + 1);
  avg_normals_.resize(depth + 1);
  avg_features_.resize(depth + 1);
  avg_pts_.resize(depth + 1);
  avg_labels_.resize(depth + 1);
  dnum_.resize(depth + 1);
  didx_.resize(depth + 1);
  normal_err_.resize(depth + 1);
  distance_err_.resize(depth + 1);
}

void Octree::normalize_pts(vector<float>& pts_scaled, const Points& point_cloud) {
  const float* bbmin = oct_info_.bbmin();
  const float* pts = point_cloud.ptr(PointsInfo::kPoint);
  const int npt = point_cloud.info().pt_num();
  const float mul = float(1 << oct_info_.depth()) / oct_info_.bbox_max_width();
  pts_scaled.resize(3 * npt);

  // normalize the points into the range [0, 1 << depth_) using bbox_width
  //#pragma omp parallel for
  for (int i = 0; i < npt; i++) {
    int i3 = i * 3;
    for (int j = 0; j < 3; j++) {
      pts_scaled[i3 + j] = (pts[i3 + j] - bbmin[j]) * mul;
    }
  }
}

void Octree::sort_keys(vector<uintk>& sorted_keys, vector<uintk>& sorted_idx,
    const vector<float>& pts_scaled) {
  int depth_ = oct_info_.depth();
  int npt = pts_scaled.size() / 3;
  vector<std::pair<uintk, uintk>> code;
  code.reserve(npt);

  //#pragma omp parallel for
  for (int i = 0; i < npt; i++) {
    // compute key
    uintk pt[3], key;
    for (int j = 0; j < 3; ++j) {
      pt[j] = static_cast<uintk>(pts_scaled[3 * i + j]);
    }
    compute_key(key, pt, depth_);

    // generate code
    code.push_back(std::make_pair(key, i));
  }

  // sort all the code
  std::sort(code.begin(), code.end()); // will be sorted by generated key

  // unpack the code
  sorted_keys.resize(npt);
  sorted_idx.resize(npt);
  //#pragma omp parallel for
  for (int i = 0; i < npt; i++) {
    sorted_keys[i] = code[i].first;
    sorted_idx[i] = code[i].second;
  }
}

void Octree::build_structure(vector<uintk>& node_keys) {
  const int depth_ = oct_info_.depth();
  const int full_layer_ = oct_info_.full_layer();
  children_.resize(depth_ + 1);
  keys_.resize(depth_ + 1);

  // layer 0 to full_layer_: the octree is full in these layers
  for (int curr_depth = 0; curr_depth <= full_layer_; curr_depth++) {
    vector<int>& children = children_[curr_depth];
    vector<uintk>& keys = keys_[curr_depth];

    int n = 1 << 3 * curr_depth;
    keys.resize(n, -1); children.resize(n, -1);
    for (int i = 0; i < n; i++) {
      keys[i] = i;
      if (curr_depth != full_layer_) {
        children[i] = i;
      }
    }
  }

  // layer depth_ to full_layer_
  for (int curr_depth = depth_; curr_depth > full_layer_; --curr_depth) {
    // compute parent key, i.e. keys of layer (curr_depth -1)
    int n = node_keys.size();
    vector<uintk> parent_keys(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      parent_keys[i] = node_keys[i] >> 3;
    }

    // compute unique parent key
    vector<uintk> parent_pidx;
    unique_key(parent_keys, parent_pidx);

    // augment children keys and create nodes
    int np = parent_keys.size();
    int nch = np << 3;
    vector<int>& children = children_[curr_depth];
    vector<uintk>& keys = keys_[curr_depth];
    children.resize(nch, -1);
    keys.resize(nch, 0);

    for (int i = 0; i < nch; i++) {
      int j = i >> 3;
      keys[i] = (parent_keys[j] << 3) | (i % 8);
    }

    // compute base address for each node
    vector<uintk> addr(nch);
    for (int i = 0; i < np; i++) {
      for (uintk j = parent_pidx[i]; j < parent_pidx[i + 1]; j++) {
        addr[j] = i << 3;
      }
    }

    // set children pointer and parent pointer
    //#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      // address
      uintk k = (node_keys[i] & 7u) | addr[i];

      // set children pointer for layer curr_depth
      children[k] = i;
    }

    // save data and prepare for the following iteration
    node_keys.swap(parent_keys);
  }

  // set the children for the layer full_layer_
  // Now the node_keys are the key for full_layer
  if (depth_ > full_layer_) {
    for (int i = 0; i < node_keys.size(); i++) {
      uintk j = node_keys[i];
      children_[full_layer_][j] = i;
    }
  }
}

void Octree::calc_node_num() {
  const int depth = oct_info_.depth();

  vector<int> node_num(depth + 1, 0);
  for (int d = 0; d <= depth; ++d) {
    node_num[d] = keys_[d].size();
  }

  vector<int> node_num_nempty(depth + 1, 0);
  for (int d = 0; d <= depth; ++d) {
    // find the last element which is not equal to -1
    const vector<int>& children_d = children_[d];
    for (int i = node_num[d] - 1; i >= 0; i--) {
      if (children_d[i] != -1) {
        node_num_nempty[d] = children_d[i] + 1;
        break;
      }
    }
  }

  oct_info_.set_nnum(node_num.data());
  oct_info_.set_nempty(node_num_nempty.data());
  oct_info_.set_nnum_cum();
  oct_info_.set_ptr_dis(); // !!! note: call this function to update the ptr
}

int Octree::get_key_index(const vector<uint32>& key_d, uint32 key) {
  for (int i = 0; i < key_d.size(); i++) {
    if (key_d[i] == key) {
      return i;
    }
  }
  return -1;
}



// compute the average signal for the last octree layer
void Octree::calc_signal_implicit(Points& point_cloud, const vector<float>& pts_scaled,
    const vector<uint32>& sorted_idx, const vector<uint32>& unique_idx) {

  // rescale points to octree bounding box
  const float* bbmin = oct_info_.bbmin();
  const float mul = float(1 << oct_info_.depth()) / oct_info_.bbox_max_width();

  // construct poly helper
  Polynomial2Approx helper(point_cloud, bbmin, mul);
  helper.init_parent_approx_tracking(oct_info_.depth());

  // ----------------------

  float ERROR_THRESHOLD = oct_info_.threshold_distance();
   
  const int depth_max = oct_info_.depth();
  const int depth_adp = oct_info_.adaptive_layer();
  const int nnum_depth = oct_info_.node_num(depth_max);
  const float imul = 2.0f / sqrtf(3.0f);
  const vector<int>& children_depth = children_[depth_max];
  const float* pts_normals = point_cloud.ptr(PointsInfo::kNormal);  // hard coded channel sizes

  const int channel_normal = 10;
 
  // allocate array mem
  normal_err_[depth_max].resize(nnum_depth, 1.0e20f);
  distance_err_[depth_max].resize(nnum_depth, 1.0e20f);

  // iterate over each depth layer
  for (int d = depth_adp; d <= depth_max; d++) {
    // number of nodes and point indices on depth
    const int nnum_d = oct_info_.node_num(d);
    const vector<int>& dnum_d = dnum_[d];
    const vector<int>& didx_d = didx_[d];

    const vector<int>& children = children_[d];
    const vector<uint32>& key_d = keys_[d];
    const float scale = static_cast<float>(1 << (depth_max - d));

    // data arrays for current depth layer
    const vector<int>& children_d = children_[d];
    vector<float>& normal_d = avg_normals_[d];
    vector<float>& pt_d = avg_pts_[d];
    vector<float>& displacement_d = displacement_[d];
    vector<float>& normal_err_d = normal_err_[d];
    vector<float>& distance_err_d = distance_err_[d];

    // allocate memory for data arrays
    normal_d.assign(nnum_d * channel_normal, 0.0f);
    normal_err_d.assign(nnum_d, 1.0e20f);   // !!! initialized
    distance_err_d.assign(nnum_d, 1.0e20f);   // !!! as 1.0e20f
    float radius = sqrtf(0.75*scale*scale) * (1.0 + overlap_amount());   // equal to half cube diagonal

    // iterate over all nodes at current depth
    for (int i = 0; i < nnum_d; ++i) {

      // leaf nodes are empty nodes
      // Note: some nodes wont be considered, because they have no points in cell, but their support radius is big enough to capture points from other cells
      if (node_type(children_d[i]) == kLeaf) {
        continue;
      }

      // ----------- OCTREE POINT IN RADIUS QUERY ----------------
      // node key to xyz index
      uint32 ptu_base[3];
      compute_pt(ptu_base, key_d[i], d);
      int xyz[] = {ptu_base[0], ptu_base[1], ptu_base[2] };

      // check if parent already well approximated already
      if (d > 3 && helper.parent_well_approximated(d, xyz)) {
          continue;
      }

      // approximate surface and measure error
      Vector3f cell_base = { xyz[0] * scale, xyz[1] * scale, xyz[2] * scale };
      Vector3f cell_center = cell_base + 0.5*Vector3f(scale, scale, scale);
      bool well_approx = helper.approx_surface(cell_base, scale, radius, ERROR_THRESHOLD);

      // treat as well approx, if next octants would have too few points for approximation
      if (helper.npt <= helper.THRESHOLD_MIN_NUM_POINTS*8) {
        well_approx = true;
        helper.error_avg_points_surface_dist = helper.error_max_surface_points_dist = 0.0;
      }

      // -------------- STORE RESULTS -----------------------
      if (helper.npt >= helper.THRESHOLD_MIN_NUM_POINTS) {
        // store surface coefficients in normal (3-9)
        for (int c = 0; c < 10; c++) {
          normal_d[c * nnum_d + i] = helper.surf_coefs(c, 0);
        }

        // -------------- ERROR -----------------------
        if (d < depth_max) {
          float max_dist = max(helper.error_avg_points_surface_dist, helper.error_max_surface_points_dist);
          normal_err_d[i] = max_dist;
          distance_err_d[i] = max_dist;

          // store values if well approximated
          if (well_approx) {
            helper.set_well_approximated(d, xyz);
          } 
        }
      }

    }
  }
}


// compute the average signal for the last octree layer
void Octree::avg_signal_implicit(Points& point_cloud, const vector<float>& pts_scaled,
    const vector<uint32>& sorted_idx, const vector<uint32>& unique_idx) {

  float ERROR_THRESHOLD = oct_info_.threshold_distance();
   
  const int depth_max = oct_info_.depth();
  const int depth_adp = oct_info_.adaptive_layer();

  const int channel_normal = 10;

  // iterate over each depth layer
  for (int d = depth_adp-1; d >= 0; d--) {
    // number of nodes and point indices on depth
    const int nnum_d = oct_info_.node_num(d);

    const vector<int>& children = children_[d];
    const vector<uint32>& key_d = keys_[d];

    // data arrays for current depth layer
    const vector<int>& children_d = children_[d];
    vector<float>& normal_d = avg_normals_[d];
    vector<float>& normal_err_d = normal_err_[d];
    vector<float>& distance_err_d = distance_err_[d];

    // allocate memory for data arrays
    normal_d.assign(nnum_d * channel_normal, 0.0f);
    normal_err_d.assign(nnum_d, 1.0e20f);   // !!! initialized
    distance_err_d.assign(nnum_d, 1.0e20f);   // !!! as 1.0e20f

    // iterate over all nodes at current depth
    for (int i = 0; i < nnum_d; ++i) {

      // average features of non empty children into parent nodes
      if (node_type(children_d[i]) != kLeaf) {
        // compute parent xyz
        uint32 ptu_base[3];
        compute_pt(ptu_base, key_d[i], d);

        // iterate over children
        int num_non_empty_children = 0;
        for (int a = 0; a < oct_info_.node_num(d+1); a++) {
          uint32 ptu_base2[3];
          compute_pt(ptu_base2, keys_[d+1][a], d+1);

          if (node_type(children_[d+1][a]) != kLeaf && ptu_base[0] == ptu_base2[0] / 2 && 
                ptu_base[1] == ptu_base2[1] / 2 && ptu_base[2] == ptu_base2[2]/2) 
          {
            num_non_empty_children++;

            for (int c = 0; c < channel_normal; c++) {
              // store surface in normal (0-3)
              normal_d[c * nnum_d + i] = avg_normals_[d+1][c * nnum_d + a];
            }
          }
        }

        // average all copied
        for (int c = 0; c < channel_normal; c++) {
          normal_d[c * nnum_d + i] /= num_non_empty_children;
        }

        normal_err_d[i] = ERROR_THRESHOLD + 1.0;
        distance_err_d[i] = ERROR_THRESHOLD + 1.0;
      }


    }
  }
}


void Octree::extrapolate_signal() {
  const int depth = oct_info_.depth();
  const int full_layer = oct_info_.full_layer();
  const int nnum_depth = oct_info_.node_num(depth);
  const float imul = 2.0f / sqrtf(3.0f);

  const int channel_normal = avg_normals_[depth].size() / nnum_depth;
  const int channel_feature = avg_features_[depth].size() / nnum_depth;
  const int channel_label = avg_labels_[depth].size() / nnum_depth;

  const bool has_dis = !displacement_[depth].empty();
  const bool has_normal = !avg_normals_[depth].empty();
  const bool has_feature = !avg_features_[depth].empty();
  const bool has_label = !avg_labels_[depth].empty();

  for (int d = depth; d >= full_layer; --d) {
    const int nnum_d = oct_info_.node_num(d);
    const vector<int>& children_d = children_[d];
    vector<float>& normal_d = avg_normals_[d];
    vector<float>& label_d = avg_labels_[d];
    vector<float>& feature_d = avg_features_[d];
    vector<float>& displacement_d = displacement_[d];

    for (int i = 0; i < nnum_d; ++i) {
      if (node_type(children_d[i]) != kLeaf) continue;
      int id = i % 8;
      int i_base = i - id;

      float count = ESP; // the non-empty node number
      for (int j = i_base; j < i_base + 8; ++j) {
        if (node_type(children_d[j]) != kLeaf) count += 1.0f;
      }

      vector<float> n_avg(channel_normal, 0.0f);
      if (has_normal) {
        for (int j = i_base; j < i_base + 8; ++j) {
          if (node_type(children_d[j]) == kLeaf) continue;
          for (int c = 0; c < channel_normal; ++c) {
            n_avg[c] += normal_d[c * nnum_d + j];
          }
        }

        float ilen = 1.0f / (norm2(n_avg) + ESP);
        for (int c = 0; c < channel_normal; ++c) {
          n_avg[c] *= ilen;
          normal_d[c * nnum_d + i] = n_avg[c];  // output
        }
      }

      vector<float> f_avg(channel_feature, 0.0f);
      if (has_feature) {
        for (int j = i_base; j < i_base + 8; ++j) {
          if (node_type(children_d[j]) == kLeaf) continue;
          for (int c = 0; c < channel_feature; ++c) {
            f_avg[c] += feature_d[c * nnum_d + j];
          }
        }

        for (int c = 0; c < channel_feature; ++c) {
          f_avg[c] /= count;
          feature_d[c * nnum_d + i] = f_avg[c]; // output
        }
      }

      vector<int> l_avg(max_label_, 0);
      if (has_label) {
        int valid_num = 0;
        for (int j = i_base; j < i_base + 8; ++j) {
          int l = static_cast<int>(label_d[j]);
          if (l < 0) { continue; }  // invalid labels
          l_avg[l] += 1;
          valid_num += 1;
        }
        if (valid_num > 0) {
          label_d[i] = static_cast<float>(std::distance(l_avg.begin(),
                      std::max_element(l_avg.begin(), l_avg.end())));
        }
      }

      if (has_dis && count > 0.5f) {
        float xyzs[8][3] = {
          {0, 0, 0}, {0, 0, 1.0f}, {0, 1.0f, 0}, {0, 1.0f, 1.0f},
          {1.0f, 0, 0}, {1.0f, 0, 1.0f}, {1.0f, 1.0f, 0}, {1.0f, 1.0f, 1.0f},
        };
        float dis = 0;
        for (int j = i_base; j < i_base + 8; ++j) {
          if (node_type(children_d[j]) == kLeaf) continue;
          dis += displacement_d[j];
          for (int c = 0; c < channel_normal; ++c) {
            dis += normal_d[c * nnum_d + j] * (xyzs[j % 8][c] - xyzs[id][c]) * imul;
          }
        }
        dis /= count;
        if (dis > 3.0f) dis = 3.0f;
        if (dis < -3.0f) dis = -3.0f;
        if (fabsf(dis) < 1.0f) {
          //bool has_intersection = false;
          // make the voxel has no intersection with the current voxel
          uint32 cube_cases = 0;
          for (int k = 0; k < 8; ++k) {
            float fval = dis;
            for (int j = 0; j < 3; ++j) {
              fval += (0.5f - MarchingCube::corner_[k][j]) * n_avg[j] * imul;
            }
            if (fval < 0) cube_cases |= (1 << k);
          }
          if (cube_cases != 255 && cube_cases != 0) {
            dis = dis < 0 ? -1.0f : 1.0f;
          }
        }

        displacement_d[i] = dis;
      }

      if (has_dis && count < 0.5f) {
        // find the closest point
        int j_min = -1;
        float pti[3], ptj[3], dis_min = 1.0e30f;
        key2xyz(pti, keys_[d][i], d);
        for (int j = 0; j < nnum_d; ++j) {
          if (node_type(children_d[j]) == kLeaf) continue;
          key2xyz(ptj, keys_[d][j], d);
          float dd = fabsf(pti[0] - ptj[0]) + fabsf(pti[1] - ptj[1]) + fabsf(pti[2] - ptj[2]);
          if (dd < dis_min) {
            dis_min = dd;
            j_min = j;
          }
        }
        // calc the displacement
        float dis = displacement_d[j_min];
        key2xyz(ptj, keys_[d][j_min], d);
        for (int c = 0; c < channel_normal; ++c) {
          dis += normal_d[c * nnum_d + j_min] * (ptj[c] - pti[c]) * imul;
        }
        if (dis > 0.0f) dis = 2.0f;
        if (dis < 0.0f) dis = -2.0f;
        displacement_d[i] = dis;
      }
    }
  }
}

bool Octree::save_legacy(std::string& filename) {
  typedef typename KeyTrait<uintk>::uints uints;
  int depth_ = oct_info_.depth();
  int full_layer_ = oct_info_.full_layer();

  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) return false;

  vector<int> node_num;
  for (auto& keys : keys_) {
    node_num.push_back(keys.size());
  }

  vector<int> node_num_accu(depth_ + 2, 0);
  for (int i = 1; i < depth_ + 2; ++i) {
    node_num_accu[i] = node_num_accu[i - 1] + node_num[i - 1];
  }
  int total_node_num = node_num_accu[depth_ + 1];
  int final_node_num = node_num[depth_];

  // calc key
  std::vector<int> key(total_node_num), children(total_node_num);
  int idx = 0;
  for (int d = 0; d <= depth_; ++d) {
    vector<uintk>& keys = keys_[d];
    for (int i = 0; i < keys.size(); ++i) {
      // calc point
      uintk k = keys[i], pt[3];
      compute_pt(pt, k, d);

      // compress
      uints* ptr = reinterpret_cast<uints*>(&key[idx]);
      ptr[0] = static_cast<uints>(pt[0]);
      ptr[1] = static_cast<uints>(pt[1]);
      ptr[2] = static_cast<uints>(pt[2]);
      ptr[3] = static_cast<uints>(d);

      // children
      children[idx] = children_[d][i];

      // update index
      idx++;
    }
  }

  // write
  outfile.write((char*)&total_node_num, sizeof(int));
  outfile.write((char*)&final_node_num, sizeof(int));
  outfile.write((char*)&depth_, sizeof(int));
  outfile.write((char*)&full_layer_, sizeof(int));
  outfile.write((char*)node_num.data(), sizeof(int) * (depth_ + 1));
  outfile.write((char*)node_num_accu.data(), sizeof(int) * (depth_ + 2));
  outfile.write((char*)key.data(), sizeof(int) * total_node_num);
  outfile.write((char*)children.data(), sizeof(int) * total_node_num);
  outfile.write((char*)avg_normals_[depth_].data(), sizeof(float)*avg_normals_[depth_].size());
  outfile.write((char*)displacement_[depth_].data(), sizeof(float)*displacement_[depth_].size());
  outfile.write((char*)avg_labels_[depth_].data(), sizeof(float)*avg_labels_[depth_].size());
  outfile.close();

  return true;
}

//void Octree::set_bbox(const float* bbmin, const float* bbmax) {
//  float center[3];
//  bbox_width_ = -1.0e20f;
//  for (int i = 0; i < 3; ++i) {
//    float dis = bbmax[i] - bbmin[i];
//    if (dis > bbox_width_) bbox_width_ = dis;
//    center[i] = (bbmin[i] + bbmax[i]) * 0.5f;
//  }
//
//  // deal with degenarated case
//  if (bbox_width_ == 0.0) bbox_width_ = ESP;
//
//  // set the bounding box and place the object in the center
//  float radius = bbox_width_ * 0.5f;
//  for (int i = 0; i < 3; ++i) {
//    bbmax_[i] = center[i] + radius;
//    bbmin_[i] = center[i] - radius;
//  }
//}

void Octree::unique_key(vector<uintk>& keys, vector<uintk>& idx) {
  idx.clear();
  idx.push_back(0);

  int n = keys.size(), j = 1;
  for (int i = 1; i < n; i++) {
    if (keys[i] != keys[i - 1]) { // only push back if current key != last key, unique because sorted
      idx.push_back(i); // why push idx back?
      keys[j++] = keys[i];
    }
  }

  keys.resize(j);
  idx.push_back(n);
}


void Octree::serialize() {
  const int sz = oct_info_.sizeof_octree();
  buffer_.resize(sz, 0);
  this->set_cpu(buffer_.data(), &oct_info_);
  //info_ = reinterpret_cast<OctreeInfo*>(buffer_.data());
  //*info_ = oct_info_;

  // concatenate the avg_normals_, avg_features_, and displacement_ into features
  const int depth = oct_info_.depth();
  vector<vector<float> > features = avg_normals_;
  for (int d = 0; d <= depth; ++d) {
    if (oct_info_.has_displace()) {
      features[d].insert(features[d].end(), displacement_[d].begin(), displacement_[d].end());
    }
    // if there is no features in points, the avg_features_ will also be empty
    features[d].insert(features[d].end(), avg_features_[d].begin(), avg_features_[d].end());
    if (oct_info_.save_pts()) {
      features[d].insert(features[d].end(), avg_pts_[d].begin(), avg_pts_[d].end());
    }
  }

#define SERIALIZE_PROPERTY(Dtype, Ptype, Var)                                 \
  do { if (oct_info_.has_property(Ptype)) {                                   \
    Dtype* ptr = reinterpret_cast<Dtype*>(mutable_ptr_cpu(Ptype, 0));         \
    serialize<Dtype>(ptr, Var, oct_info_.locations(Ptype)); }                 \
  } while(0)                                                                  \

  if (oct_info_.is_key2xyz()) {
    vector<vector<uintk> > xyz;
    key_to_xyz(xyz);
    SERIALIZE_PROPERTY(uintk, OctreeInfo::kKey, xyz);
  } else {
    SERIALIZE_PROPERTY(uintk, OctreeInfo::kKey, keys_);
  }
  SERIALIZE_PROPERTY(int, OctreeInfo::kChild, children_);
  SERIALIZE_PROPERTY(float, OctreeInfo::kFeature, features);
  SERIALIZE_PROPERTY(float, OctreeInfo::kLabel, avg_labels_);
  SERIALIZE_PROPERTY(float, OctreeInfo::kSplit, split_labels_);
}

template<typename Dtype>
void Octree::serialize(Dtype* des, const vector<vector<Dtype> >& src, const int location) {
  if (location == -1) {
    for (int d = 0; d <= oct_info_.depth(); ++d) {
      des = std::copy(src[d].begin(), src[d].end(), des);
    }
  } else {
    std::copy(src[location].begin(), src[location].end(), des);
  }
}

void Octree::covered_depth_nodes() {
  // init
  const int depth_ = oct_info_.depth();
  for (int d = 0; d <= depth_; ++d) {
    int nnum = oct_info_.node_num(d);
    dnum_[d].assign(nnum, 0);
    didx_[d].assign(nnum, -1);
  }

  //layer-depth_
  int nnum = oct_info_.node_num(depth_);
  for (int i = 0; i < nnum; ++i) {
    dnum_[depth_][i] = 1;
    didx_[depth_][i] = i;
  }

  // layer-(depth_-1)
  nnum = oct_info_.node_num(depth_ - 1);
  for (int i = 0; i < nnum; ++i) {
    int t = children_[depth_ - 1][i];
    if (node_type(t) == kLeaf) continue;
    dnum_[depth_ - 1][i] = 8;
    didx_[depth_ - 1][i] = t * 8;
  }

  // layer-(depth-2) to layer-0
  for (int d = depth_ - 2; d >= 0; --d) {
    nnum = oct_info_.node_num(d);
    const vector<int> children_d = children_[d];
    for (int i = 0; i < nnum; ++i) {
      int t = children_d[i];
      if (node_type(t) == kLeaf) continue;
      t *= 8;
      for (int j = 0; j < 8; ++j) {
        dnum_[d][i] += dnum_[d + 1][t + j];
      }
      for (int j = 0; j < 8; ++j) {
        if (didx_[d + 1][t + j] != -1) {
          didx_[d][i] = didx_[d + 1][t + j];
          break;
        }
      }
    }
  }
}

void Octree::trim_octree() {
  if (!oct_info_.is_adaptive()) return;
  const int depth = oct_info_.depth();
  const int depth_adp = oct_info_.adaptive_layer();
  const float th_dist = oct_info_.threshold_distance();
  const float th_norm = oct_info_.threshold_normal();
  const bool has_dis = oct_info_.has_displace();

  // generate the drop flag
  enum TrimType { kDrop = 0, kDropChildren = 1,  kKeep = 2 };
  vector<vector<TrimType> > drop(depth + 1);
  for (int d = 0; d <= depth; ++d) {
    drop[d].resize(oct_info_.node_num(d), kKeep);
  }
  for (int d = depth_adp; d <= depth; ++d) {
    int nnum_dp = oct_info_.node_num(d - 1);
    const vector<int>& children_d = children_[d];
    const vector<int>& children_dp = children_[d - 1];
    vector<TrimType>& drop_d = drop[d];
    vector<TrimType>& drop_dp = drop[d - 1];

    bool all_drop = true;
    // generate the drop flag
    for (int i = 0; i < nnum_dp; ++i) {
      int t = children_dp[i];
      if (node_type(t) == kLeaf) continue;

      // generate the drop flag for 8 children nodes:
      // drop the node if its parent node is kDrop or kDropChildren,
      // set the node as kDropChildren if the error is smaller than a threshold
      for (int j = 0; j < 8; ++j) {
        int idx = t * 8 + j;
        if (drop_dp[i] == kKeep) {
          // note that for all the leaf nodes and the finest nodes,
          // distance_err_[d][i] is equal to 1.0e20f, so if it enters the following
          // "if" body, the node_type(children_d[idx]) must be kInternelNode
          //if (distance_err_[d][idx] < th_dist) {
          if ((!has_dis || (has_dis && distance_err_[d][idx] < th_dist)) &&
              normal_err_[d][idx] < th_norm) {
            drop_d[idx] = kDropChildren;
          }
        } else {
          drop_d[idx] = kDrop;
        }

        if (all_drop) {
          // all_drop is false: there is at least one internal node which is kept
          all_drop = !(drop_d[idx] == kKeep &&
                  node_type(children_d[idx]) == kInternelNode);
        }
      }
    }

    // make sure that there is at least one octree node in each layer
    if (all_drop) {
      int max_idx = 0;
      float max_err = -1.0f;
      for (int i = 0; i < nnum_dp; ++i) {
        int t = children_dp[i];
        if (node_type(t) == kLeaf || drop_dp[i] != kKeep) continue;

        for (int j = 0; j < 8; ++j) {
          int idx = t * 8 + j;
          if (node_type(children_d[idx]) == kInternelNode &&
              normal_err_[d][idx] > max_err) {
            max_err = normal_err_[d][idx];
            max_idx = idx;
          }
        }
      }
      drop_d[max_idx] = kKeep;
    }
  }

  // trim the octree
  for (int d = depth_adp; d <= depth; ++d) {
    int nnum_d = oct_info_.node_num(d);
    const vector<TrimType>& drop_d = drop[d];

    vector<uintk> key;
    for (int i = 0; i < nnum_d; ++i) {
      if (drop_d[i] == kDrop) continue;
      key.push_back(keys_[d][i]);
    }
    keys_[d].swap(key);

    vector<int> children;
    for (int i = 0, id = 0; i < nnum_d; ++i) {
      if (drop_d[i] == kDrop) continue;
      int ch = (drop_d[i] == kKeep && node_type(children_[d][i]) != kLeaf) ? id++ : -1;
      children.push_back(ch);
    }
    children_[d].swap(children);

    auto trim_data = [&](vector<float>& signal) {
      vector<float> data;
      int channel = signal.size() / nnum_d;
      if (channel == 0) return;
      for (int i = 0; i < nnum_d; ++i) {
        if (drop_d[i] == kDrop) continue;
        for (int c = 0; c < channel; ++c) {
          data.push_back(signal[c * nnum_d + i]);
        }
      }
      //signal.swap(data);
      // transpose
      int num = data.size() / channel;
      signal.resize(data.size());
      for (int i = 0; i < num; ++i) {
        for (int c = 0; c < channel; ++c) {
          signal[c * num + i] = data[i * channel + c];
        }
      }
    };

    trim_data(displacement_[d]);
    trim_data(avg_normals_[d]);
    trim_data(avg_features_[d]);
    trim_data(avg_labels_[d]);
  }

  // update the node number
  calc_node_num();

  // generate split label
  if (oct_info_.has_property(OctreeInfo::kSplit)) {
    calc_split_label();
  }

  // serialization
  serialize();
}

void Octree::key_to_xyz(vector<vector<uintk> >& xyz) {
  typedef typename KeyTrait<uintk>::uints uints;
  const int depth = oct_info_.depth();
  const int channel = oct_info_.channel(OctreeInfo::kKey);
  //assert(channel == 1);
  xyz.resize(depth + 1);
  for (int d = 0; d <= depth; ++d) {
    int nnum = oct_info_.node_num(d);
    xyz[d].resize(nnum * channel, 0);
    uintk* xyz_d = xyz[d].data();
    for (int i = 0; i < nnum; ++i) {
      uintk pt[3] = { 0, 0, 0 };
      compute_pt(pt, keys_[d][i], d);
      uints* ptr = reinterpret_cast<uints*>(xyz_d + i);
      for (int c = 0; c < 3; ++c) {
        ptr[c] = static_cast<uints>(pt[c]);
      }
    }
  }
}

void Octree::calc_split_label() {
  const int depth = oct_info_.depth();
  const int channel = oct_info_.channel(OctreeInfo::kSplit); // is 1 (by default)
  const bool adaptive = oct_info_.is_adaptive();

  for (int d = 0; d <= depth; ++d) {
    int nnum_d = oct_info_.node_num(d);
    split_labels_[d].assign(nnum_d, 1);       // initialize as 1 (non-empty, split)
    for (int i = 0; i < nnum_d; ++i) {
      if (node_type(children_[d][i]) == kLeaf) {
        split_labels_[d][i] = 0;              // empty node
        if (adaptive) {
          float t = fabsf(avg_normals_[d][i]) + fabsf(avg_normals_[d][nnum_d + i]) +
                    fabsf(avg_normals_[d][2 * nnum_d + i]);
          // todo: t != 0 && has_intersection
          if (t != 0) split_labels_[d][i] = 2; // surface-well-approximated
        }
      }
    }
  }
}


void Octree::valid_depth_range(int& depth_start, int& depth_end) const {
  const int depth = info_->depth();
  const int depth_full = info_->full_layer();
  const int depth_adpt = info_->adaptive_layer();

  depth_start = clamp(depth_start, depth_full, depth);
  if (info_->is_adaptive() && depth_start < depth_adpt) depth_start = depth_adpt;
  if (-1 != info_->locations(OctreeInfo::kFeature)) depth_start = depth;

  depth_end = clamp(depth_end, depth_start, depth);
}


void Octree::octree2pts(Points& point_cloud, int depth_start, int depth_end,
    bool rescale) const {
  const int depth = info_->depth();
  const float* bbmin = info_->bbmin();
  const float kMul = info_->bbox_max_width() / float(1 << depth);
  valid_depth_range(depth_start, depth_end);

  vector<float> pts, normals, labels;
  for (int d = depth_start; d <= depth_end; ++d) {
    const int* child_d = children_cpu(d);
    const float* label_d = label_cpu(d);
    const float scale = (1 << (depth - d)) * kMul;
    const int num = info_->node_num(d);

    for (int i = 0; i < num; ++i) {
      if ((node_type(child_d[i]) == kInternelNode && d != depth) ||
          (node_type(child_d[i]) == kLeaf && d == depth)) continue;

      float n[3], pt[3];
      node_normal(n, i, d);
      float len = fabsf(n[0]) + fabsf(n[1]) + fabsf(n[2]);
      if (len == 0 && d != depth) continue;  // for adaptive octree
      node_pos(pt, i, d);

      for (int c = 0; c < 3; ++c) {
        if (rescale) {                       // !!! note the scale and bbmin
          pt[c] = pt[c] * scale + bbmin[c];
        }
        normals.push_back(n[c]);
        pts.push_back(pt[c]);
      }
      if (label_d != nullptr) labels.push_back(label_d[i]);
    }
  }

  point_cloud.set_points(pts, normals, vector<float>(), labels);
}

/*void Octree::export_points_scaled(std::string fname, vector<float> pts_scaled) {
  const int depth = info_->depth();
  const float* bbmin = info_->bbmin();
  const float scale = info_->bbox_max_width() / float(1 << (depth));

  std::ofstream objFile;
  objFile.open(fname);

  float pt[3] = {0,0,0};
  float ct[3] = {-4.030212, 544.228516, 163.101822}; // only works if I have access to octree point center GOOD ENOUGH FOR NOW
  for (int i = 0; i < pts_scaled.size(); i+=3) {

    for (int c = 0; c < 3; c++) {
      pt[c] = (pts_scaled[i+c] * scale) + bbmin[c] + ct[c];
    }

    objFile << "v " << pt[0] << " " << pt[1] << " " << pt[2] << std::endl;
  }

  objFile.close();
}*/


void Octree::octree2mesh(vector<float>& V, vector<int>& F, int depth_start,
    int depth_end, bool rescale) const {
  const int depth = info_->depth();
  const float* bbmin = info_->bbmin();
  const float* bbmax = info_->bbmax();
  const float kMul = info_->bbox_max_width() / float(1 << depth);
  valid_depth_range(depth_start, depth_end);

  V.clear(); F.clear();
  for (int d = depth_start; d <= depth_end; ++d) {
    const int* child_d = children_cpu(d);
    const int num = info_->node_num(d);
    const float scale = (1 << (depth - d)) * kMul;
    const float cube_size = (1 << (depth - d));

    int num_verts = V.size();

    vector<float> pts, normals, pts_ref, coefs;
    for (int i = 0; i < num; ++i) {
      //if (node_type(child_d[i]) == kInternelNode && d != depth) continue;
      if (node_type(child_d[i]) == kInternelNode && d != depth) continue;

      float n[3], pt[3], pt_ref[3], coef[10];
      node_coefficients(coef, i, d);

      float len = 0;
      for (int c = 0; c < 10; c++) { len += fabs(n[c]); }
      if (len == 0) continue;             // if normal is zero cell has no surface
      node_pos(pt, i, d, pt_ref);

      Vector3f cell_base, surf_center;
      for (int c = 0; c < 3; ++c) {
        cell_base(c) = pt_ref[c] * cube_size;           // adjust cell base pos to global scale
        surf_center(c) = cell_base(c) + 0.5*cube_size;  // adjust surfel center to global scale
      }

      VectorXf surf_coefs(10);
      for (int c = 0; c < 10; ++c) {
        surf_coefs(c) = coef[c];
      }
      
      // render surface
      polynomial2::visualize_quadric(&V, &F, cell_base, cube_size, 4, surf_center, surf_coefs);
    }

  }

  // TODO fix this
  // rescale verts to fit to point cloud
  /*for (int i = num_verts; i < V.size()/3; i+=3) {
    for(int j=0; j<3;j++){
      V[i+j] = V[i+j]/scale + bbmin[j];
    }
  }*/

  // fix vertex indexing: indices have to start at 0, because write_obj takes care of it later
  for (int i = 0; i < F.size(); i++) {
    F[i] += 0;
  }
}


//// only work for adaptive octree
//void Octree::extrapolate_signal() {
//  const int depth = info().depth();
//  const int full_layer = info().full_layer();
//  const float imul = 2.0f / sqrtf(3.0f);
//  const int channel_label = info().channel(OctreeInfo::kLabel);
//  const int channel_normal = 3;
//  const bool has_dis = info().has_displace();
//  const bool has_label = channel_label;
//
//  for (int d = depth; d >= full_layer; --d) {
//    const int nnum_d = info().node_num(d);
//    const int* child_d = children_cpu(d);
//    const uintk* keys_d = key_cpu(d);
//    float* normal_d = mutable_feature_cpu(d);
//    //float* label_d = mutable_label(d); // !!!todo: label
//    float* displacement_d = normal_d + 3 * nnum_d;
//
//    // For adaptive octree, there is kNonEmptyLeaf,
//    // but the original child_d contains only kLeaf and kInternelNode,
//    // So we use children_d to mark the kNonEmptyLeaf
//    vector<int> children_d(child_d, child_d + nnum_d);
//    for (int i = 0; i < nnum_d; ++i) {
//      if (d == depth) break;
//      if (node_type(children_d[i]) == kLeaf) {
//        float n[3] = { 0 };
//        node_normal(n, i, d);
//        float len = fabsf(n[0]) + fabsf(n[1]) + fabsf(n[2]);
//        if (len != 0) { children_d[i] = -2; }
//      }
//    }
//
//    for (int i = 0; i < nnum_d; ++i) {
//      if (node_type(children_d[i]) != kLeaf) continue;
//      int id = i % 8;
//      int i_base = i - id;
//
//      float count = ESP; // the non-empty node number
//      for (int j = i_base; j < i_base + 8; ++j) {
//        if (node_type(children_d[j]) != kLeaf) count += 1.0f;
//      }
//
//      vector<float> n_avg(channel_normal, 0.0f);
//      for (int j = i_base; j < i_base + 8; ++j) {
//        if (node_type(children_d[j]) == kLeaf) continue;
//        for (int c = 0; c < channel_normal; ++c) {
//          n_avg[c] += normal_d[c * nnum_d + j];
//        }
//      }
//
//      float ilen = 1.0f / (norm2(n_avg) + ESP);
//      for (int c = 0; c < channel_normal; ++c) {
//        n_avg[c] *= ilen;
//        normal_d[c * nnum_d + i] = n_avg[c];  // output
//      }
//
//      if (has_dis && count > 0.5f) {
//        float xyzs[8][3] = {
//          {0, 0, 0}, {0, 0, 1.0f}, {0, 1.0f, 0}, {0, 1.0f, 1.0f},
//          {1.0f, 0, 0}, {1.0f, 0, 1.0f}, {1.0f, 1.0f, 0}, {1.0f, 1.0f, 1.0f},
//        };
//        float dis = 0;
//        for (int j = i_base; j < i_base + 8; ++j) {
//          if (node_type(children_d[j]) == kLeaf) continue;
//          dis += displacement_d[j];
//          for (int c = 0; c < channel_normal; ++c) {
//            dis += normal_d[c * nnum_d + j] * (xyzs[j % 8][c] - xyzs[id][c]) * imul;
//          }
//        }
//        dis /= count;
//        if (dis > 3.0f) dis = 3.0f;
//        if (dis < -3.0f) dis = -3.0f;
//        if (fabsf(dis) < 1.0f) {
//          // make the voxel has no intersection with the current voxel
//          uintk cube_cases = 0;
//          for (int k = 0; k < 8; ++k) {
//            float fval = dis;
//            for (int j = 0; j < 3; ++j) {
//              fval += (0.5 - MarchingCube::corner_[k][j]) * n_avg[j] * imul;
//            }
//            if (fval < 0) cube_cases |= (1 << k);
//          }
//          if (cube_cases != 255 && cube_cases != 0) {
//            dis = dis < 0 ? -1.0f : 1.0f;
//          }
//        }
//
//        displacement_d[i] = dis;
//      }
//
//      if (has_dis && count < 0.5f) {
//        // find the closest point
//        int j_min = -1;
//        float pti[3], ptj[3], dis_min = 1.0e30f;
//        key2xyz(pti, keys_d[i], d);
//        for (int j = 0; j < nnum_d; ++j) {
//          if (node_type(children_d[j]) == kLeaf) continue;
//          key2xyz(ptj, keys_d[j], d);
//          float dis = abs(pti[0] - ptj[0]) + abs(pti[1] - ptj[1]) + abs(pti[2] - ptj[2]);
//          if (dis < dis_min) {
//            dis_min = dis;
//            j_min = j;
//          }
//        }
//        // calc the displacement
//        float dis = displacement_d[j_min];
//        key2xyz(ptj, keys_d[j_min], d);
//        for (int c = 0; c < channel_normal; ++c) {
//          dis += normal_d[c * nnum_d + j_min] * (ptj[c] - pti[c]) * imul;
//        }
//        if (dis > 0.0f) dis = 2.0f;
//        if (dis < 0.0f) dis = -2.0f;
//        displacement_d[i] = dis;
//      }
//    }
//  }
//}


