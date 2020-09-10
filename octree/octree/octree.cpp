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
  vector<uint32> node_keys, sorted_idx;      

  if (overlap_amount() == 0 || true) {
    sort_keys(node_keys, sorted_idx, pts_scaled);
  } else {
    sort_keys_overlap(node_keys, sorted_idx, pts_scaled, overlap_amount());
  }

  // sort keys and numbers into cells
  vector<uint32> unique_idx;
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
  /*if (oct_info_.has_property(OctreeInfo::kSplit)) {
    calc_split_label();
  }

  // extrapolate node feature
  if (oct_info_.extrapolate() && oct_info_.locations(OctreeInfo::kFeature) == -1) {
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
  #pragma omp parallel for
  for (int i = 0; i < npt; i++) {
    int i3 = i * 3;
    for (int j = 0; j < 3; j++) {
      pts_scaled[i3 + j] = (pts[i3 + j] - bbmin[j]) * mul;
    }
  }
}

void Octree::sort_keys(vector<uint32>& sorted_keys, vector<uint32>& sorted_idx,
    const vector<float>& pts_scaled) {

  // compute the code
  int depth_ = oct_info_.depth(); // max depth
  int npt = pts_scaled.size() / 3; // number of points
  vector<uint64> code(npt);
  #pragma omp parallel for
  for (int i = 0; i < npt; i++) {
    // compute key
    uint32 pt[3], key;
    for (int j = 0; j < 3; ++j) {
      pt[j] = static_cast<uint32>(pts_scaled[3 * i + j]); // scaled points equal to flooring xyz into pt[]
      // 0,1,2 | 2,3,4 | 4,5,6 overwrite on every second write
    }
    compute_key(key, pt, depth_);

    // generate code
    uint32* ptr = reinterpret_cast<uint32*>(&code[i]);
    ptr[0] = i; // code 0-32bits = point index
    ptr[1] = key; // code 32-64bits = generated key
    
  }

  // sort all the code
  std::sort(code.begin(), code.end()); // will be sorted by generated key

  // unpack the code
  sorted_keys.resize(npt);
  sorted_idx.resize(npt);
  #pragma omp parallel for
  for (int i = 0; i < npt; i++) {
    uint32* ptr = reinterpret_cast<uint32*>(&code[i]);
    sorted_idx[i] = ptr[0]; // split code into index
    sorted_keys[i] = ptr[1]; // and key
  }
}

void Octree::sort_keys_overlap(vector<uint32>& sorted_keys, vector<uint32>& sorted_idx,
    const vector<float>& pts_scaled, float overlap) {

  int depth_ = oct_info_.depth(); // max depth
  int npt = pts_scaled.size() / 3; // number of points
  int num_codes = 0;
  float max_idx = std::pow(2, depth_) - 1.0f;

  // maximum number of cells per point allowed = 27 -> max overlap = 1.0
  vector<uint64> code(npt*27);
  #pragma omp parallel for
  for (int i = 0; i < npt; i++) {

    // shift point along xyz -> 27 combinations
    vector<uint32> new_keys, new_idx;
    float shift[3] = {0,0,0};
    for (int x = -1; x <= 1; x++) {
      shift[0] = float(x) * overlap;
      for (int y = -1; y <= 1; y++) {
        shift[1] = float(y) * overlap;
        for (int z = -1; z <= 1; z++) {
          shift[2] = float(z) * overlap;

          // compute key across xyz channel
          uint32 pt[3], key;
          for (int j = 0; j < 3; ++j) {
            // scaled points equal to flooring xyz into pt[]
            pt[j] = static_cast<uint32>(std::min(max_idx, std::max(0.0f, pts_scaled[3 * i + j] + shift[j]))); 
          }
          compute_key(key, pt, depth_);

          // check if shifted point results in new key
          bool unique = true;
          for (int j = 0; j < new_keys.size(); j++) {
            if (new_keys[j] == key) { 
              unique = false;
              break;
            }
          }

          // add if node key is unique among shifted points
          if (unique){
            new_idx.push_back(i);
            new_keys.push_back(key);
          }
        }
      }
    }

    // add shifted point indices to global code
    for (int j = 0; j < new_keys.size(); j++) {
      uint32* ptr = reinterpret_cast<uint32*>(&code[num_codes]);
      ptr[0] = new_idx[j]; // code 0-32bits = point index
      ptr[1] = new_keys[j]; // code 32-64bits = generated key*/
      num_codes++;
    }
  }
  code.resize(num_codes);

  // sort all the code
  std::sort(code.begin(), code.end()); // will be sorted by generated key

  // unpack the code
  sorted_keys.resize(num_codes);
  sorted_idx.resize(num_codes);
  #pragma omp parallel for
  for (int i = 0; i < num_codes; i++) {
    uint32* ptr = reinterpret_cast<uint32*>(&code[i]);
    sorted_idx[i] = ptr[0]; // split code into index
    sorted_keys[i] = ptr[1]; // and key
  }
}

void Octree::build_structure(vector<uint32>& node_keys) {
  const int depth_ = oct_info_.depth();
  const int full_layer_ = oct_info_.full_layer();
  children_.resize(depth_ + 1);
  keys_.resize(depth_ + 1);

  // layer 0 to full_layer_: the octree is full in these layers
  for (int curr_depth = 0; curr_depth <= full_layer_; curr_depth++) {
    vector<int>& children = children_[curr_depth];
    vector<uint32>& keys = keys_[curr_depth];

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
    vector<uint32> parent_keys(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      parent_keys[i] = node_keys[i] >> 3;
    }

    // compute unique parent key
    vector<uint32> parent_pidx;
    unique_key(parent_keys, parent_pidx);

    // augment children keys and create nodes
    int np = parent_keys.size();
    int nch = np << 3;
    vector<int>& children = children_[curr_depth];
    vector<uint32>& keys = keys_[curr_depth];
    children.resize(nch, -1);
    keys.resize(nch, 0);

    for (int i = 0; i < nch; i++) {
      int j = i >> 3;
      keys[i] = (parent_keys[j] << 3) | (i % 8);
    }

    // compute base address for each node
    vector<uint32> addr(nch);
    for (int i = 0; i < np; i++) {
      for (uint32 j = parent_pidx[i]; j < parent_pidx[i + 1]; j++) {
        addr[j] = i << 3;
      }
    }

    // set children pointer and parent pointer
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      // address
      uint32 k = (node_keys[i] & 7u) | addr[i];

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
      uint32 j = node_keys[i];
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

  const int channel_pt = 3;
  const int channel_normal = 9;
  const int channel_dis = 3;
 
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
    pt_d.assign(nnum_d * channel_pt, 0.0f);
    displacement_d.assign(channel_dis * nnum_d, 0.0f);
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
        for (int c = 0; c < 3; c++) {
          // store surface center in displacement_d (0-3)
          displacement_d[c * nnum_d + i] = (helper.surf_center(c) - cell_center(c)) / scale;
          //displacement_d[c * nnum_d + i] = 2.1+c*0.1;

          // store surface in normal (0-3)
          normal_d[c * nnum_d + i] = helper.surf_normal(c);
          //normal_d[c * nnum_d + i] = 0.1+c*0.1;
        }

        // store surface coefficients in normal (3-9)
        for (int c = 0; c < 6; c++) {
          normal_d[(c+3) * nnum_d + i] = helper.surf_coefs(c, 0);
          //normal_d[(c+3) * nnum_d + i] = 1.1+c*0.1;
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

  const int channel_pt = 3;
  const int channel_normal = 9;
  const int channel_dis = 3;

  // iterate over each depth layer
  for (int d = depth_adp-1; d >= 0; d--) {
    // number of nodes and point indices on depth
    const int nnum_d = oct_info_.node_num(d);

    const vector<int>& children = children_[d];
    const vector<uint32>& key_d = keys_[d];

    // data arrays for current depth layer
    const vector<int>& children_d = children_[d];
    vector<float>& normal_d = avg_normals_[d];
    vector<float>& pt_d = avg_pts_[d];
    vector<float>& displacement_d = displacement_[d];
    vector<float>& normal_err_d = normal_err_[d];
    vector<float>& distance_err_d = distance_err_[d];

    // allocate memory for data arrays
    normal_d.assign(nnum_d * channel_normal, 0.0f);
    pt_d.assign(nnum_d * channel_pt, 0.0f);
    displacement_d.assign(channel_dis * nnum_d, 0.0f);
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

            for (int c = 0; c < 3; c++) {
              // store surface center in displacement_d (0-3)
              displacement_d[c * nnum_d + i] += displacement_[d+1][c * nnum_d + a];
              // store surface in normal (0-3)
              normal_d[c * nnum_d + i] = avg_normals_[d+1][c * nnum_d + a];
            }

            // store surface coefficients in normal (3-9)
            for (int c = 0; c < 6; c++) {
              normal_d[(c+3) * nnum_d + i] += avg_normals_[d+1][(c+3) * nnum_d + i];
            }

          }

        }

        // average all copied
        for (int c = 0; c < 9; c++) {
          if (c < 3) {
            displacement_d[c * nnum_d + i] /= num_non_empty_children;
          }
          normal_d[c * nnum_d + i] /= num_non_empty_children;
        }

        normal_err_d[i] = ERROR_THRESHOLD + 1.0;
        distance_err_d[i] = ERROR_THRESHOLD + 1.0;
      }


    }
  }
}


// compute the average signal for the last octree layer
void Octree::calc_signal(const Points& point_cloud, const vector<float>& pts_scaled,
    const vector<uint32>& sorted_idx, const vector<uint32>& unique_idx) {
  int depth = oct_info_.depth();
  const float* normals = point_cloud.ptr(PointsInfo::kNormal);
  const float* features = point_cloud.ptr(PointsInfo::kFeature);
  const float* labels = point_cloud.ptr(PointsInfo::kLabel);
  const int nnum = oct_info_.node_num(depth);

  float support_radius = std::sqrt(3*std::pow(0.5+overlap_amount(), 2.0));

  const vector<int>& children = children_[depth];
  // --------------------------- NORMALS signal
  if (normals != nullptr) {
    const int channel = point_cloud.info().channel(PointsInfo::kNormal);
    const int offset = info().has_implicit() ? 6 : 0;
    avg_normals_[depth].assign((channel+offset) * nnum, 0.0f); // allocate mem for normals

    #pragma omp parallel for
    for (int i = 0; i < nnum; i++) {
      int t = children[i];
      if (node_type(t) == kLeaf) continue;

      // iterate over all scaled points in cell
      vector<float> avg_normal(channel, 0.0f);
      for (uint32 j = unique_idx[t]; j < unique_idx[t + 1]; j++) {
        int h = sorted_idx[j];
        for (int c = 0; c < channel; ++c) {
          avg_normal[c] += normals[channel * h + c];
        }
      }

      // calc normalization facotr
      float factor = norm2(avg_normal);
      if (factor < 1.0e-6f) {
        int h = sorted_idx[unique_idx[t]];
        for (int c = 0; c < channel; ++c) {
          avg_normal[c] = normals[channel * h + c];
        }
        factor = norm2(avg_normal) + ESP;
      }
      // store normalized avg normal
      for (int c = 0; c < channel; ++c) {
        avg_normals_[depth][c * nnum + i] = avg_normal[c] / factor;
      }
    }
  }

  // ------------------------ FEATURES
  if (features != nullptr && false) {
    const int channel = point_cloud.info().channel(PointsInfo::kFeature);
    avg_features_[depth].assign(channel * nnum, 0.0f);

    #pragma omp parallel for
    for (int i = 0; i < nnum; i++) {
      int t = children[i];
      if (node_type(t) == kLeaf) continue;

      vector<float> avg_feature(channel, 0.0f);
      for (uint32 j = unique_idx[t]; j < unique_idx[t + 1]; j++) {
        int h = sorted_idx[j];
        for (int c = 0; c < channel; ++c) {
          avg_feature[c] += features[channel * h + c];
        }
      }

      float factor = unique_idx[t + 1] - unique_idx[t] + ESP;
      for (int c = 0; c < channel; ++c) {
        avg_features_[depth][c * nnum + i] = avg_feature[c] / factor;
      }
    }
  }

  // ------------------------------ LABELS
  if (labels != nullptr && false) {
    // the channel of label is fixed as 1
    avg_labels_[depth].assign(nnum, -1.0f);   // initialize as -1
    const int npt = point_cloud.info().pt_num();
    max_label_ = static_cast<int>(*std::max_element(labels, labels + npt)) + 1;

    #pragma omp parallel for
    for (int i = 0; i < nnum; i++) {
      int t = children[i];
      if (node_type(t) == kLeaf) continue;

      vector<int> avg_label(max_label_, 0);
      for (uint32 j = unique_idx[t]; j < unique_idx[t + 1]; j++) {
        int h = sorted_idx[j];
        avg_label[static_cast<int>(labels[h])] += 1;
      }

      avg_labels_[depth][i] = static_cast<float>(std::distance(avg_label.begin(),
                  std::max_element(avg_label.begin(), avg_label.end())));
    }
  }

  // -------------------------------- DISPLACEMENT
  if (oct_info_.has_displace() || oct_info_.save_pts()) {
    const int channel = 3;
    const float mul = 1.1547f; // = 2.0f / sqrt(3.0f)
    avg_pts_[depth].assign(nnum * channel, 0.0f); // allocate mem for average point in cell
    int channel_dis = normals == nullptr ? 4 : 1;
    vector<float>& displacement = displacement_[depth];
    displacement.assign(channel_dis * nnum, 0.0f); // allocate mem for displacement factor

    #pragma omp parallel for
    for (int i = 0; i < nnum; i++) {
      int t = children[i];
      if (node_type(t) == kLeaf) continue;

      float avg_pt[3] = { 0.0f, 0.0f, 0.0f };
      for (uint32 j = unique_idx[t]; j < unique_idx[t + 1]; j++) {
        int h = sorted_idx[j];
        for (int c = 0; c < 3; ++c) {
          avg_pt[c] += pts_scaled[3 * h + c];
        }
      }

      float dis[4] = {0.0f, 0.0f, 0.0f, 0.0f };
      float factor = unique_idx[t + 1] - unique_idx[t] + ESP; // points number
      for (int c = 0; c < 3; ++c) {
        avg_pt[c] /= factor;

        float fract_part = 0.0f, int_part = 0.0f;
        fract_part = std::modf(avg_pt[c], &int_part);

        dis[c] = fract_part - 0.5f;
        if (normals != nullptr) {
          dis[3] += dis[c] * avg_normals_[depth][c * nnum + i];
        } else {
          dis[3] = 1.0f;
        }

        avg_pts_[depth][c * nnum + i] = avg_pt[c];
      }

      if (normals != nullptr) {
        displacement[i] = dis[3] * mul;            // !!! note the *mul* !!!
      } else {
        for (int c = 0; c < 3; ++c) {
          displacement[c * nnum + i] = dis[c];
        }
        displacement[3 * nnum + i] = dis[3] * mul; // !!! note the *mul* !!!
      }
    }
  }

  if (oct_info_.has_implicit()) {
    vector<float>& displacement = displacement_[depth];

    #pragma omp parallel for
    for (int i = 0; i < nnum; i++) {
      int t = children[i];
      if (node_type(t) == kLeaf) continue;

      // only approximate if enough points in support radius
      int num_points = unique_idx[t+1] - unique_idx[t];
      if (num_points >= 20) {

        // calc uv-plane center
        //uint32 pt[3] = { 0, 0, 0 };
        //compute_pt(pt, keys_[depth][i], depth);
        Eigen::Vector3f plane_center;
        for (int c = 0; c < 3; c++) {
          plane_center(c) = avg_pts_[depth][c * nnum + i];
        }

        // get uv-plane normal 
        Eigen::Vector3f plane_normal;
        plane_normal << avg_normals_[depth][i], avg_normals_[depth][1 * nnum + i], avg_normals_[depth][2 * nnum + i];
        plane_normal.normalize();
        Eigen::MatrixXf R = polynomial::calc_rotation_matrix(plane_normal);

        // approximate surface         
        Eigen::MatrixXf coefs = polynomial::biquad_approximation(pts_scaled, sorted_idx, unique_idx[t], unique_idx[t+1], R, plane_center, support_radius);

        // calc coefficient sum as measure of good approximation
        float abs_sum = 0;
        for (int c = 0; c < 6; c++) {
          abs_sum += abs(coefs(c, 0));
        }

        // if good approximation
        float threshold = 5;
        bool use_any = false;
        if (!((coefs.maxCoeff() < threshold && coefs.minCoeff() > -threshold && abs_sum < threshold*2) || use_any)) {
          for (int c = 0; c < 6; c++) { coefs(c,0) = 0; }
        }

        // store coefficients into the remainings space 3-9
        for (int c = 0; c < 6; c++) {
          avg_normals_[depth][(c+3) * nnum + i] = coefs(c, 0);
        }
      }
    }
  }
}

void Octree::calc_signal(const bool calc_normal_err, const bool calc_dist_err, const vector<float>& pts_scaled, const vector<uint32>& sorted_idx, const vector<uint32>& unique_idx) {
  const int depth = oct_info_.depth();
  const int depth_adp = oct_info_.adaptive_layer();
  const int nnum_depth = oct_info_.node_num(depth);
  const float imul = 2.0f / sqrtf(3.0f);
  const vector<int>& children_depth = children_[depth];
  const vector<float>& normal_depth = avg_normals_[depth];
  const vector<float>& pt_depth = avg_pts_[depth];
  const vector<float>& feature_depth = avg_features_[depth];
  const vector<float>& label_depth = avg_labels_[depth];

  const int channel_pt = pt_depth.size() / nnum_depth;
  //const int channel_normal = (normal_depth.size() / nnum_depth) + oct_info_.has_implicit() ? 6 : 0;
  const int channel_normal = normal_depth.size() / nnum_depth;
  const int channel_feature = feature_depth.size() / nnum_depth;
  const int channel_label = label_depth.size() / nnum_depth;

  const bool has_pt = !pt_depth.empty();
  const bool has_dis = !displacement_[depth].empty();
  const bool has_normal = !normal_depth.empty();
  const bool has_implicit = oct_info_.has_implicit();
  const bool has_feature = !feature_depth.empty();
  const bool has_label = !label_depth.empty();

  if (calc_normal_err) normal_err_[depth].resize(nnum_depth, 1.0e20f);
  if (calc_dist_err) distance_err_[depth].resize(nnum_depth, 1.0e20f);

  //std::ofstream outfile("/home/jeri/dev/implicit_ocnn/debug.txt");

  for (int d = depth - 1; d >= 0; --d) {
    const vector<int>& dnum_d = dnum_[d];
    const vector<int>& didx_d = didx_[d];

    const vector<int>& children_d = children_[d];
    const vector<uint32>& key_d = keys_[d];
    const float scale = static_cast<float>(1 << (depth - d));

    vector<float>& normal_d = avg_normals_[d];
    vector<float>& pt_d = avg_pts_[d];
    vector<float>& label_d = avg_labels_[d];
    vector<float>& feature_d = avg_features_[d];
    vector<float>& displacement_d = displacement_[d];
    vector<float>& normal_err_d = normal_err_[d];
    vector<float>& distance_err_d = distance_err_[d];

    const int nnum_d = oct_info_.node_num(d);
    int channel_dis = has_normal ? 1 : 4;
    if (has_normal) normal_d.assign(nnum_d * channel_normal, 0.0f);
    if (has_pt) pt_d.assign(nnum_d * channel_pt, 0.0f);
    if (has_feature) feature_d.assign(nnum_d * channel_feature, 0.0f);
    if (has_label) label_d.assign(nnum_d * channel_label, -1.0f);// !!! init as -1
    if (has_dis) displacement_d.assign(channel_dis * nnum_d, 0.0f);
    if (calc_normal_err) normal_err_d.assign(nnum_d, 1.0e20f);   // !!! initialized
    if (calc_dist_err) distance_err_d.assign(nnum_d, 1.0e20f);   // !!! as 1.0e20f


    // iterate over all nodes at current depth
    for (int i = 0; i < nnum_d; ++i) {
      if (node_type(children_d[i]) == kLeaf) continue;

      vector<float> n_avg(channel_normal, 0.0f);
      if (has_normal) {

        // iterate over all nodes from final layer contained in cube
        for (int j = didx_d[i]; j < didx_d[i] + dnum_d[i]; ++j) {
          if (node_type(children_depth[j]) == kLeaf) continue;

          for (int c = 0; c < channel_normal; ++c) {
            n_avg[c] += normal_depth[c * nnum_depth + j]; // copy normal from deepest layer
          }
        }

        // iterate over contained  nodes and avg their normals
        float len = norm2(n_avg);
        if (len < 1.0e-6f) {
          for (int j = didx_d[i]; j < didx_d[i] + dnum_d[i]; ++j) {
            if (node_type(children_depth[j]) == kLeaf) continue;
            for (int c = 0; c < channel_normal; ++c) {
              n_avg[c] = normal_depth[c * nnum_depth + j];
            }
          }
          len = norm2(n_avg) + ESP;
        }

        // average normal of contained nodes
        for (int c = 0; c < channel_normal; ++c) {
          n_avg[c] /= len;
          normal_d[c * nnum_d + i] = n_avg[c];  // output
        }
      }

      float count = ESP; // the non-empty leaf node in the finest layer
      for (int j = didx_d[i]; j < didx_d[i] + dnum_d[i]; ++j) {
        if (node_type(children_depth[j]) != kLeaf) count += 1.0f;
      }

      // avg points of finest contained nodes
      vector<float> pt_avg(channel_pt, 0.0f);
      if (has_pt) {
        for (int j = didx_d[i]; j < didx_d[i] + dnum_d[i]; ++j) {
          if (node_type(children_depth[j]) == kLeaf) continue;
          for (int c = 0; c < channel_pt; ++c) {
            pt_avg[c] += pt_depth[c * nnum_depth + j];
          }
        }

        for (int c = 0; c < channel_pt; ++c) {
          pt_avg[c] /= count * scale;         // !!! note the scale
          pt_d[c * nnum_d + i] = pt_avg[c];   // output
        }
      }

      vector<float> f_avg(channel_feature, 0.0f);
      if (has_feature) {
        for (int j = didx_d[i]; j < didx_d[i] + dnum_d[i]; ++j) {
          if (node_type(children_depth[j]) == kLeaf) continue;
          for (int c = 0; c < channel_feature; ++c) {
            f_avg[c] += feature_depth[c * nnum_depth + j];
          }
        }

        for (int c = 0; c < channel_feature; ++c) {
          f_avg[c] /= count;
          feature_d[c * nnum_d + i] = f_avg[c]; // output
        }
      }

      vector<int> l_avg(max_label_, 0);
      if (has_label) {
        for (int j = didx_d[i]; j < didx_d[i] + dnum_d[i]; ++j) {
          if (node_type(children_depth[j]) == kLeaf) continue;
          l_avg[static_cast<int>(label_depth[j])] += 1;
        }
        label_d[i] = static_cast<float>(std::distance(l_avg.begin(),
                    std::max_element(l_avg.begin(), l_avg.end())));
      }

      uint32 ptu_base[3];
      compute_pt(ptu_base, key_d[i], d);
      float pt_base[3] = { ptu_base[0], ptu_base[1], ptu_base[2] };
      if (has_dis) {
        float dis_avg[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (int c = 0; c < 3; ++c) {
          float fract_part = pt_avg[c] - pt_base[c];
          dis_avg[c] = fract_part - 0.5f;
          if (has_normal) {
            dis_avg[3] += dis_avg[c] * n_avg[c];
          } else {
            dis_avg[3] = 1.0f;
          }
        }
        if (!has_normal) {
          for (int c = 0; c < 3; ++c) {
            displacement_d[c * nnum_d + i] = dis_avg[c];
          }
          displacement_d[3 * nnum_d + i] = dis_avg[3] * imul;
        } else {
          displacement_d[i] = dis_avg[3] * imul; // IMPORTANT: RESCALE
        }
      }

      float nm_err = 0.0f;
      if (calc_normal_err && has_normal && d >= depth_adp) {
        for (int j = didx_d[i]; j < didx_d[i] + dnum_d[i]; ++j) {
          if (node_type(children_depth[j]) == kLeaf)  continue;
          for (int c = 0; c < 3; ++c) {
            float tmp = normal_depth[c * nnum_depth + j] - n_avg[c];
            nm_err += tmp * tmp;
          }
        }
        nm_err /= count;
        normal_err_d[i] = nm_err;
      }

      if (calc_dist_err && has_pt && d >= depth_adp) {
        // the error from the original geometry to the averaged geometry
        float distance_max1 = -1.0f;
        // !!! note the scale
        float pt_avg1[3] = { pt_avg[0] * scale, pt_avg[1] * scale, pt_avg[2] * scale };
        for (int j = didx_d[i]; j < didx_d[i] + dnum_d[i]; ++j) {
          if (node_type(children_depth[j]) == kLeaf) continue;

          float dis = 0.0f;
          for (int c = 0; c < 3; ++c) {
            dis += (pt_depth[c * nnum_depth + j] - pt_avg1[c]) * n_avg[c];
          }
          dis = abs(dis);
          if (dis > distance_max1) distance_max1 = dis;
        }

        // the error from the averaged geometry to the original geometry
        float distance_max2 = -1;
        vector<float> vtx;
        intersect_cube(vtx, pt_avg.data(), pt_base, n_avg.data());
        if (vtx.empty()) distance_max2 = 5.0e10f; // !!! the degenerated case, ||n_avg|| == 0
        for (auto& v : vtx) v *= scale;           // !!! note the scale
        for (int k = 0; k < vtx.size() / 3; ++k) {
          // min
          float distance_min = 1.0e30f;
          for (int j = didx_d[i]; j < didx_d[i] + dnum_d[i]; ++j) {
            if (node_type(children_depth[j]) == kLeaf)  continue;
            float dis = 0.0f;
            for (int c = 0; c < 3; ++c) {
              float ptc = pt_depth[c * nnum_depth + j] - vtx[3 * k + c];
              dis += ptc * ptc;
            }
            dis = sqrtf(dis);
            if (dis < distance_min) distance_min = dis;
          }

          // max
          if (distance_min > distance_max2) distance_max2 = distance_min;
        }

        distance_err_d[i] = std::max<float>(distance_max2, distance_max1);
      }

      if (has_implicit) {

        //outfile << "\t node_d " << children_d[i] << std::endl;

        // calc uv-plane center
        Eigen::Vector3f plane_center;
        Eigen::Vector3f plane_normal; 
        for (int c = 0; c < 3; c++) {
          plane_center(c) = pt_d[c * nnum_d + i];
          plane_normal(c) = normal_d[c * nnum_d + i];
        }
        Eigen::MatrixXf R = polynomial::calc_rotation_matrix(plane_normal);

        //outfile << "\t plane center " << plane_center(0) << ", " << plane_center(1) << ", " << plane_center(2) << std::endl;

        Eigen::MatrixXf coef = polynomial::biquad_approximation(this, children_depth, didx_d[i], didx_d[i] + dnum_d[i], pts_scaled, scale, unique_idx, sorted_idx, R, plane_center, 1.0);

        for (int c = 0; c < 6; c++) {
          normal_d[(c+3) * nnum_d + i] = coef(c, 0);
        }

        normal_err_d[i] = 20000;
        distance_err_d[i] = 20000;
      }
    }
  }

  //outfile.close();
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
        for (int j = i_base; j < i_base + 8; ++j) {
          if (node_type(children_d[j]) == kLeaf) continue;
          l_avg[static_cast<int>(label_d[j])] += 1;
        }
        label_d[i] = static_cast<float>(std::distance(l_avg.begin(),
                    std::max_element(l_avg.begin(), l_avg.end())));
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
        if (abs(dis) < 1.0f) {
          //bool has_intersection = false;
          // make the voxel has no intersection with the current voxel
          unsigned int cube_cases = 0;
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
          float dd = abs(pti[0] - ptj[0]) + abs(pti[1] - ptj[1]) + abs(pti[2] - ptj[2]);
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
    vector<uint32>& keys = keys_[d];
    for (int i = 0; i < keys.size(); ++i) {
      // calc point
      uint32 k = keys[i], pt[3];
      compute_pt(pt, k, d);

      // compress
      unsigned char* ptr = reinterpret_cast<unsigned char*>(&key[idx]);
      ptr[0] = static_cast<unsigned char>(pt[0]);
      ptr[1] = static_cast<unsigned char>(pt[1]);
      ptr[2] = static_cast<unsigned char>(pt[2]);
      ptr[3] = static_cast<unsigned char>(d);

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

void Octree::unique_key(vector<uint32>& keys, vector<uint32>& idx) {
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
    vector<vector<uint32> > xyz;
    key_to_xyz(xyz);
    SERIALIZE_PROPERTY(uint32, OctreeInfo::kKey, xyz);
  } else {
    SERIALIZE_PROPERTY(uint32, OctreeInfo::kKey, keys_);
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

    vector<uint32> key;
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

void Octree::key_to_xyz(vector<vector<uint32> >& xyz) {
  const int depth = oct_info_.depth();
  const int channel = oct_info_.channel(OctreeInfo::kKey);
  xyz.resize(depth + 1);
  for (int d = 0; d <= depth; ++d) {
    int nnum = oct_info_.node_num(d);
    xyz[d].resize(nnum * channel, 0);
    uint32* xyz_d = xyz[d].data();
    for (int i = 0; i < nnum; ++i) {
      uint32 pt[3] = { 0, 0, 0 };
      compute_pt(pt, keys_[d][i], d);

      if (channel == 1) {
        unsigned char* ptr = reinterpret_cast<unsigned char*>(xyz_d + i);
        for (int c = 0; c < 3; ++c) {
          ptr[c] = static_cast<unsigned char>(pt[c]);
        }
      } else {
        unsigned short* ptr = reinterpret_cast<unsigned short*>(xyz_d + 2 * i);
        for (int c = 0; c < 3; ++c) {
          ptr[c] = static_cast<unsigned short>(pt[c]);
        }
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
          float t = abs(avg_normals_[d][i]) + abs(avg_normals_[d][nnum_d + i]) +
              abs(avg_normals_[d][2 * nnum_d + i]);
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
  const float kMul = rescale ? info_->bbox_max_width() / float(1 << depth) : 1.0f;
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
      float len = abs(n[0]) + abs(n[1]) + abs(n[2]);
      if (len == 0 && d != depth) continue;  // for adaptive octree
      node_pos(pt, i, d);

      for (int c = 0; c < 3; ++c) {
        normals.push_back(n[c]);
        pts.push_back(pt[c] * scale + bbmin[c]); // !!! note the scale and bbmin
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
    int depth_end) const {
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

      float n[3], pt[3], pt_ref[3], coef[6];
      node_normal(n, i, d);
      node_slim_coefficients(coef, i, d);
      float len = fabs(n[0]) + fabs(n[1]) + fabs(n[2]);
      if (len == 0) continue;             // if normal is zero cell has no surface
      node_pos(pt, i, d, pt_ref);
      node_dis_xyz(pt, i, d);

      Vector3f cell_base, surf_center, surf_normal;
      for (int c = 0; c < 3; ++c) {
        cell_base(c) = pt_ref[c] * cube_size;           // adjust cell base pos to global scale
        surf_center(c) = cell_base(c) + 0.5*cube_size;  // adjust surfel center to global scale
        surf_center(c) += pt[c] * cube_size;
        surf_normal(c) = n[c];
      }

      MatrixXf surf_coefs(6,1);
      for (int c = 0; c < 6; ++c) {
        surf_coefs(c,0) = coef[c];
      }
      
      // render surface
      polynomial2::sample_surface_along_normal_rt(&V, &F, cell_base, cube_size, 8, surf_center, surf_normal, surf_coefs);
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
    F[i] -= 1;
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
//    const uint32* keys_d = key_cpu(d);
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
//        float len = abs(n[0]) + abs(n[1]) + abs(n[2]);
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
//        if (abs(dis) < 1.0f) {
//          // make the voxel has no intersection with the current voxel
//          uint32 cube_cases = 0;
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


