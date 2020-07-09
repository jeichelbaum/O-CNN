#ifndef _CHAMFER_DIST_
#define _CHAMFER_DIST_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <ctime>
#include <nanoflann.hpp>

#include "points.h"

using namespace std;


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

class chamfer_dist {
  public: 
    static void closet_pts(vector<size_t>& idx, vector<float>& distance,
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
};


#endif // _CHAMFER_DIST_