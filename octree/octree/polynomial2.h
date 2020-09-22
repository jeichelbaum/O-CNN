#ifndef POLYNOMIAL2_H_
#define POLYNOMIAL2_H_

#include <iostream>
#include <limits>
#include <vector>

#include <eigen3/Eigen/Dense>

#include <points_info.h>
#include <points.h>
#include <points_parser.h>

#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>


using namespace std;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::VectorXf;
using Eigen::MatrixXf;

class polynomial2 {
 public:
    static float eval_quadric(Eigen::Vector3f point, Eigen::Vector3f center, float scale, Eigen::VectorXf coefs);

    static void visualize_quadric(vector<float>* verts, vector<int>* faces, Eigen::Vector3f base, float size, int n_sub,
            Eigen::Vector3f quadric_center, Eigen::VectorXf quadric_coefs);

    static float calc_taubin_dist(Eigen::Vector3f point, Eigen::Vector3f center, float scale, Eigen::VectorXf coefs);
};


class Polynomial2Approx {
    public:
        Polynomial2Approx(Points& point_cloud, const float* bbmin, const float mul);

        void init_parent_approx_tracking(int depth_max_);
        void set_well_approximated(int cur_depth, int* xyz);
        bool parent_well_approximated(int cur_depth, int* xyz);

        bool approx_surface(Vector3f cell_base, float cell_size, float support_radius, float error_threshold);
        bool check_coefs();

        int npt;
        const int THRESHOLD_MIN_NUM_POINTS = 6;

        Vector3f surf_center;
        VectorXf surf_coefs;

        float error_avg_points_surface_dist;
        float error_max_surface_points_dist;

    private:
        int SURFACE_SAMPLING_RESOLUTION = 5;
        float COEFS_CLAMP = 10.0;

        Points pts;
        pcl::PointCloud<pcl::PointXYZ>* cloud;
        pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>* octree;

        int depth_max;
        vector<int> nnum_all;
        vector<vector<bool>> ptr_well_approx;
};

#endif // POLYNOMIAL2_H_