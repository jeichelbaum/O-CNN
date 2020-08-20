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

    // ----- Coordinate System
    static MatrixXf calc_rotation_matrix(Vector3f surf_normal);

    // ----- Intersection
    static VectorXf surface_ray_intersection(Vector3f Roc, Vector3f Rrd, MatrixXf c); 
    static float surface_intersection_along_normal(Vector3f Roc, Vector3f Rrd, MatrixXf c);

    // ----- Sampling
    // sample surface in UV coordinates
    static void sample_surface(vector<float>* samples, Vector3f cell_base, float cell_size, int resolution, 
        Vector3f surf_center, Vector3f surf_normal, MatrixXf surf_coefs);

    // raytrace three axis aligned cube faces
    //static void sample_surface_rt(vector<float>* samples, Vector3f cube_base, float cube_size, int resolution, Vector3f surf_center, MatrixXf R, MatrixXf surf_coefs);
    
    // raytrace normal plane
    static void sample_surface_along_normal_rt(vector<float>* samples, vector<int>* faces, Vector3f cube_base, float cube_size, int resolution, Vector3f surf_center, Vector3f surf_normal, MatrixXf surf_coefs);

    // ----- Error
    static float calc_avg_distance_to_surface_rt(MatrixXf points, MatrixXf normals, MatrixXf coefs);

};


class Polynomial2Approx {
    public:
        Polynomial2Approx(Points& point_cloud, const float* bbmin, const float mul);

        void init_parent_approx_tracking(int depth_max_);
        void set_well_approximated(int cur_depth, int* xyz);
        bool parent_well_approximated(int cur_depth, int* xyz);

        bool approx_surface(Vector3f cell_base, float cell_size, float support_radius, float error_threshold);

        int npt;
        Vector3f surf_center;
        Vector3f surf_normal;
        MatrixXf surf_coefs;

        float error_avg_points_surface_dist;
        float error_max_surface_points_dist;

    private:
        int THRESHOLD_MIN_NUM_POINTS = 6;
        int SURFACE_SAMPLING_RESOLUTION = 5;

        Points pts;
        pcl::PointCloud<pcl::PointXYZ>* cloud;
        pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>* octree;

        int depth_max;
        vector<int> nnum_all;
        vector<vector<bool>> ptr_well_approx;
};

#endif // POLYNOMIAL2_H_