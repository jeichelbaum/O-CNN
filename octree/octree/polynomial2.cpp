#include "polynomial2.h"

MatrixXf polynomial2::calc_rotation_matrix(Vector3f surf_normal) {
    Vector3f global_normal = {0, 0, 1};

    if (surf_normal == global_normal) { return MatrixXf::Identity(3,3); }
    if (surf_normal == -1*global_normal) { return -1*MatrixXf::Identity(3,3); }

    Vector3f v = surf_normal.cross(global_normal);
    float s = sqrtf(v.squaredNorm());
    float c = surf_normal.dot(global_normal);

    MatrixXf v_skew(3, 3);
    v_skew <<   0, -v(2), v(1),
                v(2), 0, -v(0),
                -v(1), v(0), 0;

    MatrixXf R = MatrixXf::Identity(3,3);
    R += v_skew + v_skew*v_skew * ((1-c) / (s*s));
    return R;
}




// Roc: Ray origin in surface coordinate system
// Rrd: Ray direction in surface coordinate system
// c: surface coefficients
float polynomial2::surface_intersection_along_normal(Vector3f Roc, Vector3f Rrd, MatrixXf c) {
    // intersection equation: l^2*x(0) + l*x(1) + x(2)
     Vector2f x = {
        c(1)*Rrd(0) + c(2)*Rrd(1) + c(3)*2*Rrd(0)*Roc(0) + c(4)*Rrd(0)*Roc(1) + c(4)*Rrd(1)*Roc(0) + c(5)*2*Rrd(1)*Roc(1) - Rrd(2),
        c(0) + c(1)*Roc(0) + c(2)*Roc(1) + c(3)*Roc(0)*Roc(0) + c(4)*Roc(0)*Roc(1) + c(5)*Roc(1)*Roc(1) - Roc(2)
    };

    return (-x(1) / x(0));
}

// Roc: Ray origin in surface coordinate system
// Rrd: Ray direction in surface coordinate system
// c: surface coefficients
VectorXf polynomial2::surface_ray_intersection(Vector3f Roc, Vector3f Rrd, MatrixXf c) {
    // intersection equation: l^2*x(0) + l*x(1) + x(2)
    Vector3f x = {
        c(3)*Rrd(0)*Rrd(0) + c(4)*Rrd(0)*Rrd(1) + c(5)*Rrd(1)*Rrd(1),
        c(1)*Rrd(0) + c(2)*Rrd(1) + c(3)*2*Rrd(0)*Roc(0) + c(4)*Rrd(0)*Roc(1) + c(4)*Rrd(1)*Roc(0) + c(5)*2*Rrd(1)*Roc(1) - Rrd(2),
        c(0) + c(1)*Roc(0) + c(2)*Roc(1) + c(3)*Roc(0)*Roc(0) + c(4)*Roc(0)*Roc(1) + c(5)*Roc(1)*Roc(1) - Roc(2)
    };


    // solve linear
    if (x(0) == 0) {
        if (x(1) != 0) {    // 1 intersection
            VectorXf res(1);
            res << (-x(2) / x(1));
            return res;
        }                   // 0 intersection
    }
    // solve quadratic
    else {
        float b24ac = x(1) * x(1) - 4 * x(0) * x(2);
        if (b24ac >= 0) {   // 2 intersection
            VectorXf res(2);
            res <<  (-x(1) + sqrtf(b24ac)) / (2*x(0)),
                    (-x(1) - sqrtf(b24ac)) / (2*x(0));
            return res;
        }                   // 0 intersection
    }

    return VectorXf(0);
}


void polynomial2::sample_surface_along_normal_rt(vector<float>* samples, vector<int>* faces, Vector3f cube_base, float cube_size, int resolution, Vector3f surf_center, Vector3f surf_normal, MatrixXf surf_coefs) 
{
    auto steps = VectorXf::LinSpaced(resolution, -0.5*cube_size, 0.5*cube_size);
    auto R = polynomial2::calc_rotation_matrix(surf_normal);
    Vector3f axis1 = R.inverse() * MatrixXf::Identity(3,3).row(0).transpose();
    Vector3f axis2 = R.inverse() * MatrixXf::Identity(3,3).row(1).transpose();
    Vector3f ray_dir = -surf_normal;

    Vector3f cell_center = cube_base + 0.5*Vector3f(cube_size, cube_size, cube_size);

    int vi = samples->size()/3+1;
    MatrixXf vidx = MatrixXf::Zero(resolution, resolution);

    Vector3f pos = Vector3f::Zero();
    for (int s1 = 0; s1 < resolution; s1++) {
        for (int s2 = 0; s2 < resolution; s2++) {
            pos = cell_center + steps[s1]*axis1 + steps[s2]*axis2;
            auto Roc = R * (pos - surf_center);
            auto Rrd = R * ray_dir;
            auto roots = surface_intersection_along_normal(Roc, Rrd, surf_coefs);
            pos = pos + roots*ray_dir;

            // in cell
            if (cube_base(0) <= pos(0) && pos(0) <= cube_base(0) + cube_size
                && cube_base(1) <= pos(1) && pos(1) <= cube_base(1) + cube_size
                && cube_base(2) <= pos(2) && pos(2) <= cube_base(2) + cube_size) {
                    for (int c=0;c<3;c++) { samples->push_back(pos(c)); }

                    // ------------ ADD FACES
                    vidx(s1, s2) = vi;
                    vi++;

                    if (faces != NULL && s1 > 0 && s2 > 0) {
                        if (vidx(s1, s2-1) != 0 && vidx(s1-1, s2-1) != 0) {
                            faces->push_back(vidx(s1, s2));
                            faces->push_back(vidx(s1-1, s2-1));
                            faces->push_back(vidx(s1, s2-1));
                        }

                        if (vidx(s1-1, s2) != 0 && vidx(s1-1, s2-1) != 0) {
                            faces->push_back(vidx(s1, s2));
                            faces->push_back(vidx(s1-1, s2));
                            faces->push_back(vidx(s1-1, s2-1));
                        }
                    }
            }
        }
    }
}

void polynomial2::sample_surface(vector<float>* samples, Vector3f cell_base, float cell_size, int resolution, 
        Vector3f surf_center, Vector3f surf_normal, MatrixXf surf_coefs)
{
    auto R = calc_rotation_matrix(surf_normal);
    auto Rinv = R.inverse();
    Vector3f axis1 = Vector3f(1.0, 0, 0);
    Vector3f axis2 = Vector3f(0, 1.0, 0);

    int npt = 0;
    samples->resize(3*resolution*resolution);
    auto steps = VectorXf::LinSpaced(resolution, -0.5*cell_size, 0.5*cell_size);

    for (int i1 = 0; i1 < resolution; i1++) {
        for (int i2 = 0; i2 < resolution; i2++) {
            Vector3f point = steps[i1]*axis1 + steps[i2]*axis2;
            point(2) = surf_coefs(0) + surf_coefs(1)*point(0) + surf_coefs(2)*point(1) + 
                    surf_coefs(3)*point(0)*point(0) + surf_coefs(4)*point(0)*point(1) + surf_coefs(5)*point(1)*point(1);
            point = Rinv * point + surf_center;

            for (int c=0; c<3; c++) { 
                (*samples)[npt*3+c] = point(c);
            }
            npt++;
        }
    }
}




/*void polynomial2::sample_surface_rt(vector<float>* samples, Vector3f cube_base, float cube_size, int resolution, Vector3f surf_center, MatrixXf R, MatrixXf surf_coefs) 
{
    // SAMPLE TOP CUBE FACE
    for (int face = 0; face < 1; face++) {
        auto steps = VectorXf::LinSpaced(resolution, 0, cube_size);
        Vector3f axis1 = MatrixXf::Identity(3,3).row((face+0) % 3);
        Vector3f axis2 = MatrixXf::Identity(3,3).row((face+1) % 3);
        Vector3f ray_dir = MatrixXf::Identity(3,3).row((face+2) % 3);
        auto Rrd = R*ray_dir;

        Vector3f pos = Vector3f::Zero();
        for (int s1 = 0; s1 < resolution; s1++) {
            for (int s2 = 0; s2 < resolution; s2++) {
                pos = cube_base + steps[s1]*axis1 + steps[s2]*axis2;

                for (int c=0;c<3;c++) { samples->push_back(pos(c)); }

                auto Roc = R * (pos - surf_center);
                auto roots = surface_ray_intersection(Roc, Rrd, surf_coefs);
                for (int r = 0; r < roots.rows(); r++) {
                    pos = pos + ray_dir*roots(r); // intersection point in real world coords

                    // if in cube then push back
                    if (cube_base(0) <= pos(0) && pos(0) <= cube_base(0) + cube_size
                        && cube_base(1) <= pos(1) && pos(1) <= cube_base(1) + cube_size
                        && cube_base(2) <= pos(2) && pos(2) <= cube_base(2) + cube_size) {
                        samples->push_back(pos(0));
                        samples->push_back(pos(1));
                        samples->push_back(pos(2));
                    }
                }
        }
        }
    }
}*/



// points (Nx3):    in surface coordinates
// normals (Nx3):   in surface orientation
// coefs (6x1):     surface coefficients 
float polynomial2::calc_avg_distance_to_surface_rt(MatrixXf points, MatrixXf normals, MatrixXf coefs) 
{
    int num = 0;
    float dist = 0;
    for (int i = 0; i < points.rows(); i++) {
        auto intersects = surface_ray_intersection(points.row(i), normals.row(i), coefs); 
        if (intersects.rows() > 0) {
            dist += intersects.array().abs().minCoeff();
            num++;  
        }
    }
    return dist / num;
}



Polynomial2Approx::Polynomial2Approx (Points& point_cloud) {
    // store ref to source points and normals
    this->pts = point_cloud;

    // init point cloud
    cloud = (new pcl::PointCloud<pcl::PointXYZ>);
    cloud->width = pts.info().pt_num();
    cloud->height = 1;
    cloud->points.resize (cloud->width * cloud->height);

    // copy points into point cloud
    for (std::size_t i = 0; i < cloud->size (); ++i)
    {
        int i3 = i*3;
        (*cloud)[i].x = pts.points()[i*3+0];
        (*cloud)[i].y = pts.points()[i*3+1];
        (*cloud)[i].z = pts.points()[i*3+2];
    }

    // populate octree
    octree = (new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(1.0f));
    octree->setInputCloud ((pcl::PointCloud<pcl::PointXYZ>::Ptr)cloud);
    octree->addPointsFromInputCloud ();
}


// init memory for approximation tracker
void Polynomial2Approx::init_parent_approx_tracking(int depth_max_) {
    depth_max = depth_max_;
    nnum_all.resize((depth_max+1)*3);
    ptr_well_approx.resize((depth_max+1));

    for (int d=0; d <= depth_max; d++) {
        nnum_all[d*3] = 1 << d;                     // num cubes along 1 axis
        nnum_all[d*3+1] = 1 << 2*d;                 // num cubes along 2 axis
        nnum_all[d*3+2] = 1 << 3*d;                 // num cubes along 3 axis
        ptr_well_approx[d].resize(nnum_all[d*3+2]); // resize bool vector to keep track of approx status
        std::fill(ptr_well_approx[d].begin(), ptr_well_approx[d].end(), false);
    }
}

// set all children to well approximated
void Polynomial2Approx::set_well_approximated(int cur_depth, int* xyz)
{
    for (int d = cur_depth+1; d<= depth_max; d++) {
        int num = 1 << (d - cur_depth);
        for (int x = 0; x < num; x++) {
            for (int y = 0; y < num; y++) {
               for (int z = 0; z < num; z++) {
                    int node_id = (xyz[0]*num+x) * nnum_all[d*3+1] + (xyz[1]*num+y) * nnum_all[d*3] + xyz[2]*num + z;
                    ptr_well_approx[d][node_id] = true;
               }
            }
        }
    }
}

// return true if parent already well approximated
bool Polynomial2Approx::parent_well_approximated(int cur_depth, int* xyz)
{
    int node_id = xyz[0] * nnum_all[cur_depth*3+1] + xyz[1] * nnum_all[cur_depth*3] + xyz[2];
    return ptr_well_approx[cur_depth][node_id];
}

// returns bool: surface_well_approximated
bool Polynomial2Approx::approx_surface(Vector3f cell_base, float cell_size, float support_radius, float error_threshold)
{
    // reset output variables
    npt = 0;
    error_max_surface_points_dist = -1;
    error_avg_points_surface_dist = -1;

    // search in cell radius
    pcl::PointXYZ cell_center;
    cell_center.x = cell_base(0) + 0.5*cell_size;
    cell_center.y = cell_base(1) + 0.5*cell_size;
    cell_center.z = cell_base(2) + 0.5*cell_size;

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    if (octree->radiusSearch (cell_center, support_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) { 
        npt = pointIdxRadiusSearch.size();

        // approximate only if enough points in support radius
        if (npt < THRESHOLD_MIN_NUM_POINTS) {
            return false;
        }

        // ----------- COPY

        // copy points and normals into Eigen Matrix shape(Nx3)
        Eigen::MatrixXf points = Eigen::MatrixXf::Zero(npt, 3);
        Eigen::MatrixXf normals = Eigen::MatrixXf::Zero(npt, 3);
        for (int p = 0; p < npt; p++) {
            for (int j=0; j<3; j++) {
                points(p, j) = pts.points()[pointIdxRadiusSearch[p]*3+j];
                normals(p, j) = pts.normal()[pointIdxRadiusSearch[p]*3+j];
            }
        }

        // ----------- APPROX

        // calc avg point, avg normal and rotation matrix
        surf_center = points.colwise().sum() / npt;
        surf_normal = (normals.colwise().sum()).normalized();
        Eigen::MatrixXf R = polynomial2::calc_rotation_matrix(surf_normal);

        // transform points and normals into surface coordinate system
        points = (R * (points.transpose().colwise() - surf_center)).transpose();
        normals = (R * normals.transpose()).transpose();

        // calculate weight based on distance, normalized to cube diagonal
        Eigen::VectorXf w = points.rowwise().norm() / (2*support_radius);
        w = 1.0 - (w.array() * w.array());

        // biquad polynomial matrix
        Eigen::MatrixXf b = Eigen::MatrixXf::Ones(npt, 6);
        b.col(1) = points.col(0);
        b.col(2) = points.col(1);
        b.col(3) = (points.col(0).array() * points.col(0).array());
        b.col(4) = (points.col(0).array() * points.col(1).array());
        b.col(5) = (points.col(1).array() * points.col(1).array());

        // shape(6x6) B = sum[wi * bi * bi.T]_i
        Eigen::MatrixXf B = (b.array().colwise() * w.array()).matrix().transpose() * b;

        // bf = sum[wi * bi * f(ui,vi)]_i
        Eigen::MatrixXf bf = ((w.transpose().array() * points.col(2).transpose().array()).matrix() * b).transpose(); 

        // invert to get surface coefficients
        surf_coefs = B.inverse() * bf;

        // ----------- ERROR: SURF -> POINTS
        
        // sample surface within octree cell
        vector<float> surf_edge_samples;
        polynomial2::sample_surface_along_normal_rt(&surf_edge_samples, NULL, cell_base, cell_size, SURFACE_SAMPLING_RESOLUTION, surf_center, surf_normal, surf_coefs);

        // calc max distance from surface sample to point cloud
        float surface_point_dist = surf_edge_samples.size() == 0 ? numeric_limits<float>::max() : -1;
        std::vector<int> pointIdxNKNSearch;
        std::vector<float> pointNKNSquaredDistance;
        for (int p = 0; p < surf_edge_samples.size(); p+=3) {
            pcl::PointXYZ edge_sample (surf_edge_samples[p],surf_edge_samples[p+1],surf_edge_samples[p+2]);
            if (octree->nearestKSearch (edge_sample, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                surface_point_dist = max(surface_point_dist, pointNKNSquaredDistance[0]);
            } 
        }
        error_max_surface_points_dist = sqrtf(surface_point_dist);
        
        // return false if surface to point cloud distance is bigger than error threshold
        if (error_max_surface_points_dist > error_threshold) {
            return false;
        }

        // ----------- ERROR: POINTS -> SURF

        // point to surface distance along point normal
        error_avg_points_surface_dist = polynomial2::calc_avg_distance_to_surface_rt(points, normals, surf_coefs);

        // return well approximated if all errors are below threshold
        return (error_avg_points_surface_dist < error_threshold);
    }

    return false;
}