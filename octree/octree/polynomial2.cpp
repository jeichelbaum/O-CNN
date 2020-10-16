#include "polynomial2.h"

#include "marching_cube.h"


float polynomial2::eval_quadric(Eigen::Vector3f point, Eigen::Vector3f center, float scale, Eigen::VectorXf coefs) {
    auto p = (point - center) / scale;
    return coefs(0) + coefs(1) * p(0) + coefs(2) * p(1) + coefs(3) * p(2)
        + coefs(4) * p(0)*p(0) + coefs(5) * p(0)*p(1) + coefs(6) * p(0)*p(2)
        + coefs(7) * p(1)*p(1) + coefs(8) * p(1)*p(2) + coefs(9) * p(2)*p(2); 
}

void polynomial2::visualize_quadric(vector<float>* verts, vector<int>* faces, Eigen::Vector3f base, float size, int n_sub,
    Eigen::Vector3f quadric_center, Eigen::VectorXf quadric_coefs)     
{
    float step = size / float(n_sub);
    float pt_ref[3] = {0};

    // compute function value for corners
    for (int x = 0; x < n_sub; x++) {
        pt_ref[0] = base(0) + x * step;
        for (int y = 0; y < n_sub; y++) {
            pt_ref[1] = base(1) + y * step;
            for (int z = 0; z < n_sub; z++) {
                pt_ref[2] = base(2) + z * step;

                Eigen::Vector3f p;
                float fval[8] = {0};
                for (int k = 0; k < 8; ++k) {
                    for (int j = 0; j < 3; ++j) {
                        p(j) = MarchingCube::corner_[k][j] * step + pt_ref[j];
                    }

                    // calcualate function value for imp3
                    fval[k] = polynomial2::eval_quadric(p, quadric_center, size, quadric_coefs);
                }

                // marching cube
                int vid = verts->size() / 3;
                float ref[3] = {0,0,0};
                MarchingCube mcube(fval, 0, ref, vid);
                mcube.contouring(*verts, *faces);
                for (int i = vid; i < verts->size() / 3; i++) {
                    for (int c = 0; c < 3; c++) { (*verts)[i*3+c] = (*verts)[i*3+c]*step + pt_ref[c]; }
                }
            }   
        }   
    }
}


float polynomial2::calc_taubin_dist(Eigen::Vector3f point, Eigen::Vector3f center, float scale, Eigen::VectorXf coefs)
{
    auto p = (point - center) / scale;
    float val = polynomial2::eval_quadric(point, center, scale, coefs);

    Eigen::Vector3f grad = Eigen::Vector3f(
        coefs(1) + 2*coefs(4)*p(0) + coefs(5)*p(1) + coefs(6)*p(2),
        coefs(2) + coefs(5)*p(0) + 2*coefs(7)*p(1) + coefs(8)*p(2),
        coefs(3) + coefs(6)*p(0) + coefs(8)*p(1) + 2*coefs(9)*p(2)
    );

    return fabs(val) / grad.norm();
}

float polynomial2::calc_taubin_dist_fast(Eigen::MatrixXf points_local, Eigen::VectorXf coefs) 
{
    int npt = points_local.rows();
    Eigen::ArrayXf ones = Eigen::ArrayXf::Ones(npt);

    // ---- polynomial
    Eigen::MatrixXf b = Eigen::MatrixXf::Zero(npt, 10);

    //  1 + x + y + z
    b.col(0) = coefs[0] * ones;
    b.col(1) = coefs[1] * points_local.col(0).array();
    b.col(2) = coefs[2] * points_local.col(1).array();
    b.col(3) = coefs[3] * points_local.col(2).array();

    // xx + xy + xz
    b.col(4) = coefs[4] * points_local.col(0).array() * points_local.col(0).array();
    b.col(5) = coefs[5] * points_local.col(0).array() * points_local.col(1).array();
    b.col(6) = coefs[6] * points_local.col(0).array() * points_local.col(2).array();

    // yy + yz + zz
    b.col(7) = coefs[7] * points_local.col(1).array() * points_local.col(1).array();
    b.col(8) = coefs[8] * points_local.col(1).array() * points_local.col(2).array();
    b.col(9) = coefs[9] * points_local.col(2).array() * points_local.col(2).array();

    // sum polynomial components
    auto f = b.rowwise().sum().rowwise().norm();

    // ---- gradient
    Eigen::MatrixXf grad = Eigen::MatrixXf::Zero(npt, 3);

    // dx = 1 + 2x + y + z
    grad.col(0) = coefs[1] * ones + 2 * coefs[4] * points_local.col(0).array()
                    + coefs[5] * points_local.col(1).array() + coefs[6] * points_local.col(2).array();

    // dy = 1 + x + 2y + z
    grad.col(1) = coefs[2] * ones + coefs[5] * points_local.col(0).array()
                    + 2 * coefs[7] * points_local.col(1).array() + coefs[8] * points_local.col(2).array();

    // dz = 1 + x + y + 2z
    grad.col(2) = coefs[3] * ones + coefs[6] * points_local.col(0).array()
                    + coefs[8] * points_local.col(1).array() + 2 * coefs[9] * points_local.col(2).array();

    // rowwise norm
    auto g = (grad.rowwise().norm());

    // evaluate taubin distances
    auto t = f.col(0).array() / g.col(0).array();

    /*if (npt == 3) {
        std::cout << "-> poly" << std::endl;
        std::cout << b << std::endl;
        std::cout << f.transpose() << std::endl;
        std::cout << "-> grad" << std::endl;
        std::cout << grad << std::endl;
        std::cout << g << std::endl;
        std::cout << "-> taubin" << std::endl;
        std::cout << t << std::endl;
    }*/

    // select max taubin distance
    return t.maxCoeff();
    // avg taubin distance
    return t.sum() / npt;

}

Polynomial2Approx::Polynomial2Approx (Points& point_cloud, const float* bbmin, const float mul) {
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
        (*cloud)[i].x = (pts.points()[i*3+0] - bbmin[0]) * mul;
        (*cloud)[i].y = (pts.points()[i*3+1] - bbmin[1]) * mul;
        (*cloud)[i].z = (pts.points()[i*3+2] - bbmin[2]) * mul;
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
bool Polynomial2Approx::approx_surface(Vector3f cell_base, float cell_size, float support_radius, float error_p2q, float error_q2p)
{
    // reset output variables
    npt = 0;
    error_max_surface_points_dist = numeric_limits<float>::max();
    error_avg_points_surface_dist = numeric_limits<float>::max();

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


        auto start_copy = system_clock::now();
        // copy points and normals into Eigen Matrix shape(Nx3)
        Eigen::MatrixXf points = Eigen::MatrixXf::Zero(npt, 3);
        Eigen::MatrixXf normals = Eigen::MatrixXf::Zero(npt, 3);
        for (int p = 0; p < npt; p++) {

            points(p, 0) = (*cloud)[pointIdxRadiusSearch[p]].x;
            points(p, 1) = (*cloud)[pointIdxRadiusSearch[p]].y;
            points(p, 2) = (*cloud)[pointIdxRadiusSearch[p]].z;

            for (int j=0; j<3; j++) {
                normals(p, j) = pts.normal()[pointIdxRadiusSearch[p]*3+j];
            }
        }
        time_copy += duration_cast<microseconds>(high_resolution_clock::now() - start_copy).count();


        auto start_approx = system_clock::now();

        //          center point cloud
        surf_center = cell_base + 0.5 * Vector3f(cell_size, cell_size, cell_size);
        points = (points.transpose().colwise() - surf_center).transpose() / cell_size;
        // calculate weight based on distance, normalized to cube diagonal
        Eigen::VectorXf w = points.rowwise().norm() / (support_radius / cell_size);
        w = 1.0 - (w.array() * w.array());

        // quadric design matrix
        int n_constraints = 4;
        Eigen::MatrixXf b = Eigen::MatrixXf::Zero(npt*n_constraints, 10);
        Eigen::VectorXf f = Eigen::VectorXf::Zero(npt*n_constraints);
        Eigen::VectorXf wN = Eigen::VectorXf::Ones(npt*n_constraints);

        for (std::size_t i = 0; i < npt; ++i)
        {
            int iN = i*n_constraints;

            // weights
            wN(iN, 0) = 1.0;
            wN(iN+1, 0) = wN(iN+2, 0) = wN(iN+3, 0) = 0.33;

            // function values
            f(iN) = 0;
        
            f(iN+1) = normals(i, 0);
            f(iN+2) = normals(i, 1);
            f(iN+3) = normals(i, 2);

            // design matrix    -   quadric
            Eigen::Vector3f p = points.row(i);

            b(iN, 0) = 1;                               // 1
            b(iN, 1) = p(0);                    // x
            b(iN, 2) = p(1);                    // y
            b(iN, 3) = p(2);                    // z

            b(iN, 4) = p(0)*p( 0);       // xx
            b(iN, 5) = p(0)*p(1);       // xy
            b(iN, 6) = p(0)*p(2);       // xz

            b(iN, 7) = p(1)*p(1);       // yy
            b(iN, 8) = p(1)*p(2);       // yz
            b(iN, 9) = p(2)*p(2);       // zz

            
            // design matrix    -   constraint px
            b(iN+1, 1) = 1;
            b(iN+1, 3) = 2*points(i, 0);
            b(iN+1, 4) = points(i, 1);
            b(iN+1, 5) = points(i, 2);

            // design matrix    -   constraint py
            b(iN+2, 2) = 1;
            b(iN+2, 5) = points(i, 0);
            b(iN+2, 7) = 2*points(i, 1);
            b(iN+2, 8) = points(i, 2);

            // design matrix    -   constraint pz
            b(iN+3, 3) = 1;
            b(iN+3, 6) = points(i, 0);
            b(iN+3, 8) = points(i, 1);
            b(iN+3, 9) = 2*points(i, 2);
        }


        // shape(6x6) B = sum[wi * bi * bi.T]_i
        Eigen::MatrixXf B = (b.array().colwise() * wN.array()).matrix().transpose() * b;

        // bf = sum[wi * bi * f(ui,vi)]_i
        Eigen::MatrixXf bf = ((wN.transpose().array() * f.transpose().array()).matrix() * b).transpose(); 

        // invert to get surface coefficients
        //surf_coefs = B.inverse() * bf;        // never invert matrices
        surf_coefs = B.colPivHouseholderQr().solve(bf);
        if (!check_coefs()) { return false; }     
        time_approx += duration_cast<microseconds>(high_resolution_clock::now() - start_approx).count();


        // skip error calculation for last layer
        if (cell_size == 1.0) {
            return true;
        }

        auto start_taubin = system_clock::now();

        /*Eigen::MatrixXf testp = Eigen::MatrixXf::Ones(3, 3);
        testp(0,0) = 0.0; testp(0,1) = -1.0; testp(0,2) = 0.0; 
        testp(1,0) = -1.0; testp(1,1) = 0.0; testp(1,2) = 0.5; 
        testp(2,0) = 2.0; testp(2,1) = 0.0; testp(2,2) = 0.0; 
        Eigen::VectorXf testc = Eigen::VectorXf::Ones(10);
        testc(4) = testc(5) = testc(6) = testc(7) = testc(8) = -1.0;*/

        // calc point2surface via taubin distance
        // return false if surface to point cloud distance is bigger than error threshold
        error_avg_points_surface_dist = polynomial2::calc_taubin_dist_fast(points, surf_coefs);

        /*for (std::size_t i = 0; i < npt; ++i) {
            error_avg_points_surface_dist += polynomial2::calc_taubin_dist(points.row(i), surf_center, cell_size, surf_coefs) / npt;
        }*/

        time_taubin += duration_cast<microseconds>(high_resolution_clock::now() - start_taubin).count();
        if (error_avg_points_surface_dist > error_p2q) { return false; }

        auto start_mc = system_clock::now();
        // calc surface2points
        vector<float> surf_edge_samples;
        vector<int> faces;
        polynomial2::visualize_quadric(&surf_edge_samples, &faces, cell_base, cell_size, SURFACE_SAMPLING_RESOLUTION, surf_center, surf_coefs);

        // calc max distance from surface sample to point cloud
        float surface_point_dist = surf_edge_samples.size() == 0 ? numeric_limits<float>::max() : 0;
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
        time_mc += duration_cast<microseconds>(high_resolution_clock::now() - start_mc).count();


        // return well approximated if all errors are below threshold
        return (error_max_surface_points_dist <= error_q2p);
    }

    return false;
}

// check for NaN or out of clamp surf_coefs
bool Polynomial2Approx::check_coefs() {
    bool result_ok = true;
    for (int c = 0; c < 6; c++) {
        if (surf_coefs(c, 0) <  -COEFS_CLAMP || COEFS_CLAMP < surf_coefs(c, 0) || surf_coefs(c, 0) != surf_coefs(c, 0)) {
            result_ok = false;
        }
    }    

    // set coefs to zero if result bad
    if (!result_ok) {
        for (int c = 0; c < 6; c++) {
            surf_coefs(c,0) = 0;
        }
    }

    return result_ok;
}