#include "polynomial.h"

int polynomial::num_points(OctreeParser* octree, const vector<int>& children_depth, int cstart, int cend, 
    const vector<OctreeParser::uint32>& unique_idx, const vector<OctreeParser::uint32>& sorted_idx)
{
    int num_points = 0;

    // iterate over all nodes from final layer contained in cube
    for (int j = cstart; j < cend; ++j) {
        if (octree->node_type(children_depth[j]) == OctreeParser::NodeType::kLeaf) continue;

        int t = children_depth[j];
        for (int l = unique_idx[t]; l < unique_idx[t + 1]; l++) {
            num_points ++;
        }
    }

    return num_points;
}

Vector3f polynomial::avg_point(OctreeParser* octree, const vector<int>& children_depth, int cstart, int cend, 
    const vector<float>& pts_scaled, const int num_points, const vector<OctreeParser::uint32>& unique_idx, const vector<OctreeParser::uint32>& sorted_idx)
{
    Vector3f avg = {0, 0, 0};

    // iterate over all nodes from final layer contained in cube
    for (int j = cstart; j < cend; ++j) {
        // TODO WHY?!
        if (octree->node_type(children_depth[j]) == OctreeParser::NodeType::kLeaf) continue;

        // for each point in node at finest level
        int t = children_depth[j];
        for (int l = unique_idx[t]; l < unique_idx[t + 1]; l++) {
            
            // get normal and sum to average
            for (int c = 0; c < 3; c++) {
                avg(c) += pts_scaled[3*sorted_idx[l] + c];
            }
        }
    }

    // divide by number normals
    return avg / float(num_points);
}

Vector3f polynomial::avg_normal(OctreeParser* octree, const vector<int>& children_depth, int cstart, int cend, 
    const float* normals, const int num_normals, const vector<OctreeParser::uint32>& unique_idx, const vector<OctreeParser::uint32>& sorted_idx) 
{
    Vector3f avg = {0, 0, 0};

    // iterate over all nodes from final layer contained in cube
    for (int j = cstart; j < cend; ++j) {
        // TODO WHY?!
        if (octree->node_type(children_depth[j]) == OctreeParser::NodeType::kLeaf) continue;

        // for each point in node at finest level
        int t = children_depth[j];
        for (int l = unique_idx[t]; l < unique_idx[t + 1]; l++) {
            
            // get normal and sum to average
            for (int c = 0; c < 3; c++) {
                avg(c) += normals[3*sorted_idx[l] + c];
            }
        }
    }

    // divide by number normals
    return avg / float(num_normals);
}


MatrixXf polynomial::calc_rotation_matrix(Vector3f norm1) {
    Vector3f norm2 = {0, 0, 1};

    if (norm1 == norm2) { return MatrixXf::Identity(3,3); }
    if (norm1 == -1*norm2) { return -1*MatrixXf::Identity(3,3); }

    Vector3f v = norm1.cross(norm2);
    float s = sqrt(v.squaredNorm());
    float c = norm1.dot(norm2);

    MatrixXf v_skew(3, 3);
    v_skew <<   0, -v(2), v(1),
                v(2), 0, -v(0),
                -v(1), v(0), 0;

    MatrixXf R = MatrixXf::Identity(3,3);
    R += v_skew + v_skew*v_skew * ((1-c) / (s*s));
    return R;
}

MatrixXf polynomial::biquad(float u, float v) {
    MatrixXf poly(6, 1);
    poly << 1., u, v, u*u, u*v, v*v;
    return poly;
}

MatrixXf polynomial::triquad(Vector3f p) {
    MatrixXf poly(10, 1);
    poly << 1., p(0), p(1), p(2), p(0)*p(0), p(0)*p(1), p(0)*p(2), p(1)*p(1), p(1)*p(2), p(2)*p(2);
    return poly;
}

MatrixXf polynomial::biquad_approximation(const vector<float>& pts_scaled, const vector<OctreeParser::uint32>& sorted_idx,
    int jstart, int jend, MatrixXf R, Vector3f plane_center, float support_radius) {

    MatrixXf B = MatrixXf::Zero(6,6);
    MatrixXf b = MatrixXf::Zero(6,1);
    MatrixXf bf = MatrixXf::Zero(6,1);
    MatrixXf p = MatrixXf::Zero(3,1);

    // todo : think if matrix notation is possible instead of for loop
    float w;

    for ( int j = jstart; j < jend; j++) {
      // local coordinate system
      p << pts_scaled[3*sorted_idx[j]], pts_scaled[3*sorted_idx[j] + 1], pts_scaled[3*sorted_idx[j] + 2];
      p = p - plane_center;
      w = p.norm() / support_radius; // calculate distance - support radius is always 1.0 unless there is overlap then equal radius
      p = R * p;

      // polynomial for each point
      b = biquad(p(0, 0), p(1, 0));
      w = 1.0f - (w*w);

      B += (w*b) * b.transpose(); 
      bf += w * b * p(2, 0);
    }

    MatrixXf c(6,1);
    c = B.inverse() * bf;
    return c;
}

MatrixXf polynomial::biquad_approximation(OctreeParser* octree, const vector<int>& children_depth, int cstart, int cend, 
    const vector<float>& pts_scaled, float scale_factor, const vector<OctreeParser::uint32>& unique_idx, const vector<OctreeParser  ::uint32>& sorted_idx, 
    MatrixXf R, Vector3f plane_center, float support_radius) {

    MatrixXf B = MatrixXf::Zero(6,6);
    MatrixXf b = MatrixXf::Zero(6,1);
    MatrixXf bf = MatrixXf::Zero(6,1);
    MatrixXf p = MatrixXf::Zero(3,1);

    float w;

    // iterate over all nodes from final layer contained in cube
    for (int j = cstart; j < cend; ++j) {
        if (octree->node_type(children_depth[j]) == OctreeParser::NodeType::kLeaf) continue;

        int t = children_depth[j];

        for (int l = unique_idx[t]; l < unique_idx[t + 1]; l++) {

            // local coordinate system
            p << pts_scaled[3*sorted_idx[l]] / scale_factor, pts_scaled[3*sorted_idx[l] + 1] / scale_factor, pts_scaled[3*sorted_idx[l] + 2] / scale_factor;
            p = p - plane_center;
            w = p.norm() / support_radius; // calculate distance - support radius is always 1.0 unless there is overlap then equal radius
            p = R * p;

            // polynomial for each point
            b = biquad(p(0, 0), p(1, 0));
            w = 1.0f - (w*w);

            B += (w*b) * b.transpose(); 
            bf += w * b * p(2, 0);
        }
    }

    MatrixXf c(6,1);
    c = B.inverse() * bf;
    return c;
}

float polynomial::biquad_approximation_error(OctreeParser* octree, const vector<int>& children_depth, int cstart, int cend, 
    const vector<float>& pts_scaled, const int num_points, float scale_factor, const vector<OctreeParser::uint32>& unique_idx, const vector<OctreeParser::uint32>& sorted_idx, 
    MatrixXf R, Vector3f plane_center, MatrixXf coef, float support_radius)
{
    MatrixXf b = MatrixXf::Zero(6,1);
    MatrixXf p = MatrixXf::Zero(3,1);

    float w;
    float error = 0;

    // iterate over all nodes from final layer contained in cube
    for (int j = cstart; j < cend; ++j) {
        if (octree->node_type(children_depth[j]) == OctreeParser::NodeType::kLeaf) continue;

        int t = children_depth[j];

        for (int l = unique_idx[t]; l < unique_idx[t + 1]; l++) {

            // local coordinate system
            p << pts_scaled[3*sorted_idx[l]] / scale_factor, pts_scaled[3*sorted_idx[l] + 1] / scale_factor, pts_scaled[3*sorted_idx[l] + 2] / scale_factor;
            p = p - plane_center;
            w = p.norm() / support_radius; // calculate distance - support radius is always 1.0 unless there is overlap then equal radius
            p = R * p;

            // polynomial for each point
            /*b = biquad(p(0, 0), p(1, 0));
            for (int c = 0; c < 6; c++) {
                p(2,0) -= b(c, 0);
            }
            error += 1.0 * (fabs(p(2,0)) / num_points);*/

            error = fmax(error, taubin_distance_biquad(p(0, 0), p(1, 0), coef));
        }
    }

 return error;   
}


float polynomial::biquad_approximation_chamfer_dist(OctreeParser* octree, const vector<int>& children_depth, int cstart, int cend, 
    const vector<float>& pts_scaled, const int num_points, float scale_factor, const vector<OctreeParser::uint32>& unique_idx, const vector<OctreeParser::uint32>& sorted_idx, 
    Vector3f node_center, Vector3f plane_center, Vector3f plane_normal, MatrixXf coef, float support_radius)
{

    // create Points instance from Points in Cell
    // iterate over all nodes from final layer contained in cube
    std::vector<float> pts;
    for (int j = cstart; j < cend; ++j) {
        if (octree->node_type(children_depth[j]) == OctreeParser::NodeType::kLeaf) continue;
        int t = children_depth[j];
        for (int l = unique_idx[t]; l < unique_idx[t + 1]; l++) {
            for (int c = 0; c < 3; c++) {
                pts.push_back(pts_scaled[3*sorted_idx[l] + c] / scale_factor - node_center(c));
            }
        }
    }
    if (pts.size() <= 0) return -1;
    Points pts_pc;
    pts_pc.set_points((const std::vector<float>&) pts,(const std::vector<float>&) pts);

    // create Points instance from sampling on surface
    //marching_cube_octree_implicit(vector<float>& V, vector<int>& F, const vector<float>& pts,
    //  const vector<float>& pts_ref, const vector<float>& normals, const vector<float>& coefs, const int n_subdivision) 
    std::vector<float> verts;
    std::vector<int> faces;
    std::vector<float> vnode_pos { node_center(0), node_center(1), node_center(2) };
    //std::vector<float> vnode_pos { 0, 0, 0 };
    std::vector<float> vplane_pos { plane_center(0) / scale_factor, plane_center(1) / scale_factor, plane_center(2) / scale_factor };
    std::vector<float> vplane_normal { plane_normal(0), plane_normal(1), plane_normal(2) };
    std::vector<float> vcoefs { coef(0), coef(1), coef(2), coef(3), coef(4), coef(5) };
    marching_cube_octree_implicit(verts, faces, (const std::vector<float>&) vplane_pos, (const std::vector<float>&) vnode_pos, 
    (const std::vector<float>&) vplane_normal, (const std::vector<float>&) vcoefs, 5);

    if (verts.size() <= 0) return -1;
    Points sampled_pc;
    sampled_pc.set_points((const std::vector<float>&) verts,(const std::vector<float>&) verts);
    const float vtrans[3] { -plane_center(0) / scale_factor, -plane_center(1) / scale_factor, -plane_center(2) / scale_factor };
    sampled_pc.translate(vtrans);
    
    // calculate chamfer distance
    vector<size_t> idx;
    vector<float> distance;
    //chamfer_dist::closet_pts(idx, distance, pts_pc, sampled_pc);
    chamfer_dist::closet_pts(idx, distance, sampled_pc, pts_pc);

    // calc avgs
    float avg_ab = 0;
    for (auto& d : distance) { avg_ab += d; }
    avg_ab /= (float)distance.size();

    return avg_ab;   
}


float polynomial::fval_biquad(float u, float v, MatrixXf c) {
    MatrixXf b(6, 1);
    b = biquad(u, v); // get polynomial
    for (int ci = 0; ci < 6; ci++) { b(ci,0) *= c(ci,0); };
    return b.sum();
}

float polynomial::fval_triquad(Vector3f p, Vector3f plane_center, MatrixXf c)
{
    MatrixXf b(10, 1);
    b = triquad(p); // get polynomial
    for (int ci = 0; ci < 10; ci++) { b(ci,0) *= c(ci,0); }; // calc polynomial
    return b.sum();
}

float polynomial::taubin_distance_biquad(float u, float v, MatrixXf c) 
{
    float dx = c(1, 0) + c(3, 0)*u + c(4, 0)*v;
    float dy = c(2, 0) + c(4, 0)*u + c(5, 0)*v;
    return fval_biquad(u, v, c) / (dx*dx + dy*dy);  
}

Vector3f polynomial::uv2xyz(Vector2f uv, Vector3f plane_center, MatrixXf R, MatrixXf c)
{
    Vector3f p = {
        uv(0), 
        uv(1), 
        fval_biquad(uv(0), uv(1), c)
    };
    return R.inverse() * p + plane_center;
}

Vector3f polynomial::uv2norm(Vector2f uv, Vector3f pc, MatrixXf R, MatrixXf c)
{
    Vector2f du = {0.001, 0}, dv = {0, 0.001};
    auto xyz = uv2xyz(uv, pc, R, c);
    auto x0yz = uv2xyz(uv - du, pc, R, c);
    auto xy0z = uv2xyz(uv - dv, pc, R, c);
    return (xyz - x0yz).normalized().cross((xyz - xy0z).normalized()).normalized();
}


MatrixXf polynomial::biquad2triquad(Vector3f plane_center, MatrixXf R, MatrixXf c, float range)
{
    MatrixXf B = MatrixXf::Zero(10,10);
    MatrixXf b = MatrixXf::Zero(10,1);
    MatrixXf bf = MatrixXf::Zero(10,1);
    MatrixXf c_new(10,1);

    float offset = 0.001;
    Vector2f uv;
    Vector3f xyz;
    Vector3f norm;
    
    int n_samples = 3;

    // generate points on, inside and outside the surface
    for (int u = 0; u < n_samples; u++) {
        uv(0) = (float(u) / float(n_samples-1))*2*range - range;
        for (int v = 0; v < n_samples; v++) {
            uv(1) = (float(v) / float(n_samples-1))*2*range - range;
            xyz = uv2xyz(uv, plane_center, R, c);     // point on surface
            norm = uv2norm(uv, plane_center, R, c);   // normal at point to offset point

            // inside, on, outside
            for (int dir = -1; dir < 2; dir++) {
                b = triquad(xyz + norm*float(dir)*offset);
                B += b * b.transpose(); 
                bf += b * float(dir) * offset;
            }
        }
    }

    c_new = B.inverse() * bf;
    return c_new;
}