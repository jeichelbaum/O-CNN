#include "polynomial.h"


MatrixXf polynomial::calc_rotation_matrix(Vector3f norm1) {
    Vector3f norm2 = {0, 0, 1};

    if (norm1 == norm2) { return MatrixXf::Identity(3,3); }

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
    int jstart, int jend, MatrixXf R, Vector3f plane_center) {

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
      w = p.norm(); // calculate distance - support radius is always scaled to 1
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