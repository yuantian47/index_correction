#include "raycast.hpp"

#define ZERO_THRESHOLD  1e-5
//
//short4 operator&(short4 a, short b)
//{
//    return make_short4(a.x & b, a.y & b, a.z & b, a.w & b);
//}
//
//short4 operator>>(short4 a, size_t b)
//{
//    return make_short4(a.x >> b, a.y >> b, a.z >> b, a.w >> b);
//}
//
//float4 operator/(short4 a, float b)
//{
//    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
//}
//
//float4 operator/(char4 a, float b)
//{
//    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
//}
//
//float3 operator/(char3 a, float b)
//{
//    return make_float3(a.x / b, a.y / b, a.z / b);
//}

float3 make_float3(const float4& a) {
    return { a.x, a.y, a.z };
}

float3 make_float3(float x, float y, float z) {
    return { x, y, z };
}

float4 make_float4(float x, float y, float z, float w) {
    return { x, y, z, w };
}

uchar4 float4_to_uchar4(const float4& rgba) {
    uchar4 out;
    out.x = (unsigned char)(255 * clamp(rgba.x));   // clamp to [0.0, 1.0]
    out.y = (unsigned char)(255 * clamp(rgba.y));
    out.z = (unsigned char)(255 * clamp(rgba.z));
    out.w = (unsigned char)(255 * clamp(rgba.w));
    return out;
}

float3 operator*(float a, const float3& b) {
    return { a * b.x, a * b.y, a * b.z };
}
float3 operator*(const float3& b, float a) {
    return a * b;
}

float4 operator*(float a, const float4& b) {
    return { a * b.x, a * b.y, a * b.z, a * b.w };
}
float4 operator*(const float4& b, float a) {
    return a * b;
}

void operator*=(float4& a, const float4& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}

void operator*=(float4& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

float3 operator+(float a, const float3& b) {
    return { a + b.x, a + b.y, a + b.z };
}
float3 operator+(const float3& b, float a) {
    return a + b;
}

void operator+=(float4& a, const float4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

float3 operator/(const float3& a, float b) {
    return { a.x / b, a.y / b, a.z / b };
}

float3 operator+(const float3& a, const float3& b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}
float3 operator-(const float3& a, const float3& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

float dot(const float4& a, const float4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 normalize(const float3& a) {
    return a / std::sqrtf(dot(a, a));
}

float rsqrtf(float o) {
    return 1.0f / std::sqrtf(o);
}

// transform vector by matrix with translation
float4 mul(const float4x4& M, const float4& v) {
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = dot(v, M.m[3]);
    return r;
}

float3 grad3D(const NormalizedTexture3D<char>& tex, const float3& pos, float step) {
    float gradx = tex.at(pos.x + step, pos.y, pos.z) - tex.at(pos.x - step, pos.y, pos.z);
    float grady = tex.at(pos.x, pos.y + step, pos.z) - tex.at(pos.x, pos.y - step, pos.z);
    float gradz = tex.at(pos.x, pos.y, pos.z + step) - tex.at(pos.x, pos.y, pos.z - step);

    return make_float3(gradx, grady, gradz);
}

CastResult raycast(const NormalizedTexture3D<char>& volume, const Ray& ray, float tnear, float tfar, const TransferFunction& tf, const RenderSettings& rs) {
    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0);

    float3 step = rs.tstep * ray.normal;
    float t = tnear;

    float3 light;
    light.x = -cosf(1.57079f / 4) * sinf(1.57079f / 4);
    light.y = -sinf(1.57079f / 4) * sinf(1.57079f / 4);
    light.z = -cosf(1.57079f / 4);

    //float3 h = normalize(-step + light);
    //float3 h = -1 * (step + light) / fabs(step.x + light.x) / sqrtf(3);
    float3 h = (light - step) * rsqrtf(dot(step + light, step + light));

    float surface = rs.tmax;
    bool mark = false;
    for(unsigned short i = 0; i < rs.max_steps; i++) {
        // compute ray position parametrically
        float3 pos = ray.origin + t * ray.normal;
        // remap position from [-1, 1] to [0, 1]
        pos = (pos + 1) / 2;

        auto sample = volume.at(pos.x, pos.y, pos.z);
        if(sample < tf.threshold) {
            sample = 0;
        }

        // sample gradient 3D texture at ray position
        float3 grad = grad3D(volume, pos, rs.tstep);
        float gradmag = sqrtf(dot(grad, grad));

        // colorize sample based on transfer function
        float4 col = tf.function->at((sample - tf.offset) * tf.scale);
        //float4 col = { sample, sample, sample, sample };

        // set starting alpha at 40%
        col.w *= 0.4f;

        if(rs.retina) {
            // edge enhancement
            col.w *= 1 + 2.5f * rs.edge_enhancement * powf(gradmag, 0.3f);
            // feature enhancement
            col.w *= 1 + 0.5f * powf(1 - abs(dot(grad, step)), 0.4f);
        } else {
            // edge enhancement
            col.w *= 1 + 0.25f * rs.edge_enhancement * powf(gradmag, 0.3f);
            // feature enhancement
            col.w *= 1 + 0.05f * powf(1 - abs(dot(grad, step)), 0.4f);
        }

        // pre-multiply alpha
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;

        // distance color blending
        float d_ray = powf(float(i) / rs.max_steps, 4);
        col.x = (1 - 1.2f * d_ray) * col.x;
        col.y = (1 - 1.2f * d_ray) * col.y;
        col.z = (1 - 1.2f * d_ray) * col.z + 0.5f * d_ray * 0.15f * col.w;

        // Phong shader
        float shader = 0.45f * (dot(grad, light) + 0.6f * dot(grad, h)) / (gradmag + 0.001f);
        col.x *= (1.3f + shader);
        col.y *= (1.3f + shader);
        col.z *= (1.3f + shader);

        // "over" operator for front-to-back blending
        sum += col * (1 - sum.w);

        // record surface position
        if(!mark && sum.w > rs.surface_threshold) {
            surface = t;
            mark = true;
        }

        // exit early if opaque
        if(sum.w > rs.opacity_threshold) {
            break;
        }

        // advance ray parameter
        t += rs.tstep;
        // check if beyond ray endpoint
        if(t > tfar) {
            break;
        }
    }

    return { surface, sum };
}

bool check_planes(const Plane* planes, unsigned char count, const Ray& ray, float* tnear, float* tfar, unsigned char* pnear, unsigned char* pfar) {
    for(unsigned char i = 0; i < count; i++) {
        // ref: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection

        float denom = dot(ray.normal, planes[i].normal);
        float numer = dot(planes[i].point - ray.origin, planes[i].normal);

        if(abs(denom) < ZERO_THRESHOLD) {
            // ray parallel to plane

            // check if before or after plane
            if(numer > 0) {
                // after plane -> do not cast
                return false;
            } else {
                // before plane -> no affect on endpoints
            }
        } else {
            // ray intersects plane

            // compute intersection point
            float d = numer / denom;
            float3 x = ray.origin + d * ray.normal;

            // check if intersection is aligned or opposed
            if(denom >= 0) {
                // aligned -> front endpoint
                if(d > *tnear) {
                    *tnear = d;
                    *pnear = i;
                }
            } else {
                // opposed -> back endpoint
                if(d < *tfar) {
                    *tfar = d;
                    *pfar = i;
                }
            }
        }

        // check if valid casting region
        if(*tfar < *tnear) {
            // no casting region -> do not cast
            return false;
        }
    }

    // cast with given tnear and tfar
    return true;
}

void render(uchar4* color, float* depth, unsigned int width, unsigned int height, const PlaneArray& planes, const NormalizedTexture3D<char>& volume, const float4x4& inverse_modelview, const float4x4& inverse_projection, const region_t& region, const TransferFunction& tf, const RenderSettings& rs) {
#pragma omp parallel for //collapse(2)
    for(int x = 0; x < ptrdiff_t(width); x++) {
        for(int y = 0; y < ptrdiff_t(height); y++) {

            // compute ray direction in (u,v) space
            float u = region.u.min + (x / float(width - 1)) * region.u.range();
            float v = region.v.min + (y / float(height - 1)) * region.v.range();

            Ray ray;
            // compute ray origin in (x,y,z) space
            ray.origin = make_float3(mul(inverse_modelview, make_float4(0, 0, 0, 1)));

            // compute ray direction in (x,y,z) space
            // ref: http://antongerdelan.net/opengl/raycasting.html
            auto ray_eye = mul(inverse_projection, make_float4(u, v, 0, 1));
            auto ray_world = mul(inverse_modelview, make_float4(ray_eye.x, ray_eye.y, -1, 0));
            ray.normal = normalize(make_float3(ray_world.x, ray_world.y, ray_world.z));

            // find intersection with box
            float tnear = rs.tmin;
            float tfar = rs.tmax;
            unsigned char pnear, pfar;
            bool cast = check_planes(planes.planes, planes.count, ray, &tnear, &tfar, &pnear, &pfar);
            if(!cast) {
                continue;
            }

            // cast the ray only if valid
            auto result = raycast(volume, ray, tnear, tfar, tf, rs);

            // scale according to brightness
            result.color *= rs.brightness;

            // store output
            if(color) {
                color[y * width + x] = { (unsigned char)(255 * result.color.x), (unsigned char)(255 * result.color.y), (unsigned char)(255 * result.color.z), (unsigned char)(255 * result.color.w)};
                //color[y * width + x] = { (unsigned char)(255 * clamp(result.color.x)), (unsigned char)(255 * clamp(result.color.y)), (unsigned char)(255 * clamp(result.color.z)), (unsigned char)(255 * clamp(result.color.w))};
                //color[y * width + x] = float4_to_uchar4(result.color);
            }
            if(depth) {
                depth[y * width + x] = result.depth;
            }
        }
    }
}

void multirender(RenderArray* ra, const NormalizedTexture3D<char>& volume, TransferFunction tf, RenderSettings rs) {
    for(unsigned int z = 0; z < ra->count; z++) {
        Render& r = ra->renders[z];
        render(r.color, r.depth, r.width, r.height, r.pa, volume, r.inverse_modelview, r.inverse_projection, r.region, tf, rs);
    }
}

void multirender(RenderArray* ra, unsigned int index, const NormalizedTexture3D<char>& volume, TransferFunction tf, RenderSettings rs) {
    unsigned int z = index;
    if(z >= ra->count) {
        return;
    }

    Render& r = ra->renders[z];
    render(r.color, r.depth, r.width, r.height, r.pa, volume, r.inverse_modelview, r.inverse_projection, r.region, tf, rs);
}
