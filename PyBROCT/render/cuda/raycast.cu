#include "raycast.cuh"

#include "helper_math.hpp"

#define ZERO_THRESHOLD  1e-5
#define VOLUME_TYPE float

__device__
short4 operator&(short4 a, short b)
{
    return make_short4(a.x & b, a.y & b, a.z & b, a.w & b);
}

__device__
short4 operator>>(short4 a, size_t b)
{
    return make_short4(a.x >> b, a.y >> b, a.z >> b, a.w >> b);
}

__device__
float4 operator/(short4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

__device__
float4 operator/(char4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

__device__
float3 operator/(char3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__
uchar4 float4_to_uchar4(const float4& rgba) {
    uchar4 out;
    out.x = 255 * __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    out.y = 255 * __saturatef(rgba.y);
    out.z = 255 * __saturatef(rgba.z);
    out.w = 255 * __saturatef(rgba.w);
    return out;
}


// transform vector by matrix with translation
__device__
float4 mul(const float4x4& M, const float4& v) {
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = dot(v, M.m[3]);
    return r;
}

__device__
float3 grad3D(cudaTextureObject_t tex, const float3& pos, float step) {
    float gradx = tex3D<float>(tex, pos.x + step, pos.y, pos.z) - tex3D<float>(tex, pos.x - step, pos.y, pos.z);
    float grady = tex3D<float>(tex, pos.x, pos.y + step, pos.z) - tex3D<float>(tex, pos.x, pos.y - step, pos.z);
    float gradz = tex3D<float>(tex, pos.x, pos.y, pos.z + step) - tex3D<float>(tex, pos.x, pos.y, pos.z - step);

    return make_float3(gradx, grady, gradz);
}

__device__
CastResult raycast(cudaTextureObject_t volume, const Ray& ray, float tnear, float tfar, const TransferFunction& tf, const RenderSettings& rs) {
    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0);

    float3 step = rs.tstep * ray.normal;
    float t = tnear;

    float3 light;
    light.x = -cosf(1.57079 / 4) * sinf(1.57079 / 4);
    light.y = -sinf(1.57079 / 4) * sinf(1.57079 / 4);
    light.z = -cosf(1.57079 / 4);

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

#if defined(ALGORITHM_OPTIMIZED)

        VOLUME_TYPE v = tex3D<VOLUME_TYPE>(volume, pos.x, pos.y, pos.z);

        // sample volume 3D texture at ray position
        float sample = v.w / 255.0f;
        if(sample < tf.threshold) {
            sample = 0;
        }

        // sample gradient 3D texture at ray position
        float3 grad = make_float3(v.x, v.y, v.z) / 255.0f;
        float gradmag = sqrtf(dot(grad, grad));

#elif defined(ALGORITHM_SUPERPACK)

        float3 size = { 1024, 1327, 128 };

        float3 ps = pos * size;
        float3 ps_i = { floorf(ps.x), floorf(ps.y), floorf(ps.z) };
        float3 ps_t = ps_i / size;

        float3 d = ps - ps_i;
        float3 dr = 1 - d;

        VOLUME_TYPE v = tex3D<VOLUME_TYPE>(volume, ps_t.x, ps_t.y, ps_t.z);

        //WriteVector pack = {
        //    tex3D<ReadScalar>(src, xs, ys, zs) | tex3D<ReadScalar>(src, xs, ys, zs + 1) << 8,
        //    tex3D<ReadScalar>(src, xs + 1, ys, zs) | tex3D<ReadScalar>(src, xs + 1, ys, zs + 1) << 8,
        //    tex3D<ReadScalar>(src, xs, ys + 1, zs) | tex3D<ReadScalar>(src, xs, ys + 1, zs + 1) << 8,
        //    tex3D<ReadScalar>(src, xs + 1, ys + 1, zs) | tex3D<ReadScalar>(src, xs + 1, ys + 1, zs + 1) << 8,
        //};

        //float4 top = (v >> 8) / 255.0f;
        //float4 bot = (v & 0xff) / 255.0f;

        //WriteVector pack = {
        //    SUPERPACK4(
        //        tex3D<ReadScalar>(src, xs, ys, zs),
        //        tex3D<ReadScalar>(src, xs + 1, ys, zs),
        //        tex3D<ReadScalar>(src, xs, ys + 1, zs),
        //        tex3D<ReadScalar>(src, xs + 1, ys + 1, zs)
        //    ),
        //    SUPERPACK4(
        //        tex3D<ReadScalar>(src, xs, ys, zs + 1),
        //        tex3D<ReadScalar>(src, xs + 1, ys, zs + 1),
        //        tex3D<ReadScalar>(src, xs, ys + 1, zs + 1),
        //        tex3D<ReadScalar>(src, xs + 1, ys + 1, zs + 1)
        //    ),
        //    //SUPERPACK3(
        //    //    tex3D<ReadScalar>(src, xs + 1, ys, zs) - tex3D<ReadScalar>(src, xs - 1, ys, zs),
        //    //    tex3D<ReadScalar>(src, xs, ys + 1, zs) - tex3D<ReadScalar>(src, xs, ys - 1, zs),
        //    //    tex3D<ReadScalar>(src, xs, ys, zs + 1) - tex3D<ReadScalar>(src, xs, ys, zs - 1)
        //    //),
        //    //0
        //};

        float4 bot = UNSUPERPACK4(v.x) / 255.0f;
        float4 top = UNSUPERPACK4(v.y) / 255.0f;

        float c00 = bot.x * dr.x + bot.y * d.x;
        float c01 = top.x * dr.x + top.y * d.x;
        float c10 = bot.z * dr.x + bot.w * d.x;
        float c11 = top.z * dr.x + top.w * d.x;

        float c0 = c00 * dr.y + c10 * d.y;
        float c1 = c01 * dr.y + c11 * d.y;

        float sample = c0 * dr.z + c1 * d.z;
        //float sample = bot.x;
        //float sample = v.x / 255.0f;

        if(sample < tf.threshold) {
            sample = 0;
        }

        //float3 grad = {
        //    (bot.y + top.y + bot.w + top.w) - (bot.x + top.x + bot.z + top.z),
        //    (bot.z + bot.w + top.z + top.w) - (bot.x + bot.y + top.x + top.y),
        //    (top.x + top.y + top.z + top.w) - (bot.x + bot.y + bot.z + bot.w),
        //};
        //float3 grad = UNSUPERPACK3(v.z) / 255.0f;

        float3 grad = {
            dr.y * dr.z * (bot.y - bot.x) + d.y * dr.z * (bot.w - bot.z) + dr.y * d.z * (top.y - top.x) + d.y * d.z * (top.w - top.z),
            dr.x * dr.z * (bot.z - bot.x) + d.x * dr.z * (bot.w - bot.y) + dr.x * d.z * (top.z - top.x) + d.x * d.z * (top.w - top.y),
            dr.x * dr.y * (top.x - bot.x) + d.x * dr.y * (top.y - bot.y) + dr.x * d.y * (top.z - bot.z) + d.x * d.y * (top.w - bot.w),
        };
        //grad *= float(min(size.x, min(size.y, size.z))) / size;

        float gradmag = sqrtf(dot(grad, grad));

#else

        VOLUME_TYPE sample = tex3D<VOLUME_TYPE>(volume, pos.x, pos.y, pos.z);
        if(sample < tf.threshold) {
            sample = 0;
        }

        // sample gradient 3D texture at ray position
        float3 grad = grad3D(volume, pos, rs.tstep);
        float gradmag = sqrtf(dot(grad, grad));

#endif

        // colorize sample based on transfer function
        float4 col = tex1D<float4>(tf.tex, (sample - tf.offset) * tf.scale);
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

__device__
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

__device__
void render(uchar4* color, float* depth, unsigned int width, unsigned int height, const PlaneArray& planes, cudaTextureObject_t volume, const float4x4& inverse_modelview, const float4x4& inverse_projection, const region_t& region, const TransferFunction& tf, const RenderSettings& rs) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // check valid output coordinates
    if(x >= width || y >= height) {
        return;
    }

    // compute ray direction in (u,v) space
    float u = region.u.min + (x / float(width - 1)) * region.u.range();
    float v = region.v.min + (y / float(height - 1)) * region.v.range();

    Ray ray;
    // compute ray origin in (x,y,z) space
    ray.origin = make_float3(mul(inverse_modelview, make_float4(0, 0, 0, 1)));

    // compute ray direction in (x,y,z) space
    // ref: http://antongerdelan.net/opengl/raycasting.html
    float4 ray_eye = mul(inverse_projection, make_float4(u, v, 0, 1));
    float4 ray_world = mul(inverse_modelview, make_float4(ray_eye.x, ray_eye.y, -1, 0));
    ray.normal = normalize(make_float3(ray_world.x, ray_world.y, ray_world.z));

    // find intersection with box
    float tnear = rs.tmin;
    float tfar = rs.tmax;
    unsigned char pnear, pfar;
    bool cast = check_planes(planes.planes, planes.count, ray, &tnear, &tfar, &pnear, &pfar);
    if(!cast) {
        return;
    }

    // cast the ray only if valid
    CastResult result = raycast(volume, ray, tnear, tfar, tf, rs);
    
    // scale according to brightness
    result.color *= rs.brightness;

    // store output
    if(color) {
        color[y * width + x] = float4_to_uchar4(result.color);
    }
    if(depth) {
        depth[y * width + x] = result.depth;
    }
}

__global__
void multirender(RenderArray* ra, cudaTextureObject_t volume, TransferFunction tf, RenderSettings rs) {
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    if(z >= ra->count) {
        return;
    }

    Render& r = ra->renders[z];
    render(r.color, r.depth, r.width, r.height, r.pa, volume, r.inverse_modelview, r.inverse_projection, r.region, tf, rs);
}

__global__
void multirender(RenderArray* ra, unsigned int index, cudaTextureObject_t volume, TransferFunction tf, RenderSettings rs) {
    unsigned int z = index;
    if(z >= ra->count) {
        return;
    }

    Render& r = ra->renders[z];
    render(r.color, r.depth, r.width, r.height, r.pa, volume, r.inverse_modelview, r.inverse_projection, r.region, tf, rs);
}

void launch_multirender(const dim3& blocks, const dim3& threads, RenderArray* ra, cudaTextureObject_t volume, const TransferFunction& tf, const RenderSettings& rs) {
    multirender<<<blocks, threads>>>(ra, volume, tf, rs);
}
