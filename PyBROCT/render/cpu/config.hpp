
//#define BENCHMARK
//#define SAVEOUT

//#define ALGORITHM_ENHANCED
//#define ALGORITHM_OPTIMIZED
//#define ALGORITHM_UNOPTIMIZED
#define ALGORITHM_SUPERPACK

#define KERNEL_PARALLEL
//#define KERNEL_SERIAL

#define INTERPOLATION_NONE
//#define INTERPOLATION_LINEAR

//#define DTYPE_FLOAT
#define DTYPE_CHAR

#define RENDER_SIZE 768

#if defined(ALGORITHM_ENHANCED)
#define VOLUME_TYPE float

#elif defined(ALGORITHM_OPTIMIZED)
#if defined(DTYPE_FLOAT)
#define VOLUME_TYPE float4
#elif defined(DTYPE_CHAR)
#define VOLUME_TYPE char4
#endif

#elif defined(ALGORITHM_SUPERPACK)
#define VOLUME_TYPE int2
#define SUPERPACK3(x, y, z) ((x) | ((y) << 8) | ((z) << 16))
#define SUPERPACK4(x, y, z, w) (SUPERPACK3(x, y, z) | ((w) << 24))

#define UNSUPERPACK3(x) make_char3((x) & 0xff, ((x) >> 8) & 0xff, ((x) >> 16) & 0xff)
#define UNSUPERPACK4(x) make_char4((x) & 0xff, ((x) >> 8) & 0xff, ((x) >> 16) & 0xff, ((x) >> 24))

#elif defined(ALGORITHM_UNOPTIMIZED)
#define VOLUME_TYPE float

#endif

#if defined(INTERPOLATION_NONE)
#define VOLUME_INTERPOLATION cudaFilterModePoint
#define VOLUME_READ_MODE cudaReadModeElementType

#elif defined(INTERPOLATION_LINEAR)
#define VOLUME_INTERPOLATION cudaFilterModeLinear

#if defined(DTYPE_CHAR)
#define VOLUME_READ_MODE cudaReadModeNormalizedFloat
#else
#define VOLUME_READ_MODE cudaReadModeElementType
#endif

#endif
