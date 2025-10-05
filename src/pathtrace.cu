
#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)


#define DEPTH_OF_FIELD 0
#define FOCAL_DISTANCE 8.0f
#define LENS_RADIUS 0.1f
#define CACHE_FIRST_BOUNCE 0
#define RUSSIAN_ROULETTE 0
#define DIRECT_LIGHTING 1

void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif
    exit(EXIT_FAILURE);
#endif
}
__host__ __device__ glm::vec2 ConcentricSampleDisk(const glm::vec2& u) {
    glm::vec2 uOffset = 2.0f * u - glm::vec2(1.0f, 1.0f);

    if (uOffset.x == 0.0f && uOffset.y == 0.0f) {
        return glm::vec2(0.0f, 0.0f);
    }

    float theta, r;
    if (glm::abs(uOffset.x) > glm::abs(uOffset.y)) {
        r = uOffset.x;
        theta = (PI / 4.0f) * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = (PI / 2.0f) - (PI / 4.0f) * (uOffset.x / uOffset.y);
    }

    return r * glm::vec2(glm::cos(theta), glm::sin(theta));
}
__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static ShadeableIntersection* dev_intersections_first = NULL;

static bool sortByMaterialEnabled = true;

struct PathSegmentCompactionPredicate {
    __host__ __device__ bool operator()(const PathSegment& path) {
        return path.remainingBounces > 0;
    }
};

struct MaterialSortComparator {
    __host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) {
        return a.materialId < b.materialId;
    }
};

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_intersections_first, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections_first, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_intersections_first);

    checkCUDAError("pathtraceFree");
}

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        thrust::uniform_real_distribution<float> uNorm(-0.5f, 0.5f);

        float jitterX = uNorm(rng) * 0.5f;
        float jitterY = uNorm(rng) * 0.5f;

        glm::vec3 rayDir = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );

#if DEPTH_OF_FIELD
        float apertureRadius = 0.3f;
        float focalDistance = 6.0f;

        glm::vec3 focalPoint = cam.position + rayDir * focalDistance;

        float r = sqrt(u01(rng)) * apertureRadius;
        float theta = u01(rng) * 2.0f * PI;
        glm::vec3 lensOffset = cam.right * (r * cos(theta)) + cam.up * (r * sin(theta));

        segment.ray.origin = cam.position + lensOffset;
        segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);

#else
        segment.ray.origin = cam.position;
        segment.ray.direction = rayDir;
#endif

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }

            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            intersections[path_index].materialId = -1;
        }
        else
        {
            if (hit_geom_index >= 0 && hit_geom_index < geoms_size) {
                intersections[path_index].t = t_min;
                intersections[path_index].materialId = geoms[hit_geom_index].materialid;
                intersections[path_index].surfaceNormal = normal;
            }
            else {
                intersections[path_index].t = -1.0f;
                intersections[path_index].materialId = -1;
            }
        }
    }
}

__device__ glm::vec3 sampleDirectLighting(
    const glm::vec3& hitPoint,
    const glm::vec3& normal,
    const Material& material,
    Geom* geoms,
    int geoms_size,
    Material* materials,
    int materials_count,
    thrust::default_random_engine& rng) {

    thrust::uniform_real_distribution<float> u01(0, 1);

    int lightCount = 0;
    int lightIndices[10];

    for (int i = 0; i < geoms_size; i++) {
        if (i < 0 || i >= geoms_size) continue;

        int materialId = geoms[i].materialid;
        if (materialId < 0 || materialId >= materials_count) continue;

        if (materials[materialId].emittance > 0.0f) {
            if (lightCount < 10) {
                lightIndices[lightCount++] = i;
            }
        }
    }

    if (lightCount == 0) return glm::vec3(0.0f);

    int lightIndex = min((int)(u01(rng) * lightCount), lightCount - 1);
    int lightIdx = lightIndices[lightIndex];

    if (lightIdx < 0 || lightIdx >= geoms_size) {
        return glm::vec3(0.0f);
    }

    Geom& light = geoms[lightIdx];

    if (light.materialid < 0 || light.materialid >= materials_count) {
        return glm::vec3(0.0f);
    }

    Material& lightMaterial = materials[light.materialid];

    glm::vec3 lightPoint;
    float lightArea = 1.0f;
    glm::vec3 lightNormal = glm::vec3(0.0f, 1.0f, 0.0f);

    if (light.type == SPHERE) {
        glm::vec3 sphereCenter = multiplyMV(light.transform, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        float radius = light.transform[0][0] * 0.5f;

        glm::vec3 randomDir = calculateRandomDirectionInHemisphere(glm::vec3(0.0f, 1.0f, 0.0f), rng);
        lightPoint = sphereCenter + randomDir * radius;
        lightArea = 4.0f * PI * radius * radius;

        lightNormal = glm::normalize(lightPoint - sphereCenter);
    }
    else if (light.type == CUBE) {
        lightPoint = multiplyMV(light.transform, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        lightArea = 1.0f;
        lightNormal = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    else {
        return glm::vec3(0.0f);
    }

    glm::vec3 toLight = lightPoint - hitPoint;
    float distanceSquared = glm::dot(toLight, toLight);
    float distance = sqrt(distanceSquared);

    if (distance < 0.001f) {
        return glm::vec3(0.0f);
    }

    glm::vec3 lightDir = toLight / distance;

    Ray shadowRay;
    shadowRay.origin = hitPoint + normal * 0.001f;
    shadowRay.direction = lightDir;

    bool inShadow = false;
    for (int i = 0; i < geoms_size; i++) {
        if (i == lightIdx) continue;

        Geom& geom = geoms[i];
        float t;
        glm::vec3 tmp_intersect, tmp_normal;
        bool outside;

        if (geom.type == CUBE) {
            t = boxIntersectionTest(geom, shadowRay, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SPHERE) {
            t = sphereIntersectionTest(geom, shadowRay, tmp_intersect, tmp_normal, outside);
        }
        else {
            t = -1.0f;
        }

        if (t > 0.001f && t < distance - 0.001f) {
            inShadow = true;
            break;
        }
    }

    if (!inShadow) {
        float cosTheta = glm::max(0.0f, glm::dot(normal, lightDir));
        float cosLight = glm::max(0.0f, glm::dot(lightNormal, -lightDir));

        if (cosTheta < 0.001f || cosLight < 0.001f) {
            return glm::vec3(0.0f);
        }

        float pdf = 1.0f / (lightCount * lightArea);

        if (pdf > 1e-10f && distanceSquared > 1e-10f) {
            glm::vec3 brdf = material.color / PI;
            glm::vec3 lightContribution = lightMaterial.color * lightMaterial.emittance;

            glm::vec3 result = brdf * lightContribution * cosTheta * cosLight / (pdf * distanceSquared);

            return glm::min(result, glm::vec3(100.0f));
        }
    }

    return glm::vec3(0.0f);
}
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms_param,
    int geoms_size_param,
    Material* materials_param,
    int materials_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        PathSegment& pathSegment = pathSegments[idx];

        if (intersection.t > 0.0f)
        {
            if (intersection.materialId < 0 || intersection.materialId >= materials_count) {
                pathSegment.color = glm::vec3(1.0f, 0.0f, 1.0f);
                pathSegment.remainingBounces = 0;
                return;
            }

            Material material = materials[intersection.materialId];
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            glm::vec3 intersectPoint = pathSegment.ray.origin + pathSegment.ray.direction * intersection.t;

            if (material.emittance > 0.0f) {
                pathSegment.color *= (material.color * material.emittance);
                pathSegment.remainingBounces = 0;
            }
            else if (material.hasRefractive) {
                scatterRay(pathSegment, intersectPoint, intersection.surfaceNormal, material, rng);
            }
            else if (material.hasReflective) {
                scatterRay(pathSegment, intersectPoint, intersection.surfaceNormal, material, rng);
            }
            else {
                glm::vec3 intersectPoint = pathSegment.ray.origin + pathSegment.ray.direction * intersection.t;

#if DIRECT_LIGHTING
                glm::vec3 directLight = sampleDirectLighting(
                    intersectPoint,
                    intersection.surfaceNormal,
                    material,
                    geoms_param,
                    geoms_size_param,
                    materials_param,
                    materials_count,
                    rng
                );

                scatterRay(pathSegment, intersectPoint, intersection.surfaceNormal, material, rng);

                pathSegment.color += directLight;
#else
                scatterRay(pathSegment, intersectPoint, intersection.surfaceNormal, material, rng);
#endif
            }
        }
        else {
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
        }
    }
}
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

void compactPaths(int& num_paths, PathSegment* dev_paths) {
    int before = num_paths;
    thrust::device_ptr<PathSegment> thrust_paths(dev_paths);

    PathSegmentCompactionPredicate pred;
    thrust::device_ptr<PathSegment> new_end = thrust::partition(
        thrust_paths, thrust_paths + num_paths,
        pred
    );
    num_paths = new_end - thrust_paths;
    if (before > 0) {
        float percentKept = (float)num_paths / before * 100.0f;
        printf("Bounce compaction: %d -> %d paths (%.1f%% kept)\n", before, num_paths, percentKept);
    }
}

void sortPathsByMaterial(int num_paths, PathSegment* dev_paths, ShadeableIntersection* dev_intersections) {
    thrust::device_ptr<PathSegment> paths_ptr(dev_paths);
    thrust::device_ptr<ShadeableIntersection> intersections_ptr(dev_intersections);

    MaterialSortComparator comp;
    thrust::sort_by_key(intersections_ptr, intersections_ptr + num_paths, paths_ptr,
        comp
    );
}

void pathtrace(uchar4* pbo, int frame, int iter)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    const int blockSize1d = 128;

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    int num_paths = pixelcount;

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHE_FIRST_BOUNCE
        if (depth == 0) {
            if (iter == 1) {
                computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                    depth,
                    num_paths,
                    dev_paths,
                    dev_geoms,
                    hst_scene->geoms.size(),
                    dev_intersections
                    );
                checkCUDAError("compute first bounce intersections");
                cudaDeviceSynchronize();

                cudaMemcpy(dev_intersections_first, dev_intersections,
                    num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            }
            else {
                cudaMemcpy(dev_intersections, dev_intersections_first,
                    num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            }
        }
        else {
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth,
                num_paths,
                dev_paths,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_intersections
                );
            checkCUDAError("compute intersections");
            cudaDeviceSynchronize();
        }
#else
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
            );
        checkCUDAError("compute intersections");
        cudaDeviceSynchronize();
#endif

        if (sortByMaterialEnabled && depth > 0) {
            sortPathsByMaterial(num_paths, dev_paths, dev_intersections);
        }

        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_materials,
            hst_scene->materials.size()
            );
        checkCUDAError("shade material");
        cudaDeviceSynchronize();

        int paths_before_compaction = num_paths;
        compactPaths(num_paths, dev_paths);

        depth++;

        if (num_paths == 0 || depth >= traceDepth) {
            iterationComplete = true;
        }

        if (guiData != NULL) {
            guiData->TracedDepth = depth;
        }
    }

    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    printf("Iteration %d: %lld ms\n", iter, duration.count());

    checkCUDAError("pathtrace");
}