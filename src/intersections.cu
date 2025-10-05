#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    Ray q;
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;

    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];

        // NO if statement - always compute
        float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
        float t2 = (+0.5f - q.origin[xyz]) / qdxyz;

        // Ensure t1 <= t2
        if (t1 > t2) {
            float temp = t1;
            t1 = t2;
            t2 = temp;
        }

        // Update tmin
        if (t1 > tmin) {
            tmin = t1;
            tmin_n = glm::vec3(0.0f);
            tmin_n[xyz] = (q.origin[xyz] < 0) ? -1.0f : 1.0f;
        }

        // Update tmax
        if (t2 < tmax) {
            tmax = t2;
            tmax_n = glm::vec3(0.0f);
            tmax_n[xyz] = (q.origin[xyz] < 0) ? 1.0f : -1.0f;
        }
    }

    // Check for valid intersection
    if (tmax < tmin || tmax < 0.0f) {
        return -1.0f;
    }

    // Choose the appropriate t
    float t;
    glm::vec3 n;
    if (tmin < 0.0f) {
        t = tmax;
        n = tmax_n;
        outside = false;
    }
    else {
        t = tmin;
        n = tmin_n;
        outside = true;
    }

    // Transform back to world space
    glm::vec3 objspaceIntersection = getPointOnRay(q, t);
    intersectionPoint = multiplyMV(box.transform, glm::vec4(objspaceIntersection, 1.0f));
    normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(n, 0.0f)));

    return glm::length(r.origin - intersectionPoint);
}
__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}
