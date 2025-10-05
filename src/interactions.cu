#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    // Emissive materials - terminate path but accumulate light
    if (m.emittance > 0.0f) {
        pathSegment.color *= (m.color * m.emittance);
        pathSegment.remainingBounces = 0;
        return;
    }

    // Refractive materials (glass, water)
    if (m.hasRefractive) {
        float ior = m.indexOfRefraction;
        float cosi = glm::dot(pathSegment.ray.direction, normal);

        // Adjust normal and IOR based on ray direction
        bool outside = cosi < 0;
        glm::vec3 adjustedNormal = normal;
        if (!outside) {
            adjustedNormal = -normal;
            ior = 1.0f / ior;
        }

        cosi = fabs(cosi);

        // Calculate reflection and refraction
        glm::vec3 reflectedDir = reflect(pathSegment.ray.direction, adjustedNormal);
        glm::vec3 refractedDir;
        bool canRefract = refract(pathSegment.ray.direction, adjustedNormal, ior, refractedDir);

        // Fresnel effect for reflection probability
        float fresnel = schlick(cosi, m.indexOfRefraction);

        // Russian roulette between reflection and refraction
        if (u01(rng) < fresnel || !canRefract) {
            // Reflect
            pathSegment.ray.origin = intersect + adjustedNormal * 0.001f;
            pathSegment.ray.direction = reflectedDir;
        }
        else {
            // Refract
            pathSegment.ray.origin = intersect + refractedDir * 0.001f;
            pathSegment.ray.direction = refractedDir;
        }

        pathSegment.remainingBounces--;
    }
    // Specular materials (mirror)
    else if (m.hasReflective) {
        glm::vec3 reflectedDir = reflect(pathSegment.ray.direction, normal);

        pathSegment.ray.origin = intersect + normal * 0.001f;
        pathSegment.ray.direction = reflectedDir;
        pathSegment.remainingBounces--;
        pathSegment.color *= m.color;  // Multiply by material color
    }
    // Diffuse materials (lambertian)
    else {
        glm::vec3 newDirection = calculateRandomDirectionInHemisphere(normal, rng);
        float cosTheta = glm::max(0.0f, glm::dot(newDirection, normal));

        if (cosTheta > 0.0f) {
            // Monte Carlo integration for diffuse surfaces
            float pdf = cosTheta / PI;
            glm::vec3 brdf = m.color / PI;

            // Update path throughput
            pathSegment.color *= (brdf * cosTheta / pdf);

            // Set new ray direction
            pathSegment.ray.origin = intersect + normal * 0.001f;
            pathSegment.ray.direction = newDirection;
            pathSegment.remainingBounces--;
        }
        else {
            // Invalid direction - terminate path
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
        }
    }
}


__host__ __device__ glm::vec3 reflect(const glm::vec3& I, const glm::vec3& N) {
    return I - 2.0f * glm::dot(I, N) * N;
}

__host__ __device__ bool refract(const glm::vec3& I, const glm::vec3& N, float ior, glm::vec3& refracted) {
    float cosi = glm::clamp(glm::dot(I, N), -1.0f, 1.0f);
    float etai = 1.0f, etat = ior;
    glm::vec3 n = N;

    if (cosi < 0) {
        cosi = -cosi;
    }
    else {
        std::swap(etai, etat);
        n = -N;
    }

    float eta = etai / etat;
    float k = 1.0f - eta * eta * (1.0f - cosi * cosi);

    if (k < 0.0f) {
        return false; // Total internal reflection
    }
    else {
        refracted = eta * I + (eta * cosi - sqrtf(k)) * n;
        return true;
    }
}

__host__ __device__ float schlick(float cosine, float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;

    return r0 + (1.0f - r0) * powf((1.0f - glm::abs(cosine)), 5.0f);
}