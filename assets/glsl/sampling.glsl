#ifndef SAMPLING_GLSL
#define SAMPLING_GLSL
#extension GL_EXT_control_flow_attributes : enable
#define M_PI 3.14159265359
#define TWO_PI 6.28318530718

#define saturate(x) clamp(x, 0.0, 1.0)

float distanceSquared( vec3 u, vec3 v)
{
    vec3 d = u - v;
    return dot( d, d );
}

// Generate a random unsigned int from two unsigned int values, using 16 pairs
// of rounds of the Tiny Encryption Algorithm. See Zafar, Olano, and Curtis,
// "GPU Random Numbers via the Tiny Encryption Algorithm"
uint tea(uint val0, uint val1)
{
    uint v0 = val0;
    uint v1 = val1;
    uint s0 = 0;

    [[unroll]]
    for(uint n = 0; n < 16; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

// Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
float nextRand(inout uint rng)
{
    // Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
    rng  = rng * 747796405 + 1;
    uint word = ((rng >> ((rng >> 28) + 4)) ^ rng) * 277803737;
    word      = (word >> 22) ^ word;
    return float(word) / 4294967295.0f;
}

vec2 nextRand2(inout uint rng)
{
    return vec2(nextRand(rng), nextRand(rng));
}

// Sampling functions taken from metal-ray-tracer
// Refer to: https://github.com/sergeyreznik/metal-ray-tracer/blob/part-5/source/Shaders/raytracing.h

float fresnelDielectric(vec3 i, vec3 m, float eta)
{
    float result = 1.0f;
    float cosThetaI = abs(dot(i, m));
    float sinThetaOSquared = (eta * eta) * (1.0f - cosThetaI * cosThetaI);
    if (sinThetaOSquared <= 1.0f) {
        float cosThetaO = sqrt(saturate(1.0f - sinThetaOSquared));
        float Rs = (cosThetaI - eta * cosThetaO) / (cosThetaI + eta * cosThetaO);
        float Rp = (eta * cosThetaI - cosThetaO) / (eta * cosThetaI + cosThetaO);
        result = 0.5f * (Rs * Rs + Rp * Rp);
    }
    return result;
}

void buildOrthonormalBasis(vec3 n, inout vec3 u, inout vec3 v)
{
    float s = (n.z < 0.0 ? -1.0f : 1.0f);
    float a = -1.0f / (s + n.z);
    float b = n.x * n.y * a;
    u = vec3(1.0f + s * n.x * n.x * a, s * b, -s * n.x);
    v = vec3(b, s + n.y * n.y * a, -n.y);
}

vec3 alignToDirection(vec3 n, float cosTheta, float phi)
{
    float sinTheta = sqrt(saturate(1.0f - cosTheta * cosTheta));

    vec3 u;
    vec3 v;
    buildOrthonormalBasis(n, u, v);

    return (u * cos(phi) + v * sin(phi)) * sinTheta + n * cosTheta;
}

vec3 sampleGGXDistribution(vec3 n, vec2 Xi, float alphaSquared)
{
    float cosTheta = sqrt(saturate((1.0f - Xi.x) / (Xi.x * (alphaSquared - 1.0f) + 1.0f)));
    return alignToDirection(n, cosTheta, Xi.y * TWO_PI);
}

vec3 sampleCosineWeightedHemisphere(vec3 n, vec2 Xi)
{
    float cosTheta = sqrt(Xi.x);
    return alignToDirection(n, cosTheta, Xi.y * TWO_PI);
}

#endif