#ifndef PAYLOAD_GLSL
#define PAYLOAD_GLSL

struct Payload
{
    vec3 hitValue;
    uint  depth;
    uint  sampleId;
    uint  done;
    vec3 rayOrigin;
    vec3 rayDir;
    vec2 rayRange;
    float roughness;
    uint rng;
};
#endif
