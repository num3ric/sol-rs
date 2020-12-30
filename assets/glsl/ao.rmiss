#version 460
#extension GL_NV_ray_tracing : require
#include "payload.glsl"

layout(location = 0) rayPayloadInNV Payload prd;

void main()
{
    prd.done = 1;
}