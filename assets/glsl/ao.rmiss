#version 460
#extension GL_EXT_ray_tracing : require
#include "payload.glsl"

layout(location = 0) rayPayloadInEXT Payload prd;

void main()
{
    prd.done = 1;
}