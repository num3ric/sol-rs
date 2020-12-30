#version 460
#extension GL_NV_ray_tracing : require
#include "payload.glsl"

layout (constant_id = 0) const int ENABLE_SKYLIGHT = 0;
layout(location = 0) rayPayloadInNV Payload prd;

void main()
{
    if( bool(ENABLE_SKYLIGHT) ) {
        vec3 wI = normalize( gl_WorldRayDirectionNV );
	    float t = smoothstep(0.35, 0.65, 0.5*(wI.y + 1));
	    vec3 skyColor = mix(vec3(0.58,0.45,0.25), vec3(0.3, 0.4, 0.5), t);
        bool isSun = dot(wI, normalize(vec3(0.0,1.0,-0.25))) > 0.99;
        prd.hitValue = mix(skyColor, vec3(120.0, 100.0, 50.0), float(isSun));
    }
    else{
        prd.hitValue = vec3(0.0);
    }
    prd.done = 1;
}