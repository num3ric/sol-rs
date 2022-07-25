#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#include "payload.glsl"
#include "sampling.glsl"

struct ModelVertex {
	vec4 pos;
	vec4 color;
	vec4 normal;
	vec4 uv;
};

struct SceneInstance
{
	int  id;
	int  texture_offset;
	vec2 padding;
	mat4 transform;
	mat4 transform_it;
};

layout(set = 0, binding = 0) uniform Scene {
    mat4 model;
    mat4 view;
    mat4 view_inverse;
    mat4 projection;
    mat4 projection_inverse;
    mat4 model_view_projection;
    vec3 frame;
} scene;

layout(set = 1, binding = 2) uniform sampler2D blueNoise;
layout(set = 1, binding = 3, scalar) buffer ScnDesc { SceneInstance i[]; } scnDesc;
layout(set = 1, binding = 4, scalar) buffer Vertices { ModelVertex v[]; } vertices[];
layout(set = 1, binding = 5) buffer Indices { uint64_t i[]; } indices[];

layout(location = 0) rayPayloadInEXT Payload prd;

hitAttributeEXT vec3 attribs;


vec2 getBlueRand2( uint i )
{
	ivec2 texSize = textureSize( blueNoise, 0 ).xy;
	ivec2 blueCoord = ivec2( mod( gl_LaunchIDEXT.xy + nextRand2(prd.rng) * texSize, vec2( texSize ) ) );
	vec4 blue = texelFetch( blueNoise, blueCoord, 0 );
	return vec2( blue[i%4], blue[(i+1)%4] );
}

void main()
{
	// Object of this instance
	uint objId = scnDesc.i[gl_InstanceID].id;

	// Indices of the triangle
	ivec3 ind = ivec3(indices[objId].i[3 * gl_PrimitiveID + 0],   //
					  indices[objId].i[3 * gl_PrimitiveID + 1],   //
					  indices[objId].i[3 * gl_PrimitiveID + 2]);  //
	// Vertex of the triangle
	ModelVertex v0 = vertices[objId].v[ind.x];
	ModelVertex v1 = vertices[objId].v[ind.y];
	ModelVertex v2 = vertices[objId].v[ind.z];

	const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

	// Computing the normal at hit position
	vec3 normal = v0.normal.xyz * barycentrics.x + v1.normal.xyz * barycentrics.y + v2.normal.xyz * barycentrics.z;
	// Transforming the normal to world space
	normal = normalize(vec3(scnDesc.i[gl_InstanceID].transform_it * vec4(normal, 0.0)));

	// Computing the coordinates of the hit position
	vec3 world_pos = v0.pos.xyz * barycentrics.x + v1.pos.xyz * barycentrics.y + v2.pos.xyz * barycentrics.z;
	// Transforming the position to world space
	world_pos = vec3(scnDesc.i[gl_InstanceID].transform * vec4(world_pos, 1.0));

	prd.rayOrigin = world_pos + 0.00001 * gl_WorldRayDirectionEXT;
	vec2 Xi = getBlueRand2( prd.depth + prd.depth * prd.sampleId );
	vec3 hitNorm = normal * sign(dot(-gl_WorldRayDirectionEXT, normal));
	prd.rayDir = sampleCosineWeightedHemisphere(hitNorm, Xi);
	prd.rayRange = vec2(0.001f, 10.0f);
	if( prd.depth > 0 ){
		prd.hitValue += vec3(1.0);
	}
	prd.depth++;
}
