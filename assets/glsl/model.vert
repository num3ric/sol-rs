#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform Scene {
    mat4 mvp;
    mat4 normal;
} scene;

layout (location = 0) in vec4 pos;
layout (location = 1) in vec4 inColor;
layout (location = 2) in vec4 inNormal;
layout (location = 3) in vec2 inUv;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec4 outColor;

void main() {
    outColor = inColor;
    outNormal = mat3(scene.normal) * inNormal.xyz;
    gl_Position = scene.mvp * pos;
}