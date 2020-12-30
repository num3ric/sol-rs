#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(set = 0, binding = 0) uniform Scene {
    mat4 mvp;
} scene;

layout (location = 0) in vec4 pos;
layout (location = 1) in vec4 inColor;
layout (location = 2) in vec2 inUv;

layout (location = 0) out vec4 outColor;
layout (location = 1) out vec2 outUv;

void main() {
   outColor = inColor;
   outUv = inUv;
   gl_Position = scene.mvp * pos;
}