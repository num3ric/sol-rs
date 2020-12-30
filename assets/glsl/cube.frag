#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(set = 0, binding = 1) uniform sampler2D uTexture;

layout (location = 0) in vec4 color;
layout (location = 1) in vec2 uv;
layout (location = 0) out vec4 outColor;

void main() {
    outColor = color + 0.25 * texture(uTexture, uv);
}
