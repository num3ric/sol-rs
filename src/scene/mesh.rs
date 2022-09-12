use super::PrimitiveSection;
use crate::{offset_of, Buffer, Context, Resource, Vertex};
use ash::{vk};
use std::sync::Arc;

//TODO: solve non-vec4-aligned issues..
#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct ModelVertex {
    pub pos: glam::Vec4,
    pub color: glam::Vec4,
    pub normal: glam::Vec4,
    pub uv: glam::Vec4,
}

impl Default for ModelVertex {
    fn default() -> Self {
        ModelVertex {
            pos: glam::vec4(0f32, 0.0, 0.0, 1.0),
            color: glam::Vec4::splat(1.0),
            normal: glam::Vec4::ZERO,
            uv: glam::Vec4::ZERO,
        }
    }
}

impl Vertex for ModelVertex {
    fn stride() -> u32 {
        std::mem::size_of::<ModelVertex>() as u32
    }
    fn format_offset() -> Vec<(vk::Format, u32)> {
        vec![
            (
                vk::Format::R32G32B32A32_SFLOAT,
                offset_of!(ModelVertex, pos) as u32,
            ),
            (
                vk::Format::R32G32B32A32_SFLOAT,
                offset_of!(ModelVertex, color) as u32,
            ),
            (
                vk::Format::R32G32B32A32_SFLOAT,
                offset_of!(ModelVertex, normal) as u32,
            ),
            (
                vk::Format::R32G32B32A32_SFLOAT,
                offset_of!(ModelVertex, uv) as u32,
            ),
        ]
    }
}

pub struct Mesh {
    pub context: Arc<Context>,
    pub name: String,
    pub vertex_buffer: Buffer,
    pub index_buffer: Option<Buffer>,
    pub index_storage: Option<Buffer>,
    pub transform: glam::Mat4,
    pub primitive_sections: Vec<PrimitiveSection>,
}

impl Mesh {
    pub fn cmd_draw(&self, cmd: vk::CommandBuffer) {
        let device = self.context.device();
        unsafe {
            match &self.index_buffer {
                Some(indices) => {
                    for section in &self.primitive_sections {
                        device.cmd_bind_vertex_buffers(
                            cmd,
                            0,
                            &[self.vertex_buffer.handle()],
                            &[section.get_vertex_offset_size()],
                        );
                        device.cmd_bind_index_buffer(
                            cmd,
                            indices.handle(),
                            section.get_index_offset_size::<u32>(),
                            vk::IndexType::UINT32,
                        );
                        device.cmd_draw_indexed(cmd, section.get_index_count(), 1, 0, 0, 1);
                    }
                }
                None => {
                    for section in &self.primitive_sections {
                        device.cmd_bind_vertex_buffers(
                            cmd,
                            0,
                            &[self.vertex_buffer.handle()],
                            &[section.get_vertex_offset_size()],
                        );
                        device.cmd_draw(cmd, section.get_vertex_count(), 1, 0, 1);
                    }
                }
            }
        }
    }
}
