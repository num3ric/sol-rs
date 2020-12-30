// Much of this was directly based on:
// https://github.com/adrien-ben/gltf-viewer-rs/blob/master/model/src/mesh.rs

use crate::{offset_of, Buffer, BufferInfo, Context, Resource, Vertex};
use ash::{version::DeviceV1_0, vk};
use gltf::{
    buffer::Buffer as GltfBuffer,
    mesh::{Reader, Semantic},
};
// use std::error::Error;
use std::path::PathBuf;
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

impl std::default::Default for ModelVertex {
    fn default() -> Self {
        ModelVertex {
            pos: glam::vec4(0f32, 0.0, 0.0, 1.0),
            color: glam::Vec4::splat(1.0),
            normal: glam::Vec4::zero(),
            uv: glam::Vec4::zero(),
        }
    }
}

impl Vertex for ModelVertex {
    fn stride() -> u32 {
        std::mem::size_of::<ModelVertex>() as u32
    }
    fn format_offset() -> Vec<(ash::vk::Format, u32)> {
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MaterialInfo {
    pub base_color: glam::Vec4,
    pub emissive_factor: glam::Vec3,
    pub padding0: f32,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub padding1: f32,
    pub padding2: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct BufferPart {
    pub offset: usize,
    pub element_count: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct PrimitiveSection {
    index: usize,
    vertices: BufferPart,
    indices: Option<BufferPart>,
    material_index: Option<usize>,
    //aabb: AABB<f32>,
}

impl PrimitiveSection {
    pub fn get_index_descriptor<T>(&self, buffer: &Buffer) -> vk::DescriptorBufferInfo {
        let size = std::mem::size_of::<T>() as u64;
        buffer.get_descriptor_info_offset(
            self.indices.unwrap().offset as u64 * size,
            self.indices.unwrap().element_count as u64 * size,
        )
    }

    pub fn get_vertex_descriptor(&self, buffer: &Buffer) -> vk::DescriptorBufferInfo {
        let size = std::mem::size_of::<ModelVertex>() as u64;
        buffer.get_descriptor_info_offset(
            self.vertices.offset as u64 * size,
            self.vertices.element_count as u64 * size,
        )
    }

    pub fn get_material_descriptor(&self, buffer: &Buffer) -> vk::DescriptorBufferInfo {
        let size = std::mem::size_of::<MaterialInfo>() as u64;
        buffer.get_descriptor_info_offset(self.material_index.unwrap() as u64 * size, size)
    }

    pub fn get_vertices(&self) -> &BufferPart {
        &self.vertices
    }

    pub fn get_vertex_count(&self) -> u32 {
        self.vertices.element_count as u32
    }

    pub fn get_vertex_offset_size(&self) -> vk::DeviceSize {
        let size = std::mem::size_of::<ModelVertex>() as u64;
        self.vertices.offset as u64 * size
    }

    pub fn get_indices(&self) -> &Option<BufferPart> {
        &self.indices
    }

    pub fn get_index_count(&self) -> u32 {
        self.indices.unwrap().element_count as u32
    }

    pub fn get_index_offset_size<T>(&self) -> vk::DeviceSize {
        let size = std::mem::size_of::<T>() as u64;
        self.indices.unwrap().offset as u64 * size
    }
}

pub struct Mesh {
    context: Arc<Context>,
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

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<MaterialInfo>,
    pub material_buffer: Buffer,
    pub transform: glam::Mat4,
}

pub fn load_model(context: Arc<Context>, filepath: &PathBuf) -> Model {
    let mut meshes = Vec::<Mesh>::new();
    let (gltf, buffers, _) = gltf::import(filepath).unwrap();

    //println!("{:#?}", gltf);

    let mut materials = Vec::<MaterialInfo>::new();
    for mat in gltf.materials() {
        materials.push(MaterialInfo {
            base_color: glam::Vec4::from_slice_unaligned(
                &mat.pbr_metallic_roughness().base_color_factor(),
            ),
            //double_sided: mat.double_sided(),
            metallic_factor: mat.pbr_metallic_roughness().metallic_factor(),
            roughness_factor: mat.pbr_metallic_roughness().roughness_factor(),
            emissive_factor: glam::Vec3::from_slice_unaligned(&mat.emissive_factor()),
            ..Default::default()
        });
    }
    let material_buffer = Buffer::from_data(
        context.clone(),
        BufferInfo::default().usage_storage().gpu_only(),
        &materials,
    );

    for mesh in gltf.meshes() {
        let mut mesh_indices = Vec::<u32>::new();
        let mut mesh_vertices = Vec::<ModelVertex>::new();
        let mut primitive_sections = Vec::<PrimitiveSection>::new();

        // println!("Mesh #{}", mesh.index());

        for (primitive_index, primitive) in mesh.primitives().enumerate() {
            // println!("- Primitive #{}", primitive.index());

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let offset = mesh_vertices.len();

            if let Some(_) = primitive.get(&Semantic::Positions) {
                let positions = read_positions(&reader);
                let normals = read_normals(&reader);
                let tex_coords_0 = read_tex_coords(&reader, 0);
                let colors = read_colors(&reader);

                positions.iter().enumerate().for_each(|(index, position)| {
                    let pos = *position;
                    let norm = *normals.get(index).unwrap_or(&[0.0, 1.0, 0.0]);
                    let uv = *tex_coords_0.get(index).unwrap_or(&[0.0, 0.0]);
                    let col = *colors.get(index).unwrap_or(&[1.0, 1.0, 1.0, 1.0]);
                    mesh_vertices.push(ModelVertex {
                        pos: glam::vec4(pos[0], pos[1], pos[2], 1.0),
                        normal: glam::vec4(norm[0], norm[1], norm[2], 1.0),
                        color: glam::vec4(col[0], col[1], col[2], col[3]),
                        uv: glam::vec4(uv[0], uv[1], 0.0, 0.0),
                    });
                });
            };

            primitive_sections.push(PrimitiveSection {
                index: primitive_index,
                vertices: BufferPart {
                    offset,
                    element_count: mesh_vertices.len() - offset,
                },
                indices: None,
                material_index: primitive.material().index(),
            });
            // println!("  Vertices {:?}", (offset, mesh_vertices.len() - offset));

            if let Some(iter) = reader.read_indices() {
                let offset = mesh_indices.len();
                mesh_indices.extend(iter.into_u32());
                primitive_sections.last_mut().unwrap().indices = Some(BufferPart {
                    offset,
                    element_count: mesh_indices.len() - offset,
                });
                // println!("    Indices {:?}", (offset, mesh_indices.len() - offset));
            }
        }

        let mut index_buffer = None;
        let mut index_storage = None;

        if !mesh_indices.is_empty() {
            index_buffer = Some(Buffer::from_data(
                context.clone(),
                BufferInfo::default().usage_index().gpu_only(),
                &mesh_indices,
            ));

            let storage_indices: Vec<u64> = mesh_indices.iter().map(|i| *i as u64).collect();
            index_storage = Some(Buffer::from_data(
                context.clone(),
                BufferInfo::default().usage_storage().gpu_only(),
                &storage_indices,
            ));
        }
        let vertex_buffer = Buffer::from_data(
            context.clone(),
            BufferInfo::default()
                .usage_vertex()
                .usage_storage()
                .gpu_only(),
            &mesh_vertices,
        );

        let mut node_transform = glam::Mat4::identity();
        for node in gltf.nodes() {
            let matched = match node.mesh() {
                Some(node_mesh) => node_mesh.index() == mesh.index(),
                None => false,
            };
            if matched {
                node_transform = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
                break;
            }
        }

        let name = match mesh.name() {
            Some(name) => name.to_owned(),
            None => String::new(),
        };
        meshes.push(Mesh {
            context: context.clone(),
            name,
            index_buffer,
            index_storage,
            vertex_buffer,
            transform: node_transform,
            primitive_sections,
        });
    }

    let mut transform = glam::Mat4::identity();
    for node in gltf.nodes() {
        transform = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
        break;
    }
    Model {
        meshes,
        materials,
        material_buffer,
        transform,
    }
}

fn read_indices<'a, 's, F>(reader: &Reader<'a, 's, F>) -> Option<Vec<u32>>
where
    F: Clone + Fn(GltfBuffer<'a>) -> Option<&'s [u8]>,
{
    reader
        .read_indices()
        .map(|indices| indices.into_u32().collect::<Vec<_>>())
}

fn read_positions<'a, 's, F>(reader: &Reader<'a, 's, F>) -> Vec<[f32; 3]>
where
    F: Clone + Fn(GltfBuffer<'a>) -> Option<&'s [u8]>,
{
    reader
        .read_positions()
        .expect("Position primitives should be present")
        .collect()
}

fn read_normals<'a, 's, F>(reader: &Reader<'a, 's, F>) -> Vec<[f32; 3]>
where
    F: Clone + Fn(GltfBuffer<'a>) -> Option<&'s [u8]>,
{
    reader
        .read_normals()
        .map_or(vec![], |normals| normals.collect())
}

fn read_tex_coords<'a, 's, F>(reader: &Reader<'a, 's, F>, channel: u32) -> Vec<[f32; 2]>
where
    F: Clone + Fn(GltfBuffer<'a>) -> Option<&'s [u8]>,
{
    reader
        .read_tex_coords(channel)
        .map_or(vec![], |coords| coords.into_f32().collect())
}

fn read_colors<'a, 's, F>(reader: &Reader<'a, 's, F>) -> Vec<[f32; 4]>
where
    F: Clone + Fn(GltfBuffer<'a>) -> Option<&'s [u8]>,
{
    reader
        .read_colors(0)
        .map_or(vec![], |colors| colors.into_rgba_f32().collect())
}
