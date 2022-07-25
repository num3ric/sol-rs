use crate::{Buffer, BufferInfo, Context, Resource};
use ash::vk;
use std::sync::Arc;

// https://developer.nvidia.com/rtx/raytracing/vkray_helpers
// https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/#shaderbindingtable
// TODO: Support 

pub fn align_up(x: u32, a: u32) -> u32 {
    (x + (a - 1)) & !(a - 1)
}

pub struct ShaderBindingTableInfo {
    pub raygen_indices: Vec<u64>,
    pub miss_indices: Vec<u64>,
    pub hit_group_indices: Vec<u64>,
}

impl Default for ShaderBindingTableInfo {
    fn default() -> Self {
        ShaderBindingTableInfo {
            raygen_indices: Vec::new(),
            miss_indices: Vec::new(),
            hit_group_indices: Vec::new(),
        }
    }
}

impl ShaderBindingTableInfo {
    pub fn raygen(mut self, index: u64) -> Self {
        self.raygen_indices.push(index);
        self
    }
    pub fn miss(mut self, index: u64) -> Self {
        self.miss_indices.push(index);
        self
    }
    pub fn hitgroup(mut self, index: u64) -> Self {
        self.hit_group_indices.push(index);
        self
    }
}

// Always internally stores raygen -> miss -> hit groups.
pub struct ShaderBindingTable {
    context: Arc<Context>,
    buffer: Option<Buffer>,
    handle_size: u64,
    handle_size_aligned: u64,
    info: ShaderBindingTableInfo,
}

impl ShaderBindingTable {
    pub fn new(context: Arc<Context>, info: ShaderBindingTableInfo) -> Self {
        let handle_size =
            unsafe { context.ray_tracing_properties().shader_group_handle_size } as u64;
        let group_size_align =
            unsafe { context.ray_tracing_properties().shader_group_base_alignment };
        let handle_size_aligned = align_up(handle_size as u32, group_size_align) as u64;

        ShaderBindingTable {
            context,
            buffer: None,
            handle_size,
            handle_size_aligned,
            info: info,
        }
    }

    pub fn generate(&mut self, pipeline: vk::Pipeline) {
        // Clear/reset buffer
        self.buffer = None;

        let group_count = self.get_total_group_count() as u64;
        let sbt_size = self.handle_size_aligned * group_count;
        let sbt = Buffer::new(
            self.context.clone(),
            BufferInfo::default().cpu_to_gpu().usage(
                vk::BufferUsageFlags::TRANSFER_SRC,
                // | vk::BufferUsageFlags::RAY_TRACING_NV,
                // | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            ),
            sbt_size,
            group_count as u32,
        );
        let mut src_data = vec![0u8; sbt_size as usize];
        unsafe {
            self.context
                .ray_tracing()
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    group_count as u32,
                    &mut src_data,
                )
                .unwrap();
        }
        let dst_data = sbt.map();
        let mut dst_offset: u64 = 0;
        // Raygen handles
        dst_offset += self.copy_shader_data(
            &self.info.raygen_indices,
            src_data.as_ptr(),
            dst_offset,
            dst_data,
        );
        // Ray hiss handles
        dst_offset += self.copy_shader_data(
            &self.info.miss_indices,
            src_data.as_ptr(),
            dst_offset,
            dst_data,
        );
        // Hit group handles
        self.copy_shader_data(
            &self.info.hit_group_indices,
            src_data.as_ptr(),
            dst_offset,
            dst_data,
        );
        self.buffer = Some(sbt);
    }

    fn copy_shader_data(
        &self,
        indices: &Vec<u64>,
        src_data: *const u8,
        dst_offset: u64,
        dst_data: *mut u8,
    ) -> u64 {
        for index in indices {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_data.offset(*index as isize * self.handle_size as isize),
                    dst_data.offset(dst_offset as isize),
                    self.handle_size as usize,
                );
            }
        }
        // Return the number of bytes actually written to the output buffer
        self.handle_size_aligned as u64 * indices.len() as u64
    }

    pub fn get_raygen_offset(&self) -> vk::DeviceSize {
        return 0;
    }

    pub fn get_raygen_section_size(&self) -> vk::DeviceSize {
        self.handle_size_aligned as u64 * self.info.raygen_indices.len() as u64
    }

    pub fn get_miss_offset(&self) -> vk::DeviceSize {
        self.get_raygen_section_size()
    }

    pub fn get_miss_section_size(&self) -> vk::DeviceSize {
        self.handle_size_aligned as u64 * self.info.miss_indices.len() as u64
    }

    pub fn get_hit_group_offset(&self) -> u64 {
        self.get_raygen_section_size() + self.get_miss_section_size()
    }

    pub fn get_hit_group_section_size(&self) -> u64 {
        self.handle_size_aligned as u64 * self.info.hit_group_indices.len() as u64
    }

    fn get_total_group_count(&self) -> usize {
        self.info.raygen_indices.len()
            + self.info.miss_indices.len()
            + self.info.hit_group_indices.len()
    }

    pub fn cmd_trace_rays(&self, cmd: vk::CommandBuffer, extent: vk::Extent3D) {
        unsafe {
            self.context.ray_tracing().cmd_trace_rays(
                cmd,
                self.handle(),
                self.get_raygen_offset(),
                self.handle(),
                self.get_miss_offset(),
                self.get_miss_section_size(),
                self.handle(),
                self.get_hit_group_offset(),
                self.get_hit_group_section_size(),
                vk::Buffer::null(),
                0,
                0,
                extent.width,
                extent.height,
                extent.depth,
            );
        }
    }
}

impl Resource<vk::Buffer> for ShaderBindingTable {
    fn handle(&self) -> vk::Buffer {
        self.buffer.as_ref().unwrap().handle()
    }
}
