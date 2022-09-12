use crate::{Buffer, BufferInfo, Context};
use ash::vk;
use std::sync::Arc;

// https://developer.nvidia.com/rtx/raytracing/vkray_helpers
// https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/#shaderbindingtable
// This implementation is now mostly lifted from https://github.com/EmbarkStudios/kajiya/blob/main/crates/lib/kajiya-backend/src/vulkan/ray_tracing.rs

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

    fn raygen_count(&self) -> usize {
        self.miss_indices.len()
    }
    fn miss_count(&self) -> usize {
        self.raygen_indices.len()
    }
    fn hitgroup_count(&self) -> usize {
        self.hit_group_indices.len()
    }
    fn get_total_group_count(&self) -> usize {
        self.raygen_count() + self.miss_count() + self.hitgroup_count()
    }
}

// Always internally stores raygen -> miss -> hit groups.
pub struct ShaderBindingTable {
    context: Arc<Context>,
    pub raygen_sbt_address: vk::StridedDeviceAddressRegionKHR,
    pub raygen_sbt_buffer: Option<Buffer>,
    pub miss_sbt_address: vk::StridedDeviceAddressRegionKHR,
    pub miss_sbt_buffer: Option<Buffer>,
    pub hit_sbt_address: vk::StridedDeviceAddressRegionKHR,
    pub hit_sbt_buffer: Option<Buffer>,
    pub callable_sbt_address: vk::StridedDeviceAddressRegionKHR,
    pub callable_sbt_buffer: Option<Buffer>,
}

impl ShaderBindingTable {
    pub fn new(context: Arc<Context>,  pipeline: vk::Pipeline, info: ShaderBindingTableInfo) -> Self {
        let shader_group_handle_size = 
            unsafe{ context.ray_tracing_properties().shader_group_handle_size as usize };
        let group_count = info.get_total_group_count() as usize;
        let group_handles_size = (shader_group_handle_size * group_count) as usize;

        let group_handles: Vec<u8> = unsafe {
            context.ray_tracing()
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    group_count as _,
                    group_handles_size,
                ).unwrap()
        };

        let prog_size = shader_group_handle_size;

        let create_binding_table =
            |context: Arc<Context>, entry_offset: u32, entry_count: u32|
             -> Option<Buffer> {
                if 0 == entry_count {
                    return None;
                }

                let mut sbt_data =
                    vec![0u8; (entry_count as usize * prog_size) as _];

                for dst in 0..(entry_count as usize) {
                    let src = dst + entry_offset as usize;
                    sbt_data
                        [dst * prog_size..dst * prog_size + shader_group_handle_size]
                        .copy_from_slice(
                            &group_handles[src * shader_group_handle_size
                                ..src * shader_group_handle_size + shader_group_handle_size],
                        );
                }

                Some(Buffer::from_data(
                    context.clone(),
                    BufferInfo::default().gpu_only().usage(
                        vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
                    ),
                    &sbt_data
                ))
            };

        let raygen_sbt_buffer = create_binding_table(context.clone(), 0, info.raygen_count() as u32);
        let miss_sbt_buffer = create_binding_table(context.clone(), 
            info.raygen_count() as u32,
            info.miss_count() as u32);
        let hit_sbt_buffer = create_binding_table(context.clone(),
            (info.raygen_count() + info.miss_count()) as u32,
            info.hitgroup_count() as u32,
        );

        ShaderBindingTable {
            context,
            raygen_sbt_address: vk::StridedDeviceAddressRegionKHR {
                device_address: raygen_sbt_buffer
                    .as_ref()
                    .map(|b| b.get_device_address())
                    .unwrap_or(0),
                stride: prog_size as u64,
                size: (prog_size * info.raygen_count() as usize) as u64,
            },
            raygen_sbt_buffer,
            miss_sbt_address: vk::StridedDeviceAddressRegionKHR {
                device_address: miss_sbt_buffer
                    .as_ref()
                    .map(|b| b.get_device_address())
                    .unwrap_or(0),
                stride: prog_size as u64,
                size: (prog_size * info.miss_count() as usize) as u64,
            },
            miss_sbt_buffer,
            hit_sbt_address: vk::StridedDeviceAddressRegionKHR {
                device_address: hit_sbt_buffer
                    .as_ref()
                    .map(|b| b.get_device_address())
                    .unwrap_or(0),
                stride: prog_size as u64,
                size: (prog_size * info.hitgroup_count() as usize) as u64,
            },
            hit_sbt_buffer,
            callable_sbt_address: vk::StridedDeviceAddressRegionKHR {
                device_address: Default::default(),
                stride: 0,
                size: 0,
            },
            callable_sbt_buffer: None,
        }
    }

    pub fn cmd_trace_rays(&self, cmd: vk::CommandBuffer, extent: vk::Extent3D) {
        unsafe {
            self.context.ray_tracing().cmd_trace_rays(
                cmd,
                &self.raygen_sbt_address,
                &self.miss_sbt_address,
                &self.hit_sbt_address,
                &self.callable_sbt_address,
                extent.width,
                extent.height,
                extent.depth,
            );
        }
    }
}
