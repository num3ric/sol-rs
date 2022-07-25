use crate::Context;
use ash::{util::Align, vk};
use std::sync::Arc;
use std::{ffi::c_void, mem::align_of, slice::from_raw_parts_mut};

#[derive(Clone, Copy)]
pub struct BufferInfo {
    pub usage: vk::BufferUsageFlags,
    pub mem_usage: vk_mem::MemoryUsage,
    pub memory_type_bits: u32,
    pub index_type: Option<vk::IndexType>,
    pub flags: Option<vk_mem::AllocationCreateFlags>,
}

impl std::default::Default for BufferInfo {
    fn default() -> Self {
        BufferInfo {
            usage: vk::BufferUsageFlags::default(),
            mem_usage: vk_mem::MemoryUsage::CpuToGpu,
            memory_type_bits: 0,
            index_type: None,
            flags: None,
        }
    }
}

impl BufferInfo {
    pub fn usage(mut self, usage: vk::BufferUsageFlags) -> Self {
        self.usage = usage;
        self
    }
    pub fn usage_transfer_src(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::TRANSFER_SRC;
        self
    }
    pub fn usage_transfer_dst(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::TRANSFER_DST;
        self
    }
    pub fn usage_uniform_texel(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER;
        self
    }
    pub fn usage_storage_texel(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER;
        self
    }
    pub fn usage_uniform(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        self
    }
    pub fn usage_storage(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
        self
    }
    pub fn usage_index(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::INDEX_BUFFER;
        self
    }
    pub fn usage_vertex(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::VERTEX_BUFFER;
        self
    }
    pub fn usage_indirect(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
        self
    }
    pub fn cpu_only(mut self) -> Self {
        self.mem_usage = vk_mem::MemoryUsage::CpuOnly;
        self
    }
    pub fn gpu_only(mut self) -> Self {
        self.mem_usage = vk_mem::MemoryUsage::GpuOnly;
        self
    }
    pub fn cpu_to_gpu(mut self) -> Self {
        self.mem_usage = vk_mem::MemoryUsage::CpuToGpu;
        self
    }
    pub fn gpu_to_cpu(mut self) -> Self {
        self.mem_usage = vk_mem::MemoryUsage::GpuToCpu;
        self
    }
    pub fn index_type(mut self, index_type: vk::IndexType) -> Self {
        self.index_type = Some(index_type);
        self
    }
    pub fn alloc_flags(mut self, flags: vk_mem::AllocationCreateFlags) -> Self {
        self.flags = Some(flags);
        self
    }
    pub fn memory_type_bits(mut self, memory_type_bits: u32) -> Self {
        self.memory_type_bits = memory_type_bits;
        self
    }
}

pub struct Buffer {
    context: Arc<Context>,
    handle: vk::Buffer,
    element_count: u32,
    allocation: vk_mem::Allocation,
    allocation_info: vk_mem::AllocationInfo,
    index_type: Option<vk::IndexType>,
}

impl Buffer {
    pub fn new(
        context: Arc<Context>,
        info: BufferInfo,
        device_size: vk::DeviceSize,
        element_count: u32,
    ) -> Self {
        assert_ne!(device_size, 0);

        let buffer_info = vk::BufferCreateInfo::builder()
            .size(device_size)
            .usage(info.usage);
        let mut create_info = vk_mem::AllocationCreateInfo::default();
        create_info.memory_type_bits = info.memory_type_bits;
        create_info.usage = info.mem_usage;

        match info.flags {
            Some(flags) => {
                create_info.flags = flags;
            }
            None => {}
        }
        let (buffer, allocation, allocation_info) = unsafe {
            context
                .allocator()
                .create_buffer(&buffer_info, &create_info)
                .unwrap()
        };
        Buffer {
            context: context.clone(),
            handle: buffer,
            element_count,
            allocation,
            allocation_info,
            index_type: info.index_type,
        }
    }

    pub fn from_data<T: std::marker::Copy>(
        context: Arc<Context>,
        info: BufferInfo,
        data: &[T],
    ) -> Self {
        assert!(!data.is_empty());

        let device_size = std::mem::size_of_val(data) as u64;
        let mut create_info = vk::BufferCreateInfo::builder()
            .size(device_size)
            .usage(info.usage);
        if info.mem_usage == vk_mem::MemoryUsage::GpuOnly {
            create_info.usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        let mut allocation_info = vk_mem::AllocationCreateInfo::default();
        allocation_info.usage = info.mem_usage;
        //allocation_info.flags = vk_mem::AllocationCreateFlags::MAPPED;

        let (buffer, allocation, allocation_info) = unsafe {
            context
                .allocator()
                .create_buffer(&create_info, &allocation_info)
                .unwrap()
        };
        assert_eq!(allocation_info.get_mapped_data(), std::ptr::null_mut());

        let result = Buffer {
            context: context.clone(),
            handle: buffer,
            element_count: data.len() as u32,
            allocation,
            allocation_info,
            index_type: info.index_type,
        };

        match info.mem_usage {
            vk_mem::MemoryUsage::GpuOnly => {
                let staging_info = vk::BufferCreateInfo::builder()
                    .size(device_size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC);
                let mut staging_alloc_info = vk_mem::AllocationCreateInfo::default();
                staging_alloc_info.usage = vk_mem::MemoryUsage::CpuOnly;
                staging_alloc_info.flags = vk_mem::AllocationCreateFlags::MAPPED;
                let (staging, staging_alloc, staging_alloc_info) = unsafe {
                    context
                        .allocator()
                        .create_buffer(&staging_info, &staging_alloc_info)
                        .unwrap()
                };
                let mem_ptr = staging_alloc_info.get_mapped_data();
                unsafe {
                    let mapped_slice = from_raw_parts_mut(mem_ptr as *mut T, data.len());
                    mapped_slice.copy_from_slice(data);
                }

                let cmd = context.begin_single_time_cmd();
                let region = vk::BufferCopy::builder().size(device_size).build();
                unsafe {
                    context
                        .device()
                        .cmd_copy_buffer(cmd, staging, buffer, &[region]);
                }
                context.end_single_time_cmd(cmd);

                unsafe {
                    context.allocator().destroy_buffer(staging, staging_alloc);
                }
            }
            _ => {
                result.update(data);
            }
        }
        result
    }

    pub fn update<T: Copy>(&self, data: &[T]) {
        let size = std::mem::size_of_val(&data[0]) * data.len();
        unsafe {
            let mapped_data = self
                .context
                .allocator()
                .map_memory(self.allocation)
                .unwrap();
            let mut mapped_slice = Align::new(
                mapped_data as *mut c_void,
                align_of::<T>() as u64,
                size as u64,
            );
            mapped_slice.copy_from_slice(data);
            //mapped_data.copy_from(data.as_ptr() as *const u8, size);
            self.context.allocator().unmap_memory(self.allocation);
        }
    }

    pub fn map(&self) -> *mut u8 {
        unsafe {
            self.context
                .allocator()
                .map_memory(self.allocation)
                .unwrap()
        }
    }

    pub fn unmap(&self) {
        unsafe {
            self.context.allocator().unmap_memory(self.allocation);
        }
    }

    pub fn get_descriptor_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::builder()
            .buffer(self.handle)
            .offset(0)
            .range(vk::WHOLE_SIZE)
            .build()
    }

    pub fn get_descriptor_info_offset(
        &self,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::builder()
            .buffer(self.handle)
            .offset(offset)
            .range(range)
            .build()
    }

    pub fn get_size(&self) -> vk::DeviceSize {
        self.allocation_info.get_size() as u64
    }

    pub fn get_element_count(&self) -> u32 {
        self.element_count
    }

    pub fn get_index_type(&self) -> vk::IndexType {
        self.index_type.unwrap_or_default()
    }

    pub fn get_alloc_info(&self) -> &vk_mem::AllocationInfo {
        &self.allocation_info
    }

    pub fn get_device_address(&self) -> u64 {
        unsafe {
            self.context.device().get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::builder().buffer(self.handle),
            )
        }
    }
}

impl crate::Resource<vk::Buffer> for Buffer {
    fn handle(&self) -> vk::Buffer {
        self.handle
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.context
                .allocator()
                .destroy_buffer(self.handle, self.allocation);
        }
    }
}
