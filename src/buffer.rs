use crate::{Context, Resource};
use ash::{util::Align, vk};
use std::sync::Arc;
use std::{ffi::c_void, mem::align_of};
use gpu_allocator::{MemoryLocation, vulkan::{Allocation, AllocationCreateDesc}};

#[derive(Clone, Copy)]
pub struct BufferInfo<'a> {
    pub name: &'a str,
    pub usage: vk::BufferUsageFlags,
    pub mem_usage: MemoryLocation,
    pub memory_type_bits: Option<u32>,
    pub index_type: Option<vk::IndexType>,
}

impl std::default::Default for BufferInfo<'_> {
    fn default() -> Self {
        BufferInfo {
            name: "Buffer",
            usage: vk::BufferUsageFlags::default(),
            mem_usage: MemoryLocation::CpuToGpu,
            memory_type_bits: None,
            index_type: None,
        }
    }
}

impl<'a> BufferInfo<'a> {
    pub fn name(mut self, name: &'a str ) -> Self {
        self.name = name;
        self
    }
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
    pub fn gpu_only(mut self) -> Self {
        self.mem_usage = MemoryLocation::GpuOnly;
        self
    }
    pub fn cpu_to_gpu(mut self) -> Self {
        self.mem_usage = MemoryLocation::CpuToGpu;
        self
    }
    pub fn gpu_to_cpu(mut self) -> Self {
        self.mem_usage = MemoryLocation::GpuToCpu;
        self
    }
    pub fn index_type(mut self, index_type: vk::IndexType) -> Self {
        self.index_type = Some(index_type);
        self
    }
    pub fn memory_type_bits(mut self, memory_type_bits: u32) -> Self {
        self.memory_type_bits = Some(memory_type_bits);
        self
    }
}

pub struct Buffer {
    context: Arc<Context>,
    handle: vk::Buffer,
    element_count: u32,
    allocation: Allocation,
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

        let create_info = vk::BufferCreateInfo::builder()
            .size(device_size)
            .usage(info.usage);

        let buffer = unsafe { context.device().create_buffer(&create_info, None) }.unwrap();
        let mut requirements = unsafe { context.device().get_buffer_memory_requirements(buffer) };
        if info.memory_type_bits.is_some() {
            requirements.memory_type_bits |= info.memory_type_bits.unwrap();
        }

        let allocation = context.allocator()
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: info.name,
                requirements,
                location: info.mem_usage,
                linear: true, // Buffers are always linear
            }).unwrap();
        
        // Bind memory to the buffer
        unsafe { context.device().bind_buffer_memory(buffer, allocation.memory(), allocation.offset()).unwrap() };

        Buffer {
            context: context.clone(),
            handle: buffer,
            element_count,
            allocation,
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
        if info.mem_usage == MemoryLocation::GpuOnly {
            create_info.usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }

        let buffer = unsafe { context.device().create_buffer(&create_info, None) }.unwrap();
        let mut requirements = unsafe { context.device().get_buffer_memory_requirements(buffer) };
        if info.memory_type_bits.is_some() {
            requirements.memory_type_bits |= info.memory_type_bits.unwrap();
        }

        let allocation = context.allocator()
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: info.name,
                requirements,
                location: info.mem_usage,
                linear: true, // Buffers are always linear
            }).unwrap();

        // Bind memory to the buffer
        unsafe { context.device().bind_buffer_memory(buffer, allocation.memory(), allocation.offset()).unwrap() };

        let result  = Buffer {
            context: context.clone(),
            handle: buffer,
            element_count: data.len() as u32,
            allocation,
            index_type: info.index_type,
        };

        match info.mem_usage {
            MemoryLocation::GpuOnly => {
                let staging_buffer = Self::new(
                    context.clone(),
                    BufferInfo::default()
                        .cpu_to_gpu()
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                    device_size,
                    1,
                );
                staging_buffer.update(data);

                let cmd = context.begin_single_time_cmd();
                let region = vk::BufferCopy::builder().size(device_size).build();
                unsafe {
                    context
                        .device()
                        .cmd_copy_buffer(cmd, staging_buffer.handle(), buffer, &[region]);
                }
                context.end_single_time_cmd(cmd);
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
            let mapped_data = self.allocation.mapped_ptr().unwrap().as_ptr();
            let mut mapped_slice = Align::new(
                mapped_data as *mut c_void,
                align_of::<T>() as u64,
                size as u64,
            );
            mapped_slice.copy_from_slice(data);
        }
    }

    pub fn map(&self) -> *mut u8 {
        self.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8
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
        self.allocation.size()
    }

    pub unsafe fn get_memory(&self) -> vk::DeviceMemory
    {
        self.allocation.memory()
    }

    pub fn get_offset(&self) -> vk::DeviceSize {
        self.allocation.offset()
    }

    pub fn get_element_count(&self) -> u32 {
        self.element_count
    }

    pub fn get_index_type(&self) -> vk::IndexType {
        self.index_type.unwrap_or_default()
    }

    pub fn get_alloc(&self) -> &Allocation {
        &self.allocation
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
            self.context.device().destroy_buffer(self.handle, None);
        }
        
        let to_drop = std::mem::replace(&mut self.allocation, Allocation::default());
        self.context.allocator()
            .lock()
            .unwrap()
            .free(to_drop).unwrap();
    }
}
