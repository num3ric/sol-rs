use crate::{Buffer, BufferInfo, Context, Resource};
use ash::{vk};
use std::sync::Arc;

pub struct GeometryInstance {
    pub vertex_buffer: vk::Buffer,
    pub vertex_count: u32,
    pub vertex_offset: vk::DeviceSize,
    pub index_buffer: Option<vk::Buffer>,
    pub index_count: Option<u32>,
    pub index_offset: Option<vk::DeviceSize>,
    pub transform: glam::Mat4,
}

struct AccelerationStructure {
    context: Arc<Context>,
    accel_struct: vk::AccelerationStructureNV,
    accel_struct_flags: vk::BuildAccelerationStructureFlagsNV,
    scratch_buffer: Buffer,
    buffer: Buffer,
}

struct MemorySpec {
    size: vk::DeviceSize,
    type_bits: u32,
}

impl AccelerationStructure {
    fn compute_buffer_memory(
        context: &Arc<Context>,
        acceleration_structure: vk::AccelerationStructureNV,
    ) -> (MemorySpec, MemorySpec) {
        let result: MemorySpec;
        let scratch: MemorySpec;
        unsafe {
            let info = vk::AccelerationStructureMemoryRequirementsInfoNV::builder()
                .acceleration_structure(acceleration_structure)
                .ty(vk::AccelerationStructureMemoryRequirementsTypeNV::OBJECT);
            let mem_reqs = context
                .ray_tracing()
                .get_acceleration_structure_memory_requirements(&info)
                .memory_requirements;
            result = MemorySpec {
                size: mem_reqs.size,
                type_bits: mem_reqs.memory_type_bits,
            };

            let info = vk::AccelerationStructureMemoryRequirementsInfoNV::builder()
                .acceleration_structure(acceleration_structure)
                .ty(vk::AccelerationStructureMemoryRequirementsTypeNV::BUILD_SCRATCH);
            let mem_reqs = context
                .ray_tracing()
                .get_acceleration_structure_memory_requirements(&info)
                .memory_requirements;
            let mut scratch_size = mem_reqs.size;

            let info = vk::AccelerationStructureMemoryRequirementsInfoNV::builder()
                .acceleration_structure(acceleration_structure)
                .ty(vk::AccelerationStructureMemoryRequirementsTypeNV::UPDATE_SCRATCH);
            let scratch_update_size = context
                .ray_tracing()
                .get_acceleration_structure_memory_requirements(&info)
                .memory_requirements
                .size;

            scratch_size = if scratch_size > scratch_update_size {
                scratch_size
            } else {
                scratch_update_size
            };

            scratch = MemorySpec {
                size: scratch_size,
                type_bits: mem_reqs.memory_type_bits,
            };
        }
        (result, scratch)
    }
}

impl crate::Resource<vk::AccelerationStructureNV> for AccelerationStructure {
    fn handle(&self) -> vk::AccelerationStructureNV {
        self.accel_struct
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.context
                .ray_tracing()
                .destroy_acceleration_structure(self.accel_struct, None);
        }
    }
}

pub struct BLAS {
    accel_struct: AccelerationStructure,
    geometries: Vec<vk::GeometryNV>,
    transform: glam::Mat4,
    hit_group_index: u32,
}

impl BLAS {
    pub fn new(
        context: Arc<Context>,
        cmd: vk::CommandBuffer,
        geo_intances: Vec<GeometryInstance>,
        transform: glam::Mat4,
        vertex_stride: vk::DeviceSize,
        is_opaque: bool,
    ) -> Self {
        let mut geometries = Vec::<vk::GeometryNV>::new();
        for ref geo in geo_intances {
            let flags = match is_opaque {
                true => vk::GeometryFlagsNV::OPAQUE_NV,
                false => vk::GeometryFlagsNV::empty(),
            };
            let geo_triangles = match geo.index_buffer {
                Some(_) => {
                    vk::GeometryTrianglesNV::builder()
                        .vertex_data(geo.vertex_buffer)
                        .vertex_offset(geo.vertex_offset)
                        .vertex_count(geo.vertex_count)
                        .vertex_stride(vertex_stride)
                        .vertex_format(vk::Format::R32G32B32_SFLOAT) //TODO: get from buffer
                        .index_data(geo.index_buffer.unwrap())
                        .index_offset(geo.index_offset.unwrap())
                        .index_count(geo.index_count.unwrap())
                        .index_type(vk::IndexType::UINT32) //TODO: get from buffer
                        .build()
                }
                None => {
                    vk::GeometryTrianglesNV::builder()
                        .vertex_data(geo.vertex_buffer)
                        .vertex_offset(geo.vertex_offset)
                        .vertex_count(geo.vertex_count)
                        .vertex_stride(vertex_stride)
                        .vertex_format(vk::Format::R32G32B32_SFLOAT) //TODO: get from buffer
                        .index_type(vk::IndexType::UINT32) //TODO: get from buffer
                        .build()
                }
            };
            geometries.push(
                vk::GeometryNV::builder()
                    .geometry_type(vk::GeometryTypeNV::TRIANGLES_NV)
                    .geometry(
                        vk::GeometryDataNV::builder()
                            .triangles(geo_triangles)
                            .build(),
                    )
                    .flags(flags)
                    .build(),
            );
        }

        let (accel_struct, accel_struct_flags) = Self::create_accel_struct(
            &context,
            &geometries,
            vk::BuildAccelerationStructureFlagsNV::PREFER_FAST_TRACE_NV,
        );

        let (result_specs, scratch_specs) =
            AccelerationStructure::compute_buffer_memory(&context, accel_struct);

        let buffer = Buffer::new(
            context.clone(),
            BufferInfo::default()
                .gpu_only()
                .usage(vk::BufferUsageFlags::RAY_TRACING_NV)
                .memory_type_bits(result_specs.type_bits),
            result_specs.size,
            1,
        );

        let scratch_buffer = Buffer::new(
            context.clone(),
            BufferInfo::default()
                .gpu_only()
                .usage(vk::BufferUsageFlags::RAY_TRACING_NV)
                .memory_type_bits(scratch_specs.type_bits),
            scratch_specs.size,
            1,
        );

        unsafe {
            // Bind the acceleration structure descriptor to the actual memory that will store the AS itself
            let bind_info = vk::BindAccelerationStructureMemoryInfoNV::builder()
                .acceleration_structure(accel_struct)
                .memory(buffer.get_alloc_info().get_device_memory())
                .memory_offset(buffer.get_alloc_info().get_offset() as u64)
                .build();
            context
                .ray_tracing()
                .bind_acceleration_structure_memory(&[bind_info])
                .unwrap();

            let info = vk::AccelerationStructureInfoNV::builder()
                .flags(accel_struct_flags)
                .ty(vk::AccelerationStructureTypeNV::BOTTOM_LEVEL)
                .geometries(&geometries)
                .instance_count(0);
            let previous = vk::AccelerationStructureNV::null();
            context.ray_tracing().cmd_build_acceleration_structure(
                cmd,
                &info,
                vk::Buffer::null(),
                0,
                false,
                accel_struct,
                previous,
                scratch_buffer.handle(),
                0,
            );

            let memory_barrier = vk::MemoryBarrier::builder()
                .src_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_NV
                        | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_NV,
                )
                .dst_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_NV
                        | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_NV,
                )
                .build();
            context.device().cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
        }

        BLAS {
            accel_struct: AccelerationStructure {
                context,
                accel_struct,
                accel_struct_flags,
                scratch_buffer,
                buffer,
            },
            transform,
            geometries,
            hit_group_index: 0,
        }
    }

    fn create_accel_struct(
        context: &Arc<Context>,
        geometries: &Vec<vk::GeometryNV>,
        flags: vk::BuildAccelerationStructureFlagsNV,
    ) -> (
        vk::AccelerationStructureNV,
        vk::BuildAccelerationStructureFlagsNV,
    ) {
        let create_info = vk::AccelerationStructureCreateInfoNV::builder()
            .info(
                vk::AccelerationStructureInfoNV::builder()
                    .ty(vk::AccelerationStructureTypeNV::BOTTOM_LEVEL)
                    .flags(flags)
                    .geometries(geometries)
                    .build(),
            )
            .compacted_size(0)
            .build();
        let accel_struct = unsafe {
            context
                .ray_tracing()
                .create_acceleration_structure(&create_info, None)
                .unwrap()
        };
        (accel_struct, flags)
    }

    pub fn get_transform(&self) -> glam::Mat4 {
        self.transform
    }

    pub fn set_transform(&mut self, transform: glam::Mat4) {
        self.transform = transform
    }
}

impl crate::Resource<vk::AccelerationStructureNV> for BLAS {
    fn handle(&self) -> vk::AccelerationStructureNV {
        self.accel_struct.handle()
    }
}

#[repr(C)]
#[derive(Clone, Debug, Copy)]
struct InstanceDescriptor {
    transform: [f32; 12],
    instance_id_and_mask: u32,
    instance_offset_and_flags: u32,
    acceleration_handle: u64,
}

impl InstanceDescriptor {
    fn new(
        transform: [f32; 12],
        id: u32,
        mask: u8,
        offset: u32,
        flags: vk::GeometryInstanceFlagsNV,
        acceleration_handle: u64,
    ) -> Self {
        let mut instance = InstanceDescriptor {
            transform,
            instance_id_and_mask: 0,
            instance_offset_and_flags: 0,
            acceleration_handle,
        };
        instance.set_id(id);
        instance.set_mask(mask);
        instance.set_offset(offset);
        instance.set_flags(flags);
        instance
    }

    fn set_id(&mut self, id: u32) {
        let id = id & 0x00ffffff;
        self.instance_id_and_mask |= id;
    }

    fn set_mask(&mut self, mask: u8) {
        let mask = mask as u32;
        self.instance_id_and_mask |= mask << 24;
    }

    fn set_offset(&mut self, offset: u32) {
        let offset = offset & 0x00ffffff;
        self.instance_offset_and_flags |= offset;
    }

    fn set_flags(&mut self, flags: vk::GeometryInstanceFlagsNV) {
        let flags = flags.as_raw() as u32;
        self.instance_offset_and_flags |= flags << 24;
    }
}

pub struct TLAS {
    context: Arc<Context>,
    instance_buffer: Buffer,
    accel_struct: AccelerationStructure,
}

impl TLAS {
    pub fn new(context: Arc<Context>, cmd: vk::CommandBuffer, blas: &[BLAS]) -> Self {
        let (accel_struct, accel_struct_flags) = Self::create_accel_struct(
            &context,
            blas.len() as u32,
            vk::BuildAccelerationStructureFlagsNV::ALLOW_UPDATE_NV,
        );

        let (result_specs, scratch_specs) =
            AccelerationStructure::compute_buffer_memory(&context, accel_struct);

        let buffer = Buffer::new(
            context.clone(),
            BufferInfo::default()
                .gpu_only()
                .usage(vk::BufferUsageFlags::RAY_TRACING_NV)
                .memory_type_bits(result_specs.type_bits),
            result_specs.size,
            1,
        );

        let scratch_buffer = Buffer::new(
            context.clone(),
            BufferInfo::default()
                .gpu_only()
                .usage(vk::BufferUsageFlags::RAY_TRACING_NV)
                .memory_type_bits(scratch_specs.type_bits),
            scratch_specs.size,
            1,
        );

        let instance_size = std::mem::size_of::<InstanceDescriptor>() * blas.len();
        let instance_buffer = Buffer::new(
            context.clone(),
            BufferInfo::default()
                .cpu_to_gpu()
                .usage(vk::BufferUsageFlags::RAY_TRACING_NV),
            instance_size as u64,
            blas.len() as u32,
        );

        let mut result = TLAS {
            context: context.clone(),
            accel_struct: AccelerationStructure {
                context,
                accel_struct,
                accel_struct_flags,
                scratch_buffer,
                buffer,
            },
            instance_buffer,
        };
        let previous = vk::AccelerationStructureNV::null();
        result.generate(cmd, blas, previous, false);
        result
    }

    fn create_accel_struct(
        context: &Arc<Context>,
        instance_count: u32,
        flags: vk::BuildAccelerationStructureFlagsNV,
    ) -> (
        vk::AccelerationStructureNV,
        vk::BuildAccelerationStructureFlagsNV,
    ) {
        let info = vk::AccelerationStructureInfoNV::builder()
            .ty(vk::AccelerationStructureTypeNV::TOP_LEVEL)
            .flags(flags)
            .instance_count(instance_count)
            .build();
        let create_info = vk::AccelerationStructureCreateInfoNV::builder()
            .info(info)
            .compacted_size(0)
            .build();
        let accel_struct = unsafe {
            context
                .ray_tracing()
                .create_acceleration_structure(&create_info, None)
                .unwrap()
        };
        (accel_struct, flags)
    }

    pub fn generate(
        &mut self,
        cmd: vk::CommandBuffer,
        blas: &[BLAS],
        previous: vk::AccelerationStructureNV,
        update_only: bool,
    ) {
        //TODO: Compile time asserts?
        assert_eq!(std::mem::size_of::<InstanceDescriptor>(), 64);

        let mut instances = Vec::<InstanceDescriptor>::new();
        for (i, blas) in blas.iter().enumerate() {
            let struct_handle = unsafe {
                self.context
                    .ray_tracing()
                    .get_acceleration_structure_handle(blas.handle())
                    .unwrap()
            };
            let transposed = blas.get_transform().transpose();
            let transform: [f32; 12] = unsafe { std::mem::transmute_copy(&transposed) };
            instances.push(InstanceDescriptor::new(
                transform,
                i as u32,
                0xff,
                blas.hit_group_index,
                vk::GeometryInstanceFlagsNV::TRIANGLE_CULL_DISABLE_NV,
                struct_handle,
            ));
        }
        self.instance_buffer.update(&instances);

        unsafe {
            if !update_only {
                // Bind the acceleration structure descriptor to the actual memory that will store the AS itself
                let bind_info = vk::BindAccelerationStructureMemoryInfoNV::builder()
                    .acceleration_structure(self.accel_struct.handle())
                    .memory(
                        self.accel_struct
                            .buffer
                            .get_alloc_info()
                            .get_device_memory(),
                    )
                    .memory_offset(self.accel_struct.buffer.get_alloc_info().get_offset() as u64)
                    .build();
                self.context
                    .ray_tracing()
                    .bind_acceleration_structure_memory(&[bind_info])
                    .unwrap();
            }

            let info = vk::AccelerationStructureInfoNV::builder()
                .flags(self.accel_struct.accel_struct_flags)
                .ty(vk::AccelerationStructureTypeNV::TOP_LEVEL)
                .instance_count(instances.len() as u32)
                .geometries(&[]);

            self.context.ray_tracing().cmd_build_acceleration_structure(
                cmd,
                &info,
                self.instance_buffer.handle(),
                0, //self.instance_buffer.get_alloc_info().get_offset() as u64,
                update_only,
                self.accel_struct.handle(),
                previous,
                self.accel_struct.scratch_buffer.handle(),
                0,
            );

            let memory_barrier = vk::MemoryBarrier::builder()
                .src_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_NV
                        | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_NV,
                )
                .dst_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_NV
                        | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_NV,
                )
                .build();
            self.context.device().cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
        }
    }
}

impl crate::Resource<vk::AccelerationStructureNV> for TLAS {
    fn handle(&self) -> vk::AccelerationStructureNV {
        self.accel_struct.handle()
    }
}
