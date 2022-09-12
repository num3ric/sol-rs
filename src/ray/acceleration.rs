use crate::{Buffer, BufferInfo, Context, Resource};
use ash::{vk};
use std::sync::Arc;

pub struct GeometryInstance {
    pub vertex_buffer: vk::DeviceAddress,
    pub vertex_count: u32,
    pub vertex_offset: u32,
    pub vertex_offset_size: vk::DeviceSize,
    pub index_buffer: Option<vk::DeviceAddress>,
    pub index_count: Option<u32>,
    pub index_offset_size: Option<vk::DeviceSize>,
    pub transform: glam::Mat4,
}

struct AccelerationStructure {
    context: Arc<Context>,
    accel_struct: vk::AccelerationStructureKHR,
    scratch_buffer: Buffer,
    buffer: Buffer,
}

struct MemorySpec {
    size: vk::DeviceSize,
    type_bits: u32,
}

impl Resource<vk::AccelerationStructureKHR> for AccelerationStructure {
    fn handle(&self) -> vk::AccelerationStructureKHR {
        self.accel_struct
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.context
                .acceleration_structure()
                .destroy_acceleration_structure(self.accel_struct, None);
        }
    }
}

fn create_accel_struct(
    context: &Arc<Context>,
    cmd: vk::CommandBuffer,
    ty: vk::AccelerationStructureTypeKHR,
    mut geometry_info: vk::AccelerationStructureBuildGeometryInfoKHR,
    build_range_infos: &[vk::AccelerationStructureBuildRangeInfoKHR],
    max_primitive_counts: &[u32],
    preallocate_bytes: usize,
) -> (Buffer, Buffer, vk::AccelerationStructureKHR) {

    let mem_reqs = unsafe {
        context.acceleration_structure()
            .get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &geometry_info,
                max_primitive_counts,
            )
    };

    let backing_buffer_size: usize =
        preallocate_bytes.max(mem_reqs.acceleration_structure_size as usize);

    let buffer = Buffer::new(
        context.clone(),
        BufferInfo::default()
            .gpu_only()
            .usage(vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
        backing_buffer_size as vk::DeviceSize,
        1,
    );

    let scratch_buffer = Buffer::new(
        context.clone(),
        BufferInfo::default()
            .gpu_only()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
        mem_reqs.build_scratch_size,
        1,
    );

    let create_info = vk::AccelerationStructureCreateInfoKHR::builder()
        .ty(ty)
        .buffer(buffer.handle())
        .size(buffer.get_size())
        .build();
    
    let accel_structure = unsafe {
        context.acceleration_structure().create_acceleration_structure(&create_info, None).unwrap()
    };

    geometry_info.dst_acceleration_structure = accel_structure;
    geometry_info.scratch_data = vk::DeviceOrHostAddressKHR{ device_address: scratch_buffer.get_device_address() };

    unsafe {
        context.acceleration_structure().cmd_build_acceleration_structures(
            cmd,
            std::slice::from_ref(&geometry_info),
            std::slice::from_ref(&build_range_infos),
        );

        let memory_barrier = vk::MemoryBarrier::builder()
            .src_access_mask(
                vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                    | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
            )
            .dst_access_mask(
                vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                    | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
            )
            .build();
        context.device().cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::DependencyFlags::empty(),
            &[memory_barrier],
            &[],
            &[],
        );
    }

    (buffer, scratch_buffer, accel_structure)
}

pub struct BLAS {
    accel_struct: AccelerationStructure,
    geometries: Vec<vk::AccelerationStructureGeometryKHR>,
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
        let mut geometries = Vec::<vk::AccelerationStructureGeometryKHR>::new();
        let mut max_primitive_counts = Vec::<u32>::new();
        let mut build_range_infos = Vec::<vk::AccelerationStructureBuildRangeInfoKHR>::new();

        for ref geo in geo_intances {
            let flags = match is_opaque {
                true => vk::GeometryFlagsKHR::OPAQUE,
                false => vk::GeometryFlagsKHR::empty(),
            };
            
            let triangles = match geo.index_buffer {
                Some(_) => {
                    vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                        .vertex_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: geo.vertex_buffer,
                        })
                        .vertex_stride(vertex_stride)
                        .max_vertex(geo.vertex_count - 1)
                        .vertex_format(vk::Format::R32G32B32_SFLOAT) //TODO: get from buffer
                        .index_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: geo.index_buffer.unwrap(),
                        })
                        .index_type(vk::IndexType::UINT32) //TODO: get from buffer
                        .build()
                }
                None => {
                    vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                        .vertex_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: geo.vertex_buffer,
                        })
                        .vertex_stride(vertex_stride)
                        .vertex_format(vk::Format::R32G32B32_SFLOAT) //TODO: get from buffer
                        .build()
                }
            };

            let primitive_count;
            let primitive_offset;
            if geo.index_buffer.is_some() {
                primitive_count = geo.index_count.unwrap() as u32 / 3;
                primitive_offset = geo.index_offset_size.unwrap() as u32;
            }
            else {
                primitive_count = geo.vertex_count / 3;
                primitive_offset = geo.vertex_offset_size as u32;
            }

            max_primitive_counts.push(primitive_count);

            build_range_infos.push(
                    vk::AccelerationStructureBuildRangeInfoKHR::builder()
                    .primitive_count(primitive_count)
                    .primitive_offset(primitive_offset)
                    .first_vertex(geo.vertex_offset)
                    .transform_offset(0)
                    .build()
            );

            geometries.push(
                vk::AccelerationStructureGeometryKHR::builder()
                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                    .geometry(vk::AccelerationStructureGeometryDataKHR{triangles})
                    .flags(flags)
                    .build(),
            );
        }

        let geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(geometries.as_slice())
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .build();

        let (buffer, scratch_buffer, accel_struct) = create_accel_struct(
            &context,
            cmd,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            geometry_info,
            &build_range_infos,
            &max_primitive_counts,
            0,
        );

        BLAS {
            accel_struct: AccelerationStructure {
                context,
                accel_struct,
                scratch_buffer,
                buffer,
            },
            transform,
            geometries,
            hit_group_index: 0,
        }
    }

    pub fn get_transform(&self) -> glam::Mat4 {
        self.transform
    }

    pub fn set_transform(&mut self, transform: glam::Mat4) {
        self.transform = transform
    }
}

impl crate::Resource<vk::AccelerationStructureKHR> for BLAS {
    fn handle(&self) -> vk::AccelerationStructureKHR {
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
        flags: vk::GeometryInstanceFlagsKHR,
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
    fn create_instances(context: &Arc<Context>, blas: &[BLAS]) -> Vec<InstanceDescriptor>
    {
        blas
            .iter()
            .enumerate()
            .map(|(i, blas)| {
                let struct_handle = unsafe {
                    context
                        .acceleration_structure()
                        .get_acceleration_structure_device_address(
                            &vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                                    .acceleration_structure(blas.handle())
                                    .build()
                        )
                };
                let transposed = blas.get_transform().transpose();
                let transform: [f32; 12] = unsafe { std::mem::transmute_copy(&transposed) };
                InstanceDescriptor::new(
                    transform,
                    i as u32,
                    0xff,
                    blas.hit_group_index,
                    vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE | vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE,
                    struct_handle,
                )
            })
            .collect()
    }

    pub fn new(context: Arc<Context>, cmd: vk::CommandBuffer, blas: &[BLAS]) -> Self {

        let instances = Self::create_instances(&context, blas);

        let instance_buffer = Buffer::from_data(
            context.clone(),
            BufferInfo::default()
                .cpu_to_gpu()
                .usage(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR),
            instances.as_slice(),
        );

        let geometry = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: instance_buffer.get_device_address(),
                    })
                    .build(),
            })
            .build();

        let build_range_infos = vec![vk::AccelerationStructureBuildRangeInfoKHR::builder()
            .primitive_count(instances.len() as _)
            .build()];
            
        let geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(std::slice::from_ref(&geometry))
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .build();
            
        let max_primitive_counts = [instances.len() as u32];

        let (buffer, scratch_buffer, accel_struct) = create_accel_struct(
            &context,
            cmd,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            geometry_info,
            &build_range_infos,
            &max_primitive_counts,
            0,
        );

        TLAS {
            context: context.clone(),
            accel_struct: AccelerationStructure {
                context,
                accel_struct,
                scratch_buffer,
                buffer,
            },
            instance_buffer,
        }
    }

    pub fn regenerate(
        &mut self,
        cmd: vk::CommandBuffer,
        blas: &[BLAS]
    ) {
        assert_eq!(std::mem::size_of::<InstanceDescriptor>(), 64);

        let instances = Self::create_instances(&self.context, blas);
        self.instance_buffer.update(&instances);

        let geometry = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: self.instance_buffer.get_device_address(),
                    })
                    .build(),
            })
            .build();

        let build_range_infos = vec![vk::AccelerationStructureBuildRangeInfoKHR::builder()
            .primitive_count(instances.len() as _)
            .build()];

        let mut geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(std::slice::from_ref(&geometry))
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .build();
        
        unsafe {
            geometry_info.dst_acceleration_structure = self.handle();
            geometry_info.scratch_data = vk::DeviceOrHostAddressKHR {
                device_address: self.accel_struct.scratch_buffer.get_device_address()
            };

            self.context.acceleration_structure()
                .cmd_build_acceleration_structures(
                    cmd,
                    std::slice::from_ref(&geometry_info),
                    std::slice::from_ref(&&build_range_infos[..]),
                );

            let memory_barrier = vk::MemoryBarrier::builder()
                .src_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                        | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
                )
                .dst_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                        | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
                )
                .build();
            self.context.device().cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
        }
    }
}

impl crate::Resource<vk::AccelerationStructureKHR> for TLAS {
    fn handle(&self) -> vk::AccelerationStructureKHR {
        self.accel_struct.handle()
    }
}
