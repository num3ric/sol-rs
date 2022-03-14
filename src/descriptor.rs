use crate::Context;
use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;

pub struct DescriptorSetInfo {
    pub buffer_infos: HashMap<u32, Vec<vk::DescriptorBufferInfo>>,
    pub image_infos: HashMap<u32, Vec<vk::DescriptorImageInfo>>,
    pub acceleration_structures: HashMap<u32, Vec<vk::AccelerationStructureNV>>,
}

impl std::default::Default for DescriptorSetInfo {
    fn default() -> Self {
        DescriptorSetInfo {
            buffer_infos: HashMap::new(),
            image_infos: HashMap::new(),
            acceleration_structures: HashMap::new(),
        }
    }
}
impl Eq for DescriptorSetInfo {}

fn do_vecs_match<T: PartialEq>(a: &Vec<T>, b: &Vec<T>) -> bool {
    let matching = a.iter().zip(b).filter(|&(a, b)| a == b).count();
    matching == a.len() && matching == b.len()
}

impl PartialEq for DescriptorSetInfo {
    //TODO: Do not use raw vulkan handles, neither here nor for hashing: custom uuid.
    fn eq(&self, other: &Self) -> bool {
        for (key, infos) in &self.buffer_infos {
            if !other.buffer_infos.contains_key(key) {
                return false;
            }
            if infos.len() != other.buffer_infos[key].len() {
                return false;
            }
            for (i, info) in infos.iter().enumerate() {
                let ref other_info = other.buffer_infos[key][i];
                if info.buffer != other_info.buffer {
                    return false;
                }
                if info.offset != other_info.offset {
                    return false;
                }
                if info.range != other_info.range {
                    return false;
                }
            }
        }
        for (key, infos) in &self.image_infos {
            if !other.image_infos.contains_key(key) {
                return false;
            }
            if infos.len() != other.image_infos[key].len() {
                return false;
            }
            for (i, info) in infos.iter().enumerate() {
                let ref other_info = other.image_infos[key][i];
                if info.sampler != other_info.sampler {
                    return false;
                }
                if info.image_view != other_info.image_view {
                    return false;
                }
                // if info.image_layout != other_info.image_layout {
                //     return false;
                // }
            }
        }
        for (key, structs) in &self.acceleration_structures {
            if !other.acceleration_structures.contains_key(key) {
                return false;
            }
            if structs.len() != other.acceleration_structures[key].len() {
                return false;
            }
            for (i, accel_struct) in structs.iter().enumerate() {
                if other.acceleration_structures[key][i] != *accel_struct {
                    return false;
                }
            }
        }
        true
    }
}

impl DescriptorSetInfo {
    pub fn buffer(mut self, binding: u32, info: vk::DescriptorBufferInfo) -> Self {
        self.buffer_infos.insert(binding, vec![info]);
        self
    }

    pub fn buffers(mut self, binding: u32, infos: Vec<vk::DescriptorBufferInfo>) -> Self {
        self.buffer_infos.insert(binding, infos);
        self
    }

    pub fn image(mut self, binding: u32, info: vk::DescriptorImageInfo) -> Self {
        self.image_infos.insert(binding, vec![info]);
        self
    }

    pub fn images(mut self, binding: u32, infos: Vec<vk::DescriptorImageInfo>) -> Self {
        self.image_infos.insert(binding, infos);
        self
    }

    pub fn accel_struct(mut self, binding: u32, accel_struct: vk::AccelerationStructureNV) -> Self {
        self.acceleration_structures
            .insert(binding, vec![accel_struct]);
        self
    }
    pub fn accel_structs(
        mut self,
        binding: u32,
        accel_structs: Vec<vk::AccelerationStructureNV>,
    ) -> Self {
        self.acceleration_structures.insert(binding, accel_structs);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.image_infos.is_empty()
            && self.buffer_infos.is_empty()
            && self.acceleration_structures.is_empty()
    }
}

impl std::hash::Hash for DescriptorSetInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for (key, infos) in &self.buffer_infos {
            key.hash(state);
            for info in infos {
                info.buffer.hash(state);
                info.offset.hash(state);
                info.range.hash(state);
            }
        }
        for (key, infos) in &self.image_infos {
            key.hash(state);
            for info in infos {
                info.sampler.hash(state);
                info.image_view.hash(state);
                //info.image_layout.hash(state);
            }
        }
        for (key, structs) in &self.acceleration_structures {
            key.hash(state);
            for accel_struct in structs {
                accel_struct.hash(state);
            }
        }
    }
}
#[derive(Clone, Debug, Copy)]
pub struct DescriptorSet {
    handle: vk::DescriptorSet,
}

impl crate::Resource<vk::DescriptorSet> for DescriptorSet {
    fn handle(&self) -> vk::DescriptorSet {
        self.handle
    }
}

pub struct DescriptorSetLayoutInfo {
    pub bindings: HashMap<u32, (vk::DescriptorType, vk::ShaderStageFlags, u32)>,
    pub flags: vk::DescriptorSetLayoutCreateFlags,
    pub min_max_sets: u32,
}

impl std::default::Default for DescriptorSetLayoutInfo {
    fn default() -> Self {
        DescriptorSetLayoutInfo {
            bindings: HashMap::new(),
            flags: vk::DescriptorSetLayoutCreateFlags::default(),
            min_max_sets: 64,
        }
    }
}

impl DescriptorSetLayoutInfo {
    pub fn binding(
        mut self,
        binding: u32,
        descritor_type: vk::DescriptorType,
        stage: vk::ShaderStageFlags,
    ) -> Self {
        self.bindings.insert(binding, (descritor_type, stage, 1));
        self
    }
    pub fn bindings(
        mut self,
        binding: u32,
        descritor_type: vk::DescriptorType,
        stage: vk::ShaderStageFlags,
        count: u32,
    ) -> Self {
        self.bindings
            .insert(binding, (descritor_type, stage, count));
        self
    }

    pub fn min_max_sets(mut self, min_max_sets: u32) -> Self {
        self.min_max_sets = min_max_sets;
        self
    }
}

pub struct DescriptorSetLayout {
    context: Arc<Context>,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    info: DescriptorSetLayoutInfo,
    sets: HashMap<DescriptorSetInfo, DescriptorSet>,
}

impl DescriptorSetLayout {
    pub fn new(context: Arc<Context>, info: DescriptorSetLayoutInfo) -> Self {
        let n = info.bindings.len() as usize;
        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = Vec::with_capacity(n);
        let mut pool_sizes: Vec<vk::DescriptorPoolSize> = Vec::with_capacity(n);
        let max_sets = info.min_max_sets; //TODO: max with swapchain image count
        for src_binding in &info.bindings {
            bindings.push(
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(*src_binding.0)
                    .descriptor_type((src_binding.1).0)
                    .stage_flags((src_binding.1).1)
                    .descriptor_count((src_binding.1).2)
                    .build(),
            );
            pool_sizes.push(
                vk::DescriptorPoolSize::builder()
                    .ty((src_binding.1).0)
                    .descriptor_count(max_sets * (src_binding.1).2)
                    .build(),
            );
        }

        let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(info.flags)
            .bindings(&bindings);
        unsafe {
            let layout = context
                .device()
                .create_descriptor_set_layout(&create_info, None)
                .expect("Failed to create DescriptorSetLayout");

            let pool_create_info = vk::DescriptorPoolCreateInfo::builder()
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                .max_sets(max_sets)
                .pool_sizes(&pool_sizes);
            let pool = context
                .device()
                .create_descriptor_pool(&pool_create_info, None)
                .expect("Failed to create DescriptorPool");

            DescriptorSetLayout {
                context,
                layout,
                pool,
                info: info,
                sets: HashMap::<DescriptorSetInfo, DescriptorSet>::new(),
            }
        }
    }

    pub fn get_or_create(&mut self, info: DescriptorSetInfo) -> DescriptorSet {
        assert!(!info.is_empty());

        if self.sets.contains_key(&info) {
            return self.sets[&info];
        }

        unsafe {
            let result = DescriptorSet {
                handle: self
                    .context
                    .device()
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::builder()
                            .descriptor_pool(self.pool)
                            .set_layouts(&[self.layout])
                            .build(),
                    )
                    .expect("Failed to create descriptor sets.")[0],
            };
            self.update_sets(result.handle, &info);
            self.sets.insert(info, result.clone());
            result
        }
    }

    pub fn get_descriptor_type(&self, binding: u32) -> vk::DescriptorType {
        self.info.bindings[&binding].0
    }

    pub fn get_shader_stage(&self, binding: u32) -> vk::ShaderStageFlags {
        self.info.bindings[&binding].1
    }

    pub fn get_descriptor_count(&self, binding: u32) -> u32 {
        self.info.bindings[&binding].2
    }

    fn update_sets(&self, set: vk::DescriptorSet, info: &DescriptorSetInfo) {
        let capacity =
            info.buffer_infos.len() + info.image_infos.len() + info.acceleration_structures.len();
        let mut write_descriptor_sets = Vec::<vk::WriteDescriptorSet>::with_capacity(capacity);
        for (binding, info) in &info.buffer_infos {
            write_descriptor_sets.push(
                vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .dst_array_element(0)
                    .descriptor_type(self.get_descriptor_type(*binding))
                    .buffer_info(info)
                    .build(),
            );
        }

        for (binding, info) in &info.image_infos {
            write_descriptor_sets.push(
                vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .dst_array_element(0)
                    .descriptor_type(self.get_descriptor_type(*binding))
                    .image_info(info)
                    .build(),
            );
        }

        for (binding, accel_structs) in &info.acceleration_structures {
            let mut accel_info = vk::WriteDescriptorSetAccelerationStructureNV::builder()
                .acceleration_structures(&accel_structs)
                .build();
            let mut accel_write = vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(*binding)
                .dst_array_element(0)
                .descriptor_type(self.get_descriptor_type(*binding))
                .push_next(&mut accel_info)
                .build();
            // This is only set by the builder for images, buffers, or views; need to set explicitly after
            accel_write.descriptor_count = 1;
            write_descriptor_sets.push(accel_write);
        }

        unsafe {
            self.context
                .device()
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }
    }

    pub fn reset_pool(&self) {
        unsafe {
            let flags = vk::DescriptorPoolResetFlags::default();
            self.context
                .device()
                .reset_descriptor_pool(self.pool, flags)
                .expect("Failed to reset descriptor pool.");
        }
    }
}

impl crate::Resource<vk::DescriptorSetLayout> for DescriptorSetLayout {
    fn handle(&self) -> vk::DescriptorSetLayout {
        self.layout
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device()
                .destroy_descriptor_set_layout(self.layout, None);
            self.context
                .device()
                .destroy_descriptor_pool(self.pool, None);
        }
    }
}

pub struct PipelineLayoutInfo {
    pub flags: vk::PipelineLayoutCreateFlags,
    pub desc_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub push_constant_ranges: Vec<vk::PushConstantRange>,
}

impl std::default::Default for PipelineLayoutInfo {
    fn default() -> Self {
        PipelineLayoutInfo {
            flags: vk::PipelineLayoutCreateFlags::default(),
            desc_set_layouts: Vec::new(),
            push_constant_ranges: Vec::new(),
        }
    }
}

impl PipelineLayoutInfo {
    pub fn desc_set_layout(mut self, set_layout: vk::DescriptorSetLayout) -> Self {
        self.desc_set_layouts = vec![set_layout];
        self
    }
    pub fn desc_set_layouts(mut self, set_layouts: &[vk::DescriptorSetLayout]) -> Self {
        self.desc_set_layouts.extend_from_slice(set_layouts);
        self
    }
    pub fn push_constant_range(mut self, push_constant_range: vk::PushConstantRange) -> Self {
        self.push_constant_ranges = vec![push_constant_range];
        self
    }
    pub fn push_constant_ranges(mut self, push_constant_ranges: &[vk::PushConstantRange]) -> Self {
        self.push_constant_ranges
            .extend_from_slice(push_constant_ranges);
        self
    }
}

pub struct PipelineLayout {
    context: Arc<Context>,
    layout: vk::PipelineLayout,
    info: PipelineLayoutInfo,
}

impl PipelineLayout {
    pub fn new(context: Arc<Context>, info: PipelineLayoutInfo) -> Self {
        let create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&info.desc_set_layouts)
            .push_constant_ranges(&info.push_constant_ranges)
            .build();
        unsafe {
            let layout = context
                .device()
                .create_pipeline_layout(&create_info, None)
                .expect("Failed to create pipeline layout.");
            PipelineLayout {
                context,
                layout,
                info: info,
            }
        }
    }
}

impl crate::Resource<vk::PipelineLayout> for PipelineLayout {
    fn handle(&self) -> vk::PipelineLayout {
        self.layout
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device()
                .destroy_pipeline_layout(self.layout, None);
        }
    }
}
