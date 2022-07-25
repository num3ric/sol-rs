use crate::{pipeline::Shader, Context, Resource};
use ash::{vk};
use std::{ffi::CString, path::PathBuf, sync::Arc};

pub struct PipelineInfo {
    pub layout: vk::PipelineLayout,
    pub shaders: Vec<(PathBuf, vk::ShaderStageFlags)>,
    pub name: String,
    pub specialization_data: Vec<u8>,
    pub specialization_entries: Vec<vk::SpecializationMapEntry>,
}

impl std::default::Default for PipelineInfo {
    fn default() -> Self {
        PipelineInfo {
            layout: vk::PipelineLayout::default(),
            shaders: Vec::new(),
            name: "".to_string(),
            specialization_data: Vec::new(),
            specialization_entries: Vec::new(),
        }
    }
}

impl PipelineInfo {
    pub fn layout(mut self, layout: vk::PipelineLayout) -> Self {
        self.layout = layout;
        self
    }
    pub fn shader(mut self, path: PathBuf, stage_flags: vk::ShaderStageFlags) -> Self {
        self.shaders.push((path, stage_flags));
        self
    }
    pub fn name(mut self, name: String) -> Self {
        self.name = name.to_string();
        self
    }
    pub fn specialization<T>(mut self, data: &T, constant_id: u32) -> Self {
        let slice = unsafe {
            std::slice::from_raw_parts(data as *const T as *const u8, std::mem::size_of_val(data))
        };
        self.specialization_data = slice.to_vec();
        self.specialization_entries.push(
            vk::SpecializationMapEntry::builder()
                .constant_id(constant_id)
                .offset(0)
                .size(self.specialization_data.len())
                .build(),
        );
        self
    }
}

pub struct Pipeline {
    context: Arc<Context>,
    info: PipelineInfo,
    pipeline: vk::Pipeline,
}

impl Pipeline {
    pub fn new(context: Arc<Context>, info: PipelineInfo) -> Self {
        let mut shaders = Vec::<Shader>::new();
        let mut stages = Vec::new();
        let mut groups = Vec::new();
        let shader_entry_name = CString::new("main").unwrap();
        for (index, shader_info) in info.shaders.iter().enumerate() {
            let shader = Shader::new(context.clone(), shader_info.0.clone(), shader_info.1);
            if info.specialization_entries.is_empty() {
                stages.push(shader.get_create_info(&shader_entry_name));
            } else {
                stages.push(
                    shader.get_create_info_with_specialization(
                        &shader_entry_name,
                        &vk::SpecializationInfo::builder()
                            .map_entries(&info.specialization_entries)
                            .data(&info.specialization_data),
                    ),
                );
            }
            shaders.push(shader);

            let mut group = vk::RayTracingShaderGroupCreateInfoKHR::builder()
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
                .build();
            if shader_info.1 == vk::ShaderStageFlags::CLOSEST_HIT_KHR {
                group.ty = vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP;
                group.closest_hit_shader = index as u32;
            } else {
                group.ty = vk::RayTracingShaderGroupTypeKHR::GENERAL;
                group.general_shader = index as u32;
            }
            groups.push(group);
        }
        // TODO: fetch from somewhere
        let max_recursion_depth = 8;
        let create_info = vk::RayTracingPipelineCreateInfoKHR::builder()
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(max_recursion_depth)
            .layout(info.layout)
            .build();
        let pipeline = unsafe {
            context
                .ray_tracing()
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    &[create_info],
                    None
                )
                .expect("Unable to create graphics pipeline")[0]
        };

        Pipeline {
            context,
            info,
            pipeline,
        }
    }

    pub fn update_specialization<T>(&mut self, data: &T) {
        let slice = unsafe {
            std::slice::from_raw_parts(data as *const T as *const u8, std::mem::size_of_val(data))
        };
        self.info.specialization_data = slice.to_vec();
    }
}

impl Resource<vk::Pipeline> for Pipeline {
    fn handle(&self) -> vk::Pipeline {
        self.pipeline
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.context.device().destroy_pipeline(self.pipeline, None);
        }
    }
}
