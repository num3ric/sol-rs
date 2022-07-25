use crate::{Context, RenderPass, Resource, TransientRenderPassInfo, Vertex};
use ash::vk;
use shaderc::{CompileOptions, Compiler, IncludeType, ResolvedInclude, ShaderKind};
use std::ffi::CString;
use std::fs;
use std::path::{Path, PathBuf};
use std::result::Result;
use std::string::String;
use std::sync::Arc;

const STORE_SPIRV: bool = false;
const LOAD_SPIRV: bool = false;

pub struct Shader {
    context: Arc<Context>,
    pub module: vk::ShaderModule,
    pub stage_flags: vk::ShaderStageFlags,
    pub path: PathBuf,
    text: Option<String>,
}

fn get_sharerc_include(
    requested_source: &str,
    _include_type: IncludeType,
    _origin_source: &str,
    _recursion_depth: usize,
    origin_dir: &Path,
) -> Result<ResolvedInclude, String> {
    //TODO: finish implementation
    let resolved_file = origin_dir.join(requested_source);
    let resolved_name = resolved_file
        // .file_name()
        // .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    //println!("Including: {}", resolved_name);
    let error_msg = format!("Failed to open {}.", resolved_file.to_str().unwrap());
    let content = fs::read_to_string(resolved_file.as_path()).expect(&error_msg);
    Ok(ResolvedInclude {
        resolved_name,
        content,
    })
}

fn get_shaderc_stage(stage: &vk::ShaderStageFlags) -> Option<ShaderKind> {
    if *stage == vk::ShaderStageFlags::VERTEX {
        return Some(ShaderKind::Vertex);
    } else if *stage == vk::ShaderStageFlags::FRAGMENT {
        return Some(ShaderKind::Fragment);
    } else if *stage == vk::ShaderStageFlags::COMPUTE {
        return Some(ShaderKind::Compute);
    } else if *stage == vk::ShaderStageFlags::TESSELLATION_CONTROL {
        return Some(ShaderKind::TessControl);
    } else if *stage == vk::ShaderStageFlags::TESSELLATION_EVALUATION {
        return Some(ShaderKind::TessEvaluation);
    } else if *stage == vk::ShaderStageFlags::GEOMETRY {
        return Some(ShaderKind::Geometry);
    } else if *stage == vk::ShaderStageFlags::RAYGEN_KHR {
        return Some(ShaderKind::RayGeneration);
    } else if *stage == vk::ShaderStageFlags::ANY_HIT_KHR {
        return Some(ShaderKind::AnyHit);
    } else if *stage == vk::ShaderStageFlags::CLOSEST_HIT_KHR {
        return Some(ShaderKind::ClosestHit);
    } else if *stage == vk::ShaderStageFlags::MISS_KHR {
        return Some(ShaderKind::Miss);
    } else if *stage == vk::ShaderStageFlags::INTERSECTION_KHR {
        return Some(ShaderKind::Intersection);
    }
    None
}

fn get_spirv_filepath(path: &PathBuf) -> PathBuf {
    let mut compiled_path = path.clone();
    let filename = compiled_path
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string()
        + ".spv";
    compiled_path.set_file_name(filename);
    compiled_path
}

fn is_more_recent(path: &PathBuf, other: &PathBuf) -> bool {
    let timestamp = fs::metadata(path.as_path()).unwrap().modified().unwrap();
    let other_timestamp = fs::metadata(other.as_path()).unwrap().modified().unwrap();
    timestamp > other_timestamp
}

impl Shader {
    pub fn new(context: Arc<Context>, path: PathBuf, stage_flags: vk::ShaderStageFlags) -> Self {
        let spirv_path = get_spirv_filepath(&path);
        // Only load spirv directly if its timestamp is more recent than the source file.
        if spirv_path.exists() && LOAD_SPIRV && is_more_recent(&spirv_path, &path) {
            let mut file = std::fs::File::open(&spirv_path).unwrap();
            let words = ash::util::read_spv(&mut file).unwrap();
            let shader_info = vk::ShaderModuleCreateInfo::builder().code(&words);
            unsafe {
                let module = context
                    .device()
                    .create_shader_module(&shader_info, None)
                    .unwrap();
                return Shader {
                    context,
                    module,
                    stage_flags,
                    path,
                    text: None,
                };
            }
        }

        let error_msg = format!("Failed to open {}.", path.to_str().unwrap());
        let source = fs::read_to_string(path.as_path()).expect(&error_msg);

        let mut compiler = Compiler::new().unwrap();
        let mut options = CompileOptions::new().unwrap();
        options.set_generate_debug_info();
        options.set_target_spirv(shaderc::SpirvVersion::V1_4);
        options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_2 as u32);
        let origin_path = path.clone();
        options.set_include_callback(
            move |requested_source, include_type, origin_source, recursion_depth| {
                get_sharerc_include(
                    requested_source,
                    include_type,
                    origin_source,
                    recursion_depth,
                    origin_path.parent().unwrap(),
                )
            },
        );
        let sc_stage = get_shaderc_stage(&stage_flags).unwrap();
        let code = compiler
            .compile_into_spirv(
                &source,
                sc_stage,
                path.file_name().unwrap().to_str().unwrap(),
                "main",
                Some(&options),
            )
            .unwrap();

        if STORE_SPIRV {
            std::fs::write(spirv_path, code.as_binary_u8()).expect("Failed to write spir-v.");
        }
        let shader_info = vk::ShaderModuleCreateInfo::builder().code(code.as_binary());
        unsafe {
            let module = context
                .device()
                .create_shader_module(&shader_info, None)
                .unwrap();
            Shader {
                context,
                module,
                stage_flags,
                path,
                text: Some(source),
            }
        }
    }

    pub fn get_create_info(&self, name: &'_ std::ffi::CStr) -> vk::PipelineShaderStageCreateInfo {
        vk::PipelineShaderStageCreateInfo::builder()
            .module(self.module)
            .stage(self.stage_flags)
            .name(name)
            .build()
    }
    pub fn get_create_info_with_specialization(
        &self,
        name: &'_ std::ffi::CStr,
        specialization_info: &vk::SpecializationInfo,
    ) -> vk::PipelineShaderStageCreateInfo {
        vk::PipelineShaderStageCreateInfo::builder()
            .module(self.module)
            .stage(self.stage_flags)
            .specialization_info(specialization_info)
            .name(name)
            .build()
    }
}

impl crate::Resource<vk::ShaderModule> for Shader {
    fn handle(&self) -> vk::ShaderModule {
        self.module
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device()
                .destroy_shader_module(self.module, None);
        }
    }
}

pub enum PipelineBlendMode {
    Opaque,
    Alpha,
}

impl std::default::Default for PipelineBlendMode {
    fn default() -> Self {
        PipelineBlendMode::Opaque
    }
}
pub struct PipelineInfo {
    pub layout: vk::PipelineLayout,
    pub render_pass: Option<vk::RenderPass>,
    pub transient_render_pass_info: Option<TransientRenderPassInfo>,
    pub shaders: Vec<(PathBuf, vk::ShaderStageFlags)>,
    pub name: String,
    pub depth_test_enabled: bool,
    pub depth_write_enabled: bool,
    pub blend_mode: PipelineBlendMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub vertex_stride: u32,
    pub vertex_format_offset: Vec<(vk::Format, u32)>,
    pub samples: vk::SampleCountFlags,
    pub specialization_data: Vec<u8>,
    pub specialization_entries: Vec<vk::SpecializationMapEntry>,
}

impl std::default::Default for PipelineInfo {
    fn default() -> Self {
        PipelineInfo {
            layout: vk::PipelineLayout::default(),
            render_pass: None,
            transient_render_pass_info: None,
            shaders: Vec::new(),
            name: "".to_string(),
            depth_test_enabled: true,
            depth_write_enabled: true,
            blend_mode: PipelineBlendMode::default(),
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            vertex_stride: 0,
            vertex_format_offset: Vec::new(),
            samples: vk::SampleCountFlags::TYPE_1,
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
    pub fn render_pass(mut self, render_pass: vk::RenderPass) -> Self {
        self.render_pass = Some(render_pass);
        self
    }
    pub fn render_pass_info(mut self, info: TransientRenderPassInfo) -> Self {
        self.samples = info.samples;
        self.transient_render_pass_info = Some(info);
        self
    }
    pub fn samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.samples = samples;
        self
    }
    pub fn shader(mut self, path: PathBuf, stage_flags: vk::ShaderStageFlags) -> Self {
        self.shaders.push((path, stage_flags));
        self
    }
    pub fn vert(mut self, path: PathBuf) -> Self {
        self.shaders.push((path, vk::ShaderStageFlags::VERTEX));
        self
    }
    pub fn frag(mut self, path: PathBuf) -> Self {
        self.shaders.push((path, vk::ShaderStageFlags::FRAGMENT));
        self
    }
    pub fn cull_mode(mut self, cull_mode: vk::CullModeFlags) -> Self {
        self.cull_mode = cull_mode;
        self
    }
    pub fn front_face(mut self, front_face: vk::FrontFace) -> Self {
        self.front_face = front_face;
        self
    }
    pub fn name(mut self, name: String) -> Self {
        self.name = name.to_string();
        self
    }
    pub fn vertex_type<T>(mut self) -> Self
    where
        T: Vertex,
    {
        self.vertex_stride = T::stride();
        self.vertex_format_offset = T::format_offset();
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
    transient_render_pass: Option<RenderPass>,
}

impl Pipeline {
    pub fn new(context: Arc<Context>, info: PipelineInfo) -> Self {
        assert!(info.vertex_stride > 0);
        assert!(!info.vertex_format_offset.is_empty());
        assert!(info.render_pass.is_some() || info.transient_render_pass_info.is_some());

        let mut shaders = Vec::<Shader>::new();
        let mut shader_stage_create_infos = Vec::new();
        let shader_entry_name = CString::new("main").unwrap();
        for shader_info in &info.shaders {
            let shader = Shader::new(context.clone(), shader_info.0.clone(), shader_info.1);
            if info.specialization_entries.is_empty() {
                shader_stage_create_infos.push(shader.get_create_info(&shader_entry_name));
            } else {
                shader_stage_create_infos.push(
                    shader.get_create_info_with_specialization(
                        &shader_entry_name,
                        &vk::SpecializationInfo::builder()
                            .map_entries(&info.specialization_entries)
                            .data(&info.specialization_data),
                    ),
                );
            }
            shaders.push(shader);
        }
        let vertex_input_binding_descriptions = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: info.vertex_stride,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let mut vertex_input_attribute_descriptions = Vec::new();
        for (i, format_pair) in info.vertex_format_offset.iter().enumerate() {
            vertex_input_attribute_descriptions.push(vk::VertexInputAttributeDescription {
                location: i as u32,
                binding: 0,
                format: format_pair.0,
                offset: format_pair.1,
            });
        }
        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo {
            vertex_attribute_description_count: vertex_input_attribute_descriptions.len() as u32,
            p_vertex_attribute_descriptions: vertex_input_attribute_descriptions.as_ptr(),
            vertex_binding_description_count: vertex_input_binding_descriptions.len() as u32,
            p_vertex_binding_descriptions: vertex_input_binding_descriptions.as_ptr(),
            ..Default::default()
        };
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let viewport_state_info = vk::PipelineViewportStateCreateInfo {
            scissor_count: 1,
            viewport_count: 1,
            ..Default::default()
        };

        let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
            front_face: info.front_face,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: info.cull_mode,
            ..Default::default()
        };
        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: info.samples,
            ..Default::default()
        };
        let noop_stencil_state = vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            ..Default::default()
        };
        let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: info.depth_test_enabled as u32,
            depth_write_enable: info.depth_write_enabled as u32,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            front: noop_stencil_state,
            back: noop_stencil_state,
            max_depth_bounds: 1.0,
            ..Default::default()
        };

        //TODO: Implement blending modes
        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ZERO,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        }];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op(vk::LogicOp::CLEAR)
            .attachments(&color_blend_attachment_states);

        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_state);

        let transient_render_pass = match info.transient_render_pass_info.clone() {
            Some(render_pass_info) => Some(RenderPass::new_transient(
                context.shared().clone(),
                render_pass_info,
            )),
            None => None,
        };
        let render_pass = match info.render_pass {
            Some(render_pass) => render_pass,
            None => transient_render_pass.as_ref().unwrap().handle(),
        };
        let create_infos = [vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stage_create_infos)
            .vertex_input_state(&vertex_input_state_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_state_info)
            .depth_stencil_state(&depth_state_info)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state_info)
            .layout(info.layout)
            .render_pass(render_pass)
            .build()];

        let graphics_pipelines = unsafe {
            context
                .device()
                .create_graphics_pipelines(vk::PipelineCache::null(), &create_infos, None)
                .expect("Unable to create graphics pipeline")
        };

        Pipeline {
            context,
            info,
            pipeline: graphics_pipelines[0],
            transient_render_pass,
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
