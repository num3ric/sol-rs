//#![windows_subsystem = "windows"]
use sol::prelude::*;
use sol::ray;
use sol::scene;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Default, Copy, Clone)]
struct SceneUniforms {
    model: Mat4,
    view: Mat4,
    view_inverse: Mat4,
    projection: Mat4,
    projection_inverse: Mat4,
    model_view_projection: Mat4,
    frame: UVec3,
}

impl SceneUniforms {
    pub fn from(camera: &scene::Camera, frame: UVec3) -> SceneUniforms {
        let vp = camera.perspective_matrix() * camera.view_matrix();
        SceneUniforms {
            model: Mat4::IDENTITY,
            view: camera.view_matrix(),
            view_inverse: camera.view_matrix().inverse(),
            projection: camera.perspective_matrix(),
            projection_inverse: camera.perspective_matrix().inverse(),
            model_view_projection: vp,
            frame,
        }
    }
}

pub struct PerFrameData {
    pub ubo: sol::Buffer,
    pub desc_set: sol::DescriptorSet,
}
pub struct AppData {
    pub scene: scene::Scene,
    pub pipeline_layout: sol::PipelineLayout,
    pub layout_scene: sol::DescriptorSetLayout,
    pub layout_pass: sol::DescriptorSetLayout,
    pub per_frame: Vec<PerFrameData>,
    pub manip: scene::CameraManip,

    // Raytracing tools & data
    pub scene_description: ray::SceneDescription,
    pub pipeline: ray::Pipeline,
    pub sbt: ray::ShaderBindingTable,
    pub accumulation_start_frame: u32,
    pub accum_target: sol::Image2d,
    pub render_target: sol::Image2d,

    pub enable_sky: bool, //Temporary shader hack for sky/sun light
}

fn create_image_target(
    context: &Arc<sol::Context>,
    window: &sol::Window,
    format: vk::Format,
) -> sol::Image2d {
    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(window.get_extent_3d())
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    sol::Image2d::new(
        context.shared().clone(),
        &image_info,
        vk::ImageAspectFlags::COLOR,
        1,
        "TargetRT"
    )
}

fn build_pipeline_sbt(
    context: &Arc<sol::Context>,
    pipeline_layout: &sol::PipelineLayout,
    enable_sky: bool,
) -> (ray::Pipeline, ray::ShaderBindingTable) {
    let pipeline = ray::Pipeline::new(
        context.clone(),
        ray::PipelineInfo::default()
            .layout(pipeline_layout.handle())
            .shader(
                sol::util::find_asset("glsl/pathtrace.rgen").unwrap(),
                vk::ShaderStageFlags::RAYGEN_KHR,
            )
            .shader(
                sol::util::find_asset("glsl/pathtrace.rmiss").unwrap(),
                vk::ShaderStageFlags::MISS_KHR,
            )
            .shader(
                sol::util::find_asset("glsl/pathtrace.rchit").unwrap(),
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )
            .specialization(&[enable_sky as u32], 0)
            .name("AO_mat".to_string()),
    );
    let sbt = ray::ShaderBindingTable::new(
        context.clone(),
        pipeline.handle(),
        ray::ShaderBindingTableInfo::default()
            .raygen(0)
            .miss(1)
            .hitgroup(2),
    );

    (pipeline, sbt)
}

pub fn setup(app: &mut sol::App) -> AppData {
    let context = &app.renderer.context;
    let index = std::env::args().position(|arg| arg == "--model").unwrap();
    let scene = scene::load_scene(
        context.clone(),
        &sol::util::find_asset(&std::env::args().nth(index + 1).expect("no gltf file given"))
            .unwrap(),
    );
    let scene_description = ray::SceneDescription::from_scene(context.clone(), &scene);

    let camera = match scene.camera {
        Some(scene_camera) => {
            let mut cam = scene_camera;
            cam.set_window_size(app.window.get_size());
            cam
        }
        None => scene::Camera::new(app.window.get_size()),
    };

    let mut per_frame = Vec::<PerFrameData>::new();

    let mut layout_scene = sol::DescriptorSetLayout::new(
        context.clone(),
        sol::DescriptorSetLayoutInfo::default().binding(
            0,
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::ShaderStageFlags::ALL,
        ),
    );
    let instance_count = scene_description.get_instances_buffer().get_element_count();
    let layout_pass = sol::DescriptorSetLayout::new(
        context.clone(),
        sol::DescriptorSetLayoutInfo::default()
            .binding(
                0,
                vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                vk::ShaderStageFlags::RAYGEN_KHR,
            )
            .binding(
                1,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::ShaderStageFlags::RAYGEN_KHR,
            )
            .binding(
                2,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::ShaderStageFlags::RAYGEN_KHR,
            )
            .binding(
                3,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )
            .bindings(
                4,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                instance_count,
            )
            .bindings(
                5,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                instance_count,
            )
            .bindings(
                6,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                instance_count,
            ),
    );

    for _ in 0..app.renderer.get_frames_count() {
        let uniforms = SceneUniforms::from(
            &camera,
            uvec3(app.window.get_width(), app.window.get_height(), 0),
        );
        let ubo = sol::Buffer::from_data(
            context.clone(),
            sol::BufferInfo::default()
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .cpu_to_gpu(),
            &[uniforms],
        );
        let desc_set = layout_scene
            .get_or_create(sol::DescriptorSetInfo::default().buffer(0, ubo.get_descriptor_info()));
        per_frame.push(PerFrameData { ubo, desc_set });
    }

    let pipeline_layout = sol::PipelineLayout::new(
        context.clone(),
        sol::PipelineLayoutInfo::default()
            .desc_set_layouts(&[layout_scene.handle(), layout_pass.handle()])
            .push_constant_range(
                vk::PushConstantRange::builder()
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                    .size(size_of::<u32>() as u32)
                    .build(),
            ),
    );

    let enable_sky = std::env::args().any(|arg| arg == "--sky");
    let (pipeline, sbt) = build_pipeline_sbt(&context, &pipeline_layout, enable_sky);
    let mut accum_target =
        create_image_target(&context, &app.window, vk::Format::R32G32B32A32_SFLOAT);

    let cmd = context.begin_single_time_cmd();
    accum_target.transition_image_layout(cmd, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
    context.end_single_time_cmd(cmd);

    let render_target = create_image_target(&context, &app.window, vk::Format::R8G8B8A8_UNORM);
    AppData {
        scene,
        pipeline_layout,
        layout_scene,
        layout_pass,
        per_frame,
        manip: scene::CameraManip {
            camera,
            input: scene::CameraInput::default(),
        },
        scene_description,
        pipeline,
        sbt,
        accumulation_start_frame: 0,
        accum_target,
        render_target,
        enable_sky,
    }
}

pub fn window_event(app: &mut sol::App, data: &mut AppData, event: &WindowEvent) {
    if data.manip.update(&event) {
        data.accumulation_start_frame = app.elapsed_ticks as u32;
    }
    match event {
        WindowEvent::Resized(_) => {
            data.accum_target = create_image_target(
                &app.renderer.context,
                &app.window,
                vk::Format::R32G32B32A32_SFLOAT,
            );
            data.render_target = create_image_target(
                &app.renderer.context,
                &app.window,
                vk::Format::R8G8B8A8_UNORM,
            );
            data.accumulation_start_frame = app.elapsed_ticks as u32;
            data.layout_pass.reset_pool();
        }
        WindowEvent::KeyboardInput { input, .. } => {
            if input.state == winit::event::ElementState::Pressed {
                if input.virtual_keycode == Some(winit::event::VirtualKeyCode::R) {
                    unsafe {
                        app.renderer
                            .context
                            .device()
                            .queue_wait_idle(app.renderer.context.graphics_queue())
                            .unwrap();
                    }
                    let (pipeline, sbt) = build_pipeline_sbt(
                        &app.renderer.context,
                        &data.pipeline_layout,
                        data.enable_sky,
                    );
                    data.pipeline = pipeline;
                    data.sbt = sbt;
                    data.accumulation_start_frame = app.elapsed_ticks as u32;
                }
            }
        }
        _ => {}
    }
}

pub fn render(app: &mut sol::App, data: &mut AppData) -> Result<(), sol::AppRenderError> {
    let (semaphore, frame_index) = app.renderer.acquire_next_image()?;

    let ref mut frame_ubo = data.per_frame[frame_index].ubo;
    frame_ubo.update(&[SceneUniforms::from(
        &data.manip.camera,
        uvec3(app.window.get_width(), app.window.get_height(), app.elapsed_ticks as u32)
    )]);

    let cmd = app.renderer.begin_command_buffer();
    let device = app.renderer.context.device();

    unsafe {
        device.cmd_push_constants(
            cmd,
            data.pipeline_layout.handle(),
            vk::ShaderStageFlags::RAYGEN_KHR,
            0,
            &data.accumulation_start_frame.to_ne_bytes(),
        )
    }

    data.scene_description.tlas_regenerate(cmd);

    data.render_target.transition_image_layout(
        cmd,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::GENERAL,
    );

    let desc_pass = data.layout_pass.get_or_create(
        sol::DescriptorSetInfo::default()
            .accel_struct(0, data.scene_description.tlas().handle())
            .image(1, data.accum_target.get_descriptor_info())
            .image(2, data.render_target.get_descriptor_info())
            .buffer(
                3,
                data.scene_description
                    .get_instances_buffer()
                    .get_descriptor_info(),
            )
            .buffers(4, data.scene_description.get_vertex_descriptors().clone())
            .buffers(5, data.scene_description.get_index_descriptors().clone())
            .buffers(6, data.scene_description.get_material_descriptors().clone()),
    );

    let descriptor_sets = [data.per_frame[frame_index].desc_set.handle(), desc_pass.handle()];
    unsafe {
        device.cmd_set_scissor(cmd, 0, &[app.window.get_rect()]);
        device.cmd_set_viewport(cmd, 0, &[app.window.get_viewport()]);
        device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            data.pipeline.handle(),
        );
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            data.pipeline_layout.handle(),
            0,
            descriptor_sets.as_slice(),
            &[],
        );
    }
    data.sbt.cmd_trace_rays(cmd, app.window.get_extent_3d());

    let present_image = app.renderer.swapchain.get_present_image(frame_index);
    data.render_target.cmd_blit_to(cmd, present_image, true);
    present_image.transition_image_layout(
        cmd,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::PRESENT_SRC_KHR,
    );
    app.renderer.end_command_buffer(cmd);
    app.renderer.submit_and_present(cmd, semaphore)
}

pub fn prepare() -> sol::AppSettings {
    sol::AppSettings {
        name: "Pathtrace App".to_string(),
        resolution: [1280, 720],
        render: sol::RendererSettings {
            extensions: vec![vk::KhrGetPhysicalDeviceProperties2Fn::name()],
            ..Default::default()
        },
    }
}

pub fn main() {
    sol::App::build(setup)
        .prepare(prepare)
        .render(render)
        .window_event(window_event)
        .run();
}
