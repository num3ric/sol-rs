//#![windows_subsystem = "windows"]
use sol::prelude::*;
use sol::{scene, util};

pub struct PerFrameData {
    pub ubo: sol::Buffer,
    pub desc_set: sol::DescriptorSet,
}
pub struct AppData {
    pub vertex_buffer: sol::Buffer,
    pub texture: sol::Texture2d,
    pub desc_set_layout: sol::DescriptorSetLayout,
    pub pipeline_layout: sol::PipelineLayout,
    pub pipeline: sol::Pipeline,
    pub per_frame: Vec<PerFrameData>,
    pub manip: scene::CameraManip,
}

pub fn setup(app: &mut sol::App) -> AppData {
    let context = &app.renderer.context;

    let vertex_buffer = sol::Buffer::from_data(
        context.clone(),
        sol::BufferInfo::default()
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .gpu_only(),
        &util::colored_cube_vertices(),
    );
    let texture = sol::Texture2d::new(context.clone(), util::find_asset("textures/face.png").unwrap());

    let mut desc_set_layout = sol::DescriptorSetLayout::new(
        context.clone(),
        sol::DescriptorSetLayoutInfo::default()
            .binding(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::ALL,
            )
            .binding(
                1,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::ALL,
            ),
    );
    let pipeline_layout = sol::PipelineLayout::new(
        context.clone(),
        sol::PipelineLayoutInfo::default().desc_set_layout(desc_set_layout.handle()),
    );
    let pipeline = sol::Pipeline::new(
        context.clone(),
        sol::PipelineInfo::default()
            .layout(pipeline_layout.handle())
            .render_pass_info(app.renderer.swapchain.get_transient_render_pass_info())
            .vert(util::find_asset("glsl/cube.vert").unwrap())
            .frag(util::find_asset("glsl/cube.frag").unwrap())
            .vertex_type::<util::BasicVertex>(),
    );

    let mut camera = scene::Camera::new(app.window.get_size());
    camera.look_at(Vec3::splat(5.0), Vec3::ZERO, -Vec3::Y);

    let vp = camera.perspective_matrix() * camera.view_matrix();
    let mut per_frame = Vec::<PerFrameData>::new();
    for _ in 0..app.renderer.get_frames_count() {
        let ubo = sol::Buffer::from_data(
            context.clone(),
            sol::BufferInfo::default()
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .cpu_to_gpu(),
            &vp.to_cols_array(),
        );
        let desc_set = desc_set_layout.get_or_create(
            sol::DescriptorSetInfo::default()
                .buffer(0, ubo.get_descriptor_info())
                .image(1, texture.get_descriptor_info()),
        );
        per_frame.push(PerFrameData { ubo, desc_set });
    }

    AppData {
        vertex_buffer,
        texture,
        pipeline,
        desc_set_layout,
        pipeline_layout,
        per_frame,
        manip: scene::CameraManip {
            camera,
            input: scene::CameraInput::default(),
        },
    }
}

pub fn window_event(_: &mut sol::App, data: &mut AppData, event: &winit::event::WindowEvent) {
    data.manip.update(&event);
}

pub fn render(app: &mut sol::App, data: &mut AppData) -> Result<(), sol::AppRenderError> {
    let (image_aquired_semaphore, cmd) = app.renderer.begin_frame_default()?;
    let ref camera = data.manip.camera;
    let vp = camera.perspective_matrix() * camera.view_matrix();
    data.per_frame[app.renderer.active_frame_index]
        .ubo
        .update(&vp.to_cols_array());
    let descriptor_sets = [data.per_frame[app.renderer.active_frame_index].desc_set.handle()];
    let device = app.renderer.context.device();
    unsafe {
        device.cmd_set_scissor(cmd, 0, &[app.window.get_rect()]);
        device.cmd_set_viewport(cmd, 0, &[app.window.get_viewport()]);
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, data.pipeline.handle());
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            data.pipeline_layout.handle(),
            0,
            descriptor_sets.as_slice(),
            &[],
        );
        device.cmd_bind_vertex_buffers(cmd, 0, &[data.vertex_buffer.handle()], &[0]);
        device.cmd_draw(cmd, data.vertex_buffer.get_element_count(), 1, 0, 1);
    }
    app.renderer.end_frame_default(image_aquired_semaphore, cmd)
}

pub fn prepare() -> sol::AppSettings {
    sol::AppSettings {
        name: "Cube App".to_string(),
        resolution: [900, 600],
        render: sol::RendererSettings {
            samples: 8,
            clear_color: vec4(13.0 / 255.0, 17.0 / 255.0, 23.0 / 255.0, 1.0),
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
