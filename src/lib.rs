#![allow(dead_code)]

use winit::{
    event::{ElementState, Event, ModifiersState, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

use std::ops::Drop;
use std::time::{Duration, SystemTime};

mod buffer;
mod context;
mod descriptor;
mod pipeline;
mod pools;
pub mod prelude;
mod renderer;
mod renderpass;
pub mod scene;
mod swapchain;
mod texture;
pub mod util;
mod window;
pub mod ray;

pub use crate::buffer::*;
pub use crate::context::*;
pub use crate::descriptor::*;
pub use crate::pipeline::*;
pub use crate::pools::*;
pub use crate::renderer::*;
pub use crate::renderpass::*;
pub use crate::swapchain::*;
pub use crate::texture::*;
pub use crate::window::*;
pub use ash;
pub use glam;
pub use winit;

// Simple offset_of macro akin to C++ offsetof
#[macro_export]
macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = std::mem::zeroed();
            (&b.$field as *const _ as isize) - (&b as *const _ as isize)
        }
    }};
}

pub trait Resource<T> {
    fn handle(&self) -> T;
}

pub trait Vertex {
    fn stride() -> u32;
    fn format_offset() -> Vec<(ash::vk::Format, u32)>;
}

pub struct App {
    pub settings: AppSettings,
    pub renderer: AppRenderer,
    pub window: Window,
    pub elapsed_time: Duration,
    pub elapsed_ticks: u64,
}

impl App {
    pub fn build<T>(setup: crate::SetupFn<T>) -> AppBuilder<T> {
        AppBuilder {
            prepare: None,
            setup: setup,
            update: None,
            window_event: None,
            render: None,
        }
    }

    pub fn new(settings: AppSettings, event_loop: &EventLoop<()>) -> Self {
        let mut window = Window::new(
            settings.resolution[0],
            settings.resolution[1],
            settings.name.clone(),
            &event_loop,
        );
        let renderer = AppRenderer::new(&mut window, settings.clone().render);
        App {
            settings,
            renderer,
            window,
            elapsed_time: Duration::default(),
            elapsed_ticks: 0,
        }
    }

    pub fn recreate_swapchain(&mut self) {
        self.renderer.recreate_swapchain(&self.window);
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.renderer.swapchain);
            self.window.destroy_surface();
        }
    }
}

pub type PrepareFn = fn() -> AppSettings;
pub type SetupFn<T> = fn(&mut App) -> T; // TODO: how do we specify FnOnce here?
pub type UpdateFn<T> = fn(&mut App, &mut T);
pub type RenderFn<T> = fn(&mut App, &mut T) -> Result<(), AppRenderError>;
pub type WindowEventFn<T> = fn(&mut App, &mut T, event: &WindowEvent);

#[derive(Clone, Debug)]
pub struct AppSettings {
    pub name: String,
    pub resolution: [u32; 2],
    pub render: RendererSettings,
}

impl std::default::Default for AppSettings {
    fn default() -> Self {
        AppSettings {
            name: "App".to_string(),
            resolution: [1280, 720],
            render: RendererSettings::default(),
        }
    }
}
pub struct AppBuilder<T: 'static> {
    pub prepare: Option<PrepareFn>,
    pub setup: SetupFn<T>,
    pub update: Option<UpdateFn<T>>,
    pub window_event: Option<WindowEventFn<T>>,
    pub render: Option<RenderFn<T>>,
}

impl<T> AppBuilder<T> {
    pub fn prepare(mut self, prepare: PrepareFn) -> Self {
        self.prepare = Some(prepare);
        self
    }

    pub fn update(mut self, update: UpdateFn<T>) -> Self {
        self.update = Some(update);
        self
    }

    pub fn render(mut self, render: RenderFn<T>) -> Self {
        self.render = Some(render);
        self
    }

    pub fn window_event(mut self, window_event: WindowEventFn<T>) -> Self {
        self.window_event = Some(window_event);
        self
    }

    pub fn run(self) {
        main_loop(self);
    }
}

fn main_loop<T: 'static>(builder: AppBuilder<T>) {
    let event_loop = EventLoop::new();
    let mut settings = AppSettings::default();
    match builder.prepare {
        Some(prepare) => {
            settings = prepare();
        }
        None => {}
    }
    let mut app = App::new(settings, &event_loop);
    let mut app_data = (builder.setup)(&mut app);
    let mut dirty_swapchain = false;

    let now = SystemTime::now();
    let mut modifiers = ModifiersState::default();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        if !app.window.is_minimized() {
            
            if dirty_swapchain {
                app.recreate_swapchain();
            }

            match event {
                Event::WindowEvent { event, .. } => {
                    match event {
                        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(winit::dpi::PhysicalSize { width, height }) => {
                            println!("Window was resized to: {:?}x{:?}", width, height);
                            dirty_swapchain = true;
                        }
                        WindowEvent::KeyboardInput { input, .. } => {
                            if input.state == ElementState::Pressed {
                                if input.virtual_keycode == Some(VirtualKeyCode::Q)
                                    && (modifiers.ctrl() || modifiers.logo())
                                {
                                    *control_flow = ControlFlow::Exit;
                                }
                            }
                        }
                        WindowEvent::MouseInput { .. } => {}
                        WindowEvent::ModifiersChanged(m) => modifiers = m,
                        _ => (),
                    }
                    match builder.window_event {
                        Some(event_fn) => {
                            event_fn(&mut app, &mut app_data, &event);
                        }
                        None => {}
                    }
                }
                Event::MainEventsCleared => {
                    let now = now.elapsed().unwrap();
                    if app.elapsed_ticks % 10 == 0 {
                        let cpu_time = now.as_millis() as f32 - app.elapsed_time.as_millis() as f32;
                        let title = format!("{} | cpu:{:.1} ms, gpu:{:.1} ms", app.settings.name, cpu_time, app.renderer.gpu_frame_time);
                        app.window.set_title(&title);
                    }
                    app.elapsed_time = now;

                    match builder.update {
                        Some(update_fn) => {
                            update_fn(&mut app, &mut app_data);
                        }
                        None => {}
                    }

                    dirty_swapchain = match builder.render {
                        Some(render_fn) => {
                            matches!(
                                render_fn(&mut app, &mut app_data),
                                Err(AppRenderError::DirtySwapchain)
                            )
                        }
                        None => false,
                    };

                    app.elapsed_ticks += 1;
                }
                Event::Suspended => println!("Suspended."),
                Event::Resumed => println!("Resumed."),
                Event::LoopDestroyed => unsafe {
                    app.renderer.context.device().device_wait_idle().unwrap();
                },
                _ => {}
            }
        }
    });
}
