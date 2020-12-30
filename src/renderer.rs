use crate::*;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;
use std::{ffi::CStr, mem::ManuallyDrop};

struct AppFrameData {
    pub index: usize,
    pub in_flight_fence: vk::Fence,
    pub semaphore_pool: SemaphorePool,
}

pub enum AppRenderError {
    DirtySwapchain,
}

static QUERY_POOL_SIZE: u32 = 128;
static QUERY_BEGIN_FRAME: u32 = 0;
static QUERY_END_FRAME: u32 = 1;

#[derive(Clone, Debug)]
pub struct RendererSettings {
    pub samples: u8,
    pub depth: bool,
    pub clear_color: glam::Vec4,
    pub present_mode: vk::PresentModeKHR,
    //TODO: Implement frames in flight number that differs from swapchain count
    //pub frames_in_flight: usize,
    pub extensions: Vec<&'static CStr>,
    pub device_extensions: Vec<&'static CStr>,
}

impl std::default::Default for RendererSettings {
    fn default() -> Self {
        RendererSettings {
            samples: 1,
            depth: true,
            clear_color: glam::Vec4::zero(),
            present_mode: vk::PresentModeKHR::FIFO,
            //frames_in_flight: 2,
            extensions: Vec::new(),
            device_extensions: Vec::new(),
        }
    }
}

pub struct AppRenderer {
    pub context: Arc<Context>,
    pub swapchain: ManuallyDrop<Swapchain>,
    pub renderpass: RenderPass,
    pub active_frame_index: usize,
    frames: Vec<AppFrameData>,
    framebuffers: Vec<vk::Framebuffer>,
    clear_values: [vk::ClearValue; 2],
    settings: RendererSettings,
    query_pool: vk::QueryPool,
    pub gpu_frame_time: f32,
}

impl AppRenderer {
    pub fn new(window: &mut Window, settings: RendererSettings) -> Self {
        unsafe {
            let shared_context = Arc::new(SharedContext::new(window, &settings));
            let mut swapchain = Swapchain::new(shared_context.clone(), &window, &settings);
            let context = Arc::new(Context::new(
                shared_context.clone(),
                swapchain.get_image_count(),
            ));
            swapchain.transition_depth_images(&context);
            let renderpass = swapchain.create_compatible_render_pass();
            let framebuffers = swapchain.create_framebuffers(&renderpass, &window);

            let fence_create_info =
                vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            let mut frames = Vec::<AppFrameData>::new();
            for i in 0..swapchain.get_image_count() {
                let frame = AppFrameData {
                    index: i,
                    in_flight_fence: shared_context
                        .device()
                        .create_fence(&fence_create_info, None)
                        .expect("Create fence failed."),
                    semaphore_pool: SemaphorePool::new(shared_context.clone()),
                };
                frames.push(frame);
            }
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: settings.clear_color.into(),
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let query_create_info = vk::QueryPoolCreateInfo::builder()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(QUERY_POOL_SIZE);
            let query_pool = context
                .device()
                .create_query_pool(&query_create_info, None)
                .expect("Failed to create query pool.");

            AppRenderer {
                swapchain: ManuallyDrop::new(swapchain),
                frames,
                renderpass,
                framebuffers,
                clear_values,
                context,
                active_frame_index: 0,
                settings,
                query_pool,
                gpu_frame_time: 0.0,
            }
        }
    }

    pub fn wait_for_and_reset_fence(&self, fence: vk::Fence) {
        unsafe {
            let fences = [fence];
            self.context
                .device()
                .wait_for_fences(&fences, true, std::u64::MAX)
                .expect("Wait for fence failed.");

            self.context.device().reset_fences(&fences).unwrap();
        }
    }

    pub fn recreate_swapchain(&mut self, window: &Window) {
        unsafe {
            self.context.device().device_wait_idle().unwrap();
            ManuallyDrop::drop(&mut self.swapchain);
        }
        self.swapchain = ManuallyDrop::new(Swapchain::new(
            self.context.shared().clone(),
            window,
            &self.settings,
        ));
        self.swapchain.transition_depth_images(&self.context);
        for framebuffer in self.framebuffers.iter() {
            unsafe {
                self.context
                    .device()
                    .destroy_framebuffer(*framebuffer, None);
            }
        }
        self.framebuffers = self
            .swapchain
            .create_framebuffers(&self.renderpass, &window);
    }

    pub fn acquire_next_image(&mut self) -> Result<(vk::Semaphore, usize), AppRenderError> {
        unsafe {
            let aquired_semaphore = self.frames[self.active_frame_index]
                .semaphore_pool
                .request_semaphore();
            let result = self.swapchain.swapchain_loader.acquire_next_image(
                self.swapchain.handle(),
                std::u64::MAX,
                aquired_semaphore,
                vk::Fence::null(),
            );
            let image_index = match result {
                Ok((image_index, _)) => image_index,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    return Err(AppRenderError::DirtySwapchain);
                }
                Err(vk::Result::SUBOPTIMAL_KHR) => {
                    return Err(AppRenderError::DirtySwapchain);
                }
                Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
            };

            self.active_frame_index = image_index as usize;
            self.frames[self.active_frame_index].semaphore_pool.reset();
            self.wait_for_and_reset_fence(self.frames[self.active_frame_index].in_flight_fence);

            Ok((aquired_semaphore, self.active_frame_index))
        }
    }

    pub fn begin_command_buffer(&mut self) -> vk::CommandBuffer {
        let cmd = self.context.request_command_buffer(self.active_frame_index);
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.context
                .device()
                .begin_command_buffer(cmd, &begin_info)
                .expect("Begin frame commands.");

            self.context
                .device()
                .cmd_reset_query_pool(cmd, self.query_pool, 0, QUERY_POOL_SIZE);

            self.context.device().cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                QUERY_BEGIN_FRAME,
            );
        }
        cmd
    }

    pub fn end_command_buffer(&self, cmd: vk::CommandBuffer) {
        unsafe {
            self.context.device().cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                QUERY_END_FRAME,
            );
            self.context
                .device()
                .end_command_buffer(cmd)
                .expect("End frame commands.");
        }
    }

    pub fn begin_renderpass(&self, command_buffer: vk::CommandBuffer, extent: vk::Extent2D) {
        unsafe {
            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(self.renderpass.handle())
                .framebuffer(self.framebuffers[self.active_frame_index])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .clear_values(&self.clear_values)
                .build();
            self.context.device().cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
        }
    }

    pub fn end_renderpass(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.context.device().cmd_end_render_pass(command_buffer);
        }
    }

    pub fn submit_and_present(
        &mut self,
        command_buffer: vk::CommandBuffer,
        wait_semaphore: vk::Semaphore,
    ) -> Result<(), AppRenderError> {
        let rendering_complete_semaphore = self.submit_frame(
            &[command_buffer],
            &[wait_semaphore],
            &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
        );
        self.present_frame(rendering_complete_semaphore)?;

        let mut query_data = [0u32; 2];
        unsafe {
            self.context
                .device()
                .get_query_pool_results(
                    self.query_pool,
                    0,
                    2,
                    &mut query_data,
                    vk::QueryResultFlags::WAIT,
                )
                .expect("Failed to read query results");
        }
        let begin_time = query_data[0] as f32
            * self.context.get_physical_device_limits().timestamp_period
            * 1e-6;
        let end_time = query_data[1] as f32
            * self.context.get_physical_device_limits().timestamp_period
            * 1e-6;
        self.gpu_frame_time = end_time - begin_time;
        Ok(())
    }

    pub fn submit_frame(
        &mut self,
        command_buffers: &[vk::CommandBuffer],
        wait_semaphores: &[vk::Semaphore],
        stage_flags: &[vk::PipelineStageFlags],
    ) -> vk::Semaphore {
        unsafe {
            let rendering_complete_semaphore = self.frames[self.active_frame_index]
                .semaphore_pool
                .request_semaphore();
            let signal_semaphores = [rendering_complete_semaphore];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(stage_flags)
                .command_buffers(command_buffers)
                .signal_semaphores(&signal_semaphores);

            self.context
                .device()
                .queue_submit(
                    self.context.graphics_queue(),
                    &[submit_info.build()],
                    self.frames[self.active_frame_index].in_flight_fence,
                )
                .expect("queue submit failed.");

            rendering_complete_semaphore
        }
    }

    pub fn present_frame(&self, wait_semaphore: vk::Semaphore) -> Result<(), AppRenderError> {
        let wait_semaphores = [wait_semaphore];
        let swapchains = [self.swapchain.handle()];
        let image_indices = [self.active_frame_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            let result = self
                .swapchain
                .swapchain_loader
                .queue_present(self.context.present_queue(), &present_info);

            match result {
                Ok(_) => {}
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    return Err(AppRenderError::DirtySwapchain);
                }
                Err(vk::Result::SUBOPTIMAL_KHR) => {
                    return Err(AppRenderError::DirtySwapchain);
                }
                Err(error) => panic!("Error while presenting image. Cause: {}", error),
            };

            Ok(())
        }
    }

    pub fn begin_frame_default(
        &mut self,
    ) -> Result<(ash::vk::Semaphore, ash::vk::CommandBuffer), AppRenderError> {
        let (image_aquired_semaphore, _) = self.acquire_next_image()?;
        let cmd = self.begin_command_buffer();
        self.begin_renderpass(cmd, self.swapchain.get_extent());
        Ok((image_aquired_semaphore, cmd))
    }

    pub fn end_frame_default(
        &mut self,
        image_aquired_semaphore: ash::vk::Semaphore,
        cmd: ash::vk::CommandBuffer,
    ) -> Result<(), AppRenderError> {
        self.end_renderpass(cmd);
        self.end_command_buffer(cmd);
        self.submit_and_present(cmd, image_aquired_semaphore)?;
        Ok(())
    }

    pub fn get_renderpass(&self) -> vk::RenderPass {
        self.renderpass.handle()
    }

    pub fn get_frames_count(&self) -> usize {
        self.frames.len()
    }
}

impl Drop for AppRenderer {
    fn drop(&mut self) {
        unsafe {
            let ctx = self.context.as_ref();
            let device = ctx.device();

            device.destroy_query_pool(self.query_pool, None);

            device.device_wait_idle().unwrap();

            for framebuffer in self.framebuffers.iter() {
                device.destroy_framebuffer(*framebuffer, None);
            }

            self.frames.iter().for_each(|fence| {
                device.destroy_fence(fence.in_flight_fence, None);
            });
        }
    }
}
