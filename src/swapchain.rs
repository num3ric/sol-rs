use crate::{
    Context, Image2d, RenderPass, RenderPassInfo, RendererSettings, Resource, SharedContext,
    TransientRenderPassInfo, Window,
};
use ash::vk;
use ash::{extensions::khr};
use std::sync::Arc;

pub struct Swapchain {
    context: Arc<SharedContext>,
    pub swapchain_loader: khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    present_images: Vec<Image2d>,
    depth_stencil_images: Vec<Image2d>,
    resolve_images: Vec<Image2d>,
    sample_count: vk::SampleCountFlags,
    extent: vk::Extent2D,
}

impl Swapchain {
    pub fn new(context: Arc<SharedContext>, window: &Window, settings: &RendererSettings) -> Self {
        unsafe {
            let mut sample_count = vk::SampleCountFlags::TYPE_1;
            if settings.samples == 2 {
                sample_count = vk::SampleCountFlags::TYPE_2;
            } else if settings.samples == 4 {
                sample_count = vk::SampleCountFlags::TYPE_4;
            } else if settings.samples == 8 {
                sample_count = vk::SampleCountFlags::TYPE_8;
            } else if settings.samples == 16 {
                sample_count = vk::SampleCountFlags::TYPE_16;
            } else if settings.samples == 32 {
                sample_count = vk::SampleCountFlags::TYPE_32;
            } else if settings.samples == 64 {
                sample_count = vk::SampleCountFlags::TYPE_64;
            }
            let pdevice = context.physical_device();
            let surface_capabilities = window.get_surface_capabilities(pdevice);
            let mut desired_image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0
                && desired_image_count > surface_capabilities.max_image_count
            {
                desired_image_count = surface_capabilities.max_image_count;
            }
            let extent = window.get_surface_extent(pdevice);
            let surface_format = window.get_surface_format(pdevice);
            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };
            let image_format = surface_format.format;
            let present_mode = window.get_surface_present_mode(pdevice, settings.present_mode);
            let swapchain_loader = khr::Swapchain::new(context.instance(), context.device());
            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(window.surface())
                .min_image_count(desired_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(image_format)
                .image_extent(extent)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1);
            let swapchain = swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();

            let swapchain_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
            let present_images: Vec<Image2d> = swapchain_images
                .iter()
                .map(|image| Image2d::from_swapchain(context.clone(), *image, extent, image_format))
                .collect();

            let mut depth_stencil_images = Vec::<Image2d>::new();
            if settings.depth {
                for _ in 0..present_images.len() {
                    let depth_image_create_info = vk::ImageCreateInfo::builder()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::D16_UNORM)
                        .extent(window.get_extent_3d())
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(sample_count)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE);
                    depth_stencil_images.push(Image2d::new(
                        context.clone(),
                        &depth_image_create_info,
                        vk::ImageAspectFlags::DEPTH,
                        1,
                        "SwapchainDepthStencil"
                    ));
                }
            }

            let mut resolve_images = Vec::<Image2d>::new();
            if settings.samples > 1 {
                for _ in 0..present_images.len() {
                    let image_create_info = vk::ImageCreateInfo::builder()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(image_format)
                        .extent(window.get_extent_3d())
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(sample_count)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(
                            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
                                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE);
                    resolve_images.push(Image2d::new(
                        context.clone(),
                        &image_create_info,
                        vk::ImageAspectFlags::COLOR,
                        1,
                        "SwapchainResolve"
                    ));
                }
            }

            Swapchain {
                context,
                swapchain_loader,
                swapchain,
                present_images,
                depth_stencil_images,
                resolve_images,
                sample_count,
                extent,
            }
        }
    }

    pub fn get_image_count(&self) -> usize {
        self.present_images.len()
    }

    pub fn get_present_image(&mut self, index: usize) -> &mut Image2d {
        &mut self.present_images[index]
    }

    pub fn get_sample_count(&self) -> vk::SampleCountFlags {
        self.sample_count
    }

    pub fn create_compatible_render_pass(&self) -> RenderPass {
        let color_images = vec![&self.present_images[0]];
        let mut resolve_images = Vec::<&Image2d>::new();
        match self.resolve_images.iter().nth(0) {
            Some(image) => resolve_images.push(image),
            None => {}
        };
        let depth_stencil_image = match self.depth_stencil_images.iter().nth(0) {
            Some(image) => Some(image),
            None => None,
        };
        RenderPass::new(
            self.context.clone(),
            RenderPassInfo {
                color_images,
                depth_stencil_image,
                resolve_images,
                present: true,
                samples: self.sample_count,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            },
        )
    }

    pub fn get_transient_render_pass_info(&self) -> TransientRenderPassInfo {
        let mut resolve_formats = Vec::<vk::Format>::new();
        match self.resolve_images.iter().nth(0) {
            Some(image) => resolve_formats.push(image.get_format()),
            None => {}
        };
        let depth_stencil_format = match self.depth_stencil_images.iter().nth(0) {
            Some(image) => Some(image.get_format()),
            None => None,
        };
        TransientRenderPassInfo {
            color_formats: vec![self.present_images[0].get_format()],
            depth_stencil_format,
            resolve_formats,
            samples: self.sample_count,
        }
    }

    pub fn create_framebuffers(
        &self,
        renderpass: &RenderPass,
        window: &Window,
    ) -> Vec<vk::Framebuffer> {
        let mut framebuffers = Vec::<vk::Framebuffer>::new();
        for i in 0..self.get_image_count() {
            let mut attachments = Vec::<vk::ImageView>::new();
            if self.resolve_images.is_empty() {
                attachments.push(self.present_images[i].get_image_view());
                if !self.depth_stencil_images.is_empty() {
                    attachments.push(self.depth_stencil_images[i].get_image_view());
                }
            } else {
                attachments.push(self.resolve_images[i].get_image_view());
                if !self.depth_stencil_images.is_empty() {
                    attachments.push(self.depth_stencil_images[i].get_image_view());
                }
                attachments.push(self.present_images[i].get_image_view());
            }
            let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(renderpass.handle())
                .attachments(&attachments)
                .width(window.get_extent().width)
                .height(window.get_extent().height)
                .layers(1);
            unsafe {
                framebuffers.push(
                    self.context
                        .device()
                        .create_framebuffer(&frame_buffer_create_info, None)
                        .unwrap(),
                );
            }
        }
        framebuffers
    }

    pub fn transition_depth_images(&mut self, context: &Arc<Context>) {
        let cmd = context.begin_single_time_cmd();
        self.depth_stencil_images
            .iter_mut()
            .for_each(|depth_stencil_image| {
                depth_stencil_image.transition_image_layout(
                    cmd,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                );
            });
        context.end_single_time_cmd(cmd);
    }

    pub fn get_extent(&self) -> vk::Extent2D {
        self.extent
    }
}

impl crate::Resource<vk::SwapchainKHR> for Swapchain {
    fn handle(&self) -> vk::SwapchainKHR {
        self.swapchain
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            // Since images are created by the swapchain, this automatically destroys them as well.
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
        }
    }
}
