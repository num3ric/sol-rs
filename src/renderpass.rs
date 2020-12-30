use crate::{Image2d, Resource, SharedContext};
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

#[derive(Default)]
pub struct RenderPassInfo<'a> {
    pub color_images: Vec<&'a Image2d>,
    pub depth_stencil_image: Option<&'a Image2d>,
    pub resolve_images: Vec<&'a Image2d>,
    pub present: bool,
    pub samples: vk::SampleCountFlags,
    pub final_layout: vk::ImageLayout,
}

#[derive(Clone, Default)]
pub struct TransientRenderPassInfo {
    pub color_formats: Vec<vk::Format>,
    pub depth_stencil_format: Option<vk::Format>,
    pub resolve_formats: Vec<vk::Format>,
    pub samples: vk::SampleCountFlags,
}

pub struct RenderPass {
    context: Arc<SharedContext>,
    render_pass: vk::RenderPass,
}

impl RenderPass {
    pub fn new(context: Arc<SharedContext>, info: RenderPassInfo) -> Self {
        unsafe {
            let mut index = 0u32;
            let mut attachments_desc = Vec::<vk::AttachmentDescription>::new();

            let mut color_attachment_refs = Vec::<vk::AttachmentReference>::new();
            for color_image in info.color_images {
                let mut layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                if info.present && info.resolve_images.is_empty() {
                    layout = info.final_layout;
                }
                attachments_desc.push(
                    vk::AttachmentDescription::builder()
                        .format(color_image.get_format())
                        .samples(info.samples)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .final_layout(layout)
                        .build(),
                );
                color_attachment_refs.push(vk::AttachmentReference {
                    attachment: index,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                });
                index += 1;
            }

            let mut depth_attachment_refs = Vec::<vk::AttachmentReference>::new();
            match info.depth_stencil_image {
                Some(image) => {
                    attachments_desc.push(
                        vk::AttachmentDescription::builder()
                            .format(image.get_format())
                            .samples(info.samples)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                            .build(),
                    );
                    depth_attachment_refs.push(vk::AttachmentReference {
                        attachment: index,
                        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    });
                    index += 1;
                }
                None => {}
            }

            let mut resolve_attachment_refs = Vec::<vk::AttachmentReference>::new();
            for resolve_image in &info.resolve_images {
                attachments_desc.push(
                    vk::AttachmentDescription::builder()
                        .format(resolve_image.get_format())
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .final_layout(info.final_layout)
                        .build(),
                );
                resolve_attachment_refs.push(vk::AttachmentReference {
                    attachment: index,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                });
                index += 1;
            }

            let dependencies = [vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                ..Default::default()
            }];

            let mut subpass_builder = vk::SubpassDescription::builder()
                .color_attachments(&color_attachment_refs)
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);
            if info.depth_stencil_image.is_some() {
                subpass_builder =
                    subpass_builder.depth_stencil_attachment(&depth_attachment_refs[0])
            }
            if !info.resolve_images.is_empty() {
                subpass_builder = subpass_builder.resolve_attachments(&resolve_attachment_refs);
            }
            let subpasses = [subpass_builder.build()];

            let create_info = vk::RenderPassCreateInfo::builder()
                .attachments(&attachments_desc)
                .subpasses(&subpasses)
                .dependencies(&dependencies);
            let render_pass = context
                .device()
                .create_render_pass(&create_info, None)
                .unwrap();
            Self {
                context,
                render_pass,
            }
        }
    }

    pub fn new_transient(context: Arc<SharedContext>, info: TransientRenderPassInfo) -> Self {
        let mut index = 0u32;
        let mut attachments_desc = Vec::<vk::AttachmentDescription>::new();
        let mut color_attachment_refs = Vec::<vk::AttachmentReference>::new();
        for color_format in info.color_formats {
            attachments_desc.push(
                vk::AttachmentDescription::builder()
                    .format(color_format)
                    .samples(info.samples)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) //ignored
                    .build(),
            );
            color_attachment_refs.push(vk::AttachmentReference {
                attachment: index,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            });
            index += 1;
            break;
        }

        let mut depth_attachment_refs = Vec::<vk::AttachmentReference>::new();
        match info.depth_stencil_format {
            Some(format) => {
                attachments_desc.push(
                    vk::AttachmentDescription::builder()
                        .format(format)
                        .samples(info.samples)
                        .load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) //ignored
                        .build(),
                );
                depth_attachment_refs.push(vk::AttachmentReference {
                    attachment: index,
                    layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                });
                index += 1;
            }
            None => {}
        }

        let mut resolve_attachment_refs = Vec::<vk::AttachmentReference>::new();
        for resolve_format in &info.resolve_formats {
            attachments_desc.push(
                vk::AttachmentDescription::builder()
                    .format(resolve_format.clone())
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .final_layout(vk::ImageLayout::PRESENT_SRC_KHR) //ignored
                    .build(),
            );
            resolve_attachment_refs.push(vk::AttachmentReference {
                attachment: index,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            });
            //index += 1;
            break;
        }
        let mut subpass_builder = vk::SubpassDescription::builder()
            .color_attachments(&color_attachment_refs)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);
        if info.depth_stencil_format.is_some() {
            subpass_builder = subpass_builder.depth_stencil_attachment(&depth_attachment_refs[0])
        }
        if !info.resolve_formats.is_empty() {
            subpass_builder = subpass_builder.resolve_attachments(&resolve_attachment_refs);
        }
        let subpasses = [subpass_builder.build()];
        let render_pass = unsafe {
            context
                .device()
                .create_render_pass(
                    &vk::RenderPassCreateInfo::builder()
                        .attachments(&attachments_desc)
                        .subpasses(&subpasses)
                        .dependencies(&[vk::SubpassDependency {
                            src_subpass: vk::SUBPASS_EXTERNAL,
                            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                            ..Default::default()
                        }]),
                    None,
                )
                .unwrap()
        };
        Self {
            context,
            render_pass,
        }
    }

    pub fn new_raw(context: Arc<SharedContext>, create_info: &vk::RenderPassCreateInfo) -> Self {
        unsafe {
            let render_pass = context
                .device()
                .create_render_pass(create_info, None)
                .unwrap();
            Self {
                context,
                render_pass,
            }
        }
    }
}

impl Resource<vk::RenderPass> for RenderPass {
    fn handle(&self) -> vk::RenderPass {
        self.render_pass
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device()
                .destroy_render_pass(self.render_pass, None);
        }
    }
}
