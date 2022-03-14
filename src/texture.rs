use crate::{Buffer, BufferInfo, Context, Resource, SharedContext};
use ash::{vk};
use image::GenericImageView;
use std::{cmp::max, sync::Arc};
use std::{path::PathBuf, ptr};

//TODO: image resource trait

fn has_stencil_component(format: vk::Format) -> bool {
    format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
}

fn check_mipmap_support(context: &Arc<SharedContext>, image_format: vk::Format) -> bool {
    let format_properties = unsafe {
        context
            .instance()
            .get_physical_device_format_properties(context.physical_device(), image_format)
    };

    format_properties
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
}

pub struct Image2d {
    context: Arc<SharedContext>,
    image: vk::Image,
    extent: vk::Extent3D,
    view: vk::ImageView,
    layout: vk::ImageLayout,
    format: vk::Format,
    allocation: Option<vk_mem::Allocation>,
}

impl Image2d {
    pub fn new(
        context: Arc<SharedContext>,
        image_info: &vk::ImageCreateInfo,
        aspect_mask: vk::ImageAspectFlags,
        level_count: u32,
    ) -> Self {
        unsafe {
            assert!(image_info.extent.width + image_info.extent.height > 2);

            let mut allocation_info = vk_mem::AllocationCreateInfo::default();
            allocation_info.usage = vk_mem::MemoryUsage::GpuOnly;
            let (image, alloc, _) = context
                .allocator()
                .create_image(&image_info, &allocation_info)
                .unwrap();

            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(aspect_mask)
                .level_count(level_count)
                .layer_count(1)
                .build();
            let image_view_info = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .subresource_range(subresource_range)
                .image(image)
                .format(image_info.format);

            let image_view = context
                .device()
                .create_image_view(&image_view_info, None)
                .unwrap();

            Image2d {
                context,
                image,
                extent: vk::Extent3D {
                    width: image_info.extent.width,
                    height: image_info.extent.height,
                    depth: 1,
                },
                view: image_view,
                format: image_info.format,
                allocation: Some(alloc),
                layout: vk::ImageLayout::UNDEFINED,
            }
        }
    }

    pub fn from_swapchain(
        context: Arc<SharedContext>,
        image: vk::Image,
        extent: vk::Extent2D,
        image_format: vk::Format,
    ) -> Self {
        unsafe {
            let create_view_info = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(image_format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(image)
                .build();
            let image_view = context
                .device()
                .create_image_view(&create_view_info, None)
                .unwrap();
            Image2d {
                context,
                image,
                extent: vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
                view: image_view,
                format: image_format,
                allocation: None,
                layout: vk::ImageLayout::UNDEFINED,
            }
        }
    }

    pub fn get_image_view(&self) -> vk::ImageView {
        self.view
    }

    pub fn get_format(&self) -> vk::Format {
        self.format
    }

    pub fn transition_image_layout(
        &mut self,
        cmd: vk::CommandBuffer,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
    ) {
        self.transition_image_layout_mip(cmd, old, new, 1);
    }

    pub fn transition_image_layout_mip(
        &mut self,
        cmd: vk::CommandBuffer,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
        mip_levels: u32,
    ) {
        if old == new {
            self.layout = new;
            return;
        }
        let mut aspect_mask = vk::ImageAspectFlags::COLOR;
        if new == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
            aspect_mask = vk::ImageAspectFlags::DEPTH;
            if has_stencil_component(self.format) {
                aspect_mask |= vk::ImageAspectFlags::STENCIL;
            }
        }
        let src_access_mask = match old {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
            vk::ImageLayout::PREINITIALIZED => vk::AccessFlags::HOST_WRITE,
            vk::ImageLayout::GENERAL => {
                vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::SHADER_WRITE
            }
            _ => vk::AccessFlags::default(),
        };
        let dst_access_mask = match new {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
            }
            vk::ImageLayout::GENERAL => vk::AccessFlags::empty(),
            vk::ImageLayout::PRESENT_SRC_KHR => vk::AccessFlags::empty(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => vk::AccessFlags::SHADER_READ,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => vk::AccessFlags::TRANSFER_READ,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
            _ => panic!("Unsupported layout transition!"),
        };
        let src_stage = match old {
            vk::ImageLayout::GENERAL => vk::PipelineStageFlags::ALL_COMMANDS,
            vk::ImageLayout::PREINITIALIZED => vk::PipelineStageFlags::HOST,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::PipelineStageFlags::TRANSFER,
            vk::ImageLayout::UNDEFINED => vk::PipelineStageFlags::TOP_OF_PIPE,
            _ => vk::PipelineStageFlags::ALL_COMMANDS,
        };
        let dst_stage = match new {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => {
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            }
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            }
            vk::ImageLayout::GENERAL => vk::PipelineStageFlags::HOST,
            vk::ImageLayout::PRESENT_SRC_KHR => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => vk::PipelineStageFlags::TRANSFER,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::PipelineStageFlags::TRANSFER,
            _ => vk::PipelineStageFlags::ALL_COMMANDS,
        };

        let layout_transition_barriers = vk::ImageMemoryBarrier::builder()
            .image(self.image)
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .new_layout(new)
            .old_layout(old)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(aspect_mask)
                    .layer_count(1)
                    .level_count(mip_levels)
                    .build(),
            );
        unsafe {
            self.context.device().cmd_pipeline_barrier(
                cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[layout_transition_barriers.build()],
            );
        }

        self.layout = new;
    }

    pub fn copy_to_image(&self, context: &Arc<Context>, buffer: vk::Buffer) {
        let region = vk::BufferImageCopy::builder()
            .image_subresource(
                vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .build(),
            )
            .image_extent(self.extent)
            .build();
        let cmd = context.begin_single_time_cmd();
        unsafe {
            context.device().cmd_copy_buffer_to_image(
                cmd,
                buffer,
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
        context.end_single_time_cmd(cmd);
    }

    pub fn cmd_blit_to(&mut self, cmd: vk::CommandBuffer, dst: &mut Image2d, do_transitions: bool) {
        if do_transitions {
            dst.transition_image_layout(cmd, dst.layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            self.transition_image_layout(cmd, self.layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        }
        let region = vk::ImageBlit::builder()
            .src_subresource(
                vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .build(),
            )
            .dst_subresource(
                vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .build(),
            )
            .src_offsets([
                vk::Offset3D::default(),
                vk::Offset3D::builder()
                    .x(self.extent.width as i32)
                    .y(self.extent.height as i32)
                    .z(1)
                    .build(),
            ])
            .dst_offsets([
                vk::Offset3D::default(),
                vk::Offset3D::builder()
                    .x(dst.extent.width as i32)
                    .y(dst.extent.height as i32)
                    .z(1)
                    .build(),
            ])
            .build();

        unsafe {
            self.context.device().cmd_blit_image(
                cmd,
                self.handle(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.handle(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
                vk::Filter::NEAREST,
            );
        }
    }

    pub fn generate_mipmaps(&self, context: &Arc<Context>, mip_levels: u32) {
        let command_buffer = context.begin_single_time_cmd();

        let mut image_barrier = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::empty(),
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::UNDEFINED,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: self.image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        };

        let mut mip_width = self.extent.width as i32;
        let mut mip_height = self.extent.height as i32;

        for i in 1..mip_levels {
            image_barrier.subresource_range.base_mip_level = i - 1;
            image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            image_barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            image_barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

            unsafe {
                context.device().cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier.clone()],
                );
            }

            let blits = [vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: max(mip_width / 2, 1),
                        y: max(mip_height / 2, 1),
                        z: 1,
                    },
                ],
            }];

            unsafe {
                context.device().cmd_blit_image(
                    command_buffer,
                    self.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &blits,
                    vk::Filter::LINEAR,
                );
            }

            image_barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            image_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            unsafe {
                self.context.device().cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier.clone()],
                );
            }

            mip_width = max(mip_width / 2, 1);
            mip_height = max(mip_height / 2, 1);
        }

        image_barrier.subresource_range.base_mip_level = mip_levels - 1;
        image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        image_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        unsafe {
            context.device().cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_barrier.clone()],
            );
        }

        context.end_single_time_cmd(command_buffer);
    }

    pub fn get_descriptor_info(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo::builder()
            .sampler(vk::Sampler::null())
            .image_view(self.view)
            .image_layout(self.layout)
            .build()
    }
}

impl crate::Resource<vk::Image> for Image2d {
    fn handle(&self) -> vk::Image {
        self.image
    }
}

impl Drop for Image2d {
    fn drop(&mut self) {
        unsafe {
            self.context.device().destroy_image_view(self.view, None);

            match self.allocation {
                Some(alloc) => {
                    self.context.allocator().destroy_image(self.image, alloc);
                }
                None => {}
            }
        }
    }
}

pub struct Texture2d {
    context: Arc<Context>,
    image2d: Image2d,
    sampler: vk::Sampler,
}

impl Texture2d {
    pub fn new(context: Arc<Context>, filepath: PathBuf) -> Self {
        let mut source_image = image::open(filepath).expect("Failed to find image."); // this function is slow in debug mode.
        source_image = source_image.flipv();
        let size = source_image.dimensions();
        let image_data = source_image.to_rgba8().into_raw();
        let mip_levels = ((::std::cmp::max(size.0, size.1) as f32).log2().floor() as u32) + 1;

        let format = vk::Format::R8G8B8A8_UNORM;
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: size.0,
                height: size.1,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let mut image2d = Image2d::new(
            context.shared().clone(),
            &image_info,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        );

        {
            //Load data via temporary transfer buffer
            let transfer_buffer = Buffer::from_data(
                context.clone(),
                BufferInfo::default()
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .cpu_only(),
                &image_data,
            );
            let cmd = context.begin_single_time_cmd();
            image2d.transition_image_layout_mip(
                cmd,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                mip_levels,
            );
            context.end_single_time_cmd(cmd);

            image2d.copy_to_image(&context, transfer_buffer.handle());
            if mip_levels > 1 && check_mipmap_support(&context.shared(), image2d.get_format()) {
                image2d.generate_mipmaps(&context, mip_levels);
                let cmd = context.begin_single_time_cmd();
                image2d.transition_image_layout_mip(
                    cmd,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    mip_levels,
                );
                context.end_single_time_cmd(cmd);
            } else {
                let cmd = context.begin_single_time_cmd();
                image2d.transition_image_layout_mip(
                    cmd,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    mip_levels,
                );
                context.end_single_time_cmd(cmd);
            }
        }

        let sampler_create_info = vk::SamplerCreateInfo::builder()
            .min_filter(vk::Filter::LINEAR)
            .mag_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
            .anisotropy_enable(true)
            .max_anisotropy(16.0)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .min_lod(0.0)
            .max_lod(mip_levels as f32)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .unnormalized_coordinates(false);
        let sampler: vk::Sampler;
        unsafe {
            sampler = context
                .device()
                .create_sampler(&sampler_create_info, None)
                .unwrap();
        }

        Texture2d {
            context: context.clone(),
            image2d,
            sampler,
        }
    }

    pub fn get_image2d(&self) -> &Image2d {
        &self.image2d
    }

    pub fn get_sampler(&self) -> vk::Sampler {
        self.sampler
    }

    pub fn get_descriptor_info(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo::builder()
            .sampler(self.sampler)
            .image_view(self.image2d.view)
            .image_layout(self.image2d.layout)
            .build()
    }
}

impl Drop for Texture2d {
    fn drop(&mut self) {
        unsafe {
            self.context.device().destroy_sampler(self.sampler, None);
        }
    }
}
