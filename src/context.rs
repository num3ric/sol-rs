use crate::*;
use ash::{
    extensions::{ext::DebugUtils, khr},
    vk, Device, Entry, Instance,
};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use std::borrow::Cow;
use std::mem::ManuallyDrop;
use std::ffi::{CStr, CString};
use std::{
    collections::{HashSet},
    os::raw::c_char
};
use std::sync::{Arc, Mutex};

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    vk::FALSE
}

fn find_queue_families(
    instance: &Instance,
    surface: &khr::Surface,
    surface_khr: vk::SurfaceKHR,
    device: vk::PhysicalDevice,
) -> (Option<u32>, Option<u32>) {
    let mut graphics = None;
    let mut present = None;

    let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
    for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
        let index = index as u32;

        if family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && graphics.is_none()
        {
            graphics = Some(index);
        }

        let present_support = unsafe {
            surface
                .get_physical_device_surface_support(device, index, surface_khr)
                .expect("Failed to get surface support")
        };
        if present_support && present.is_none() {
            present = Some(index);
        }

        if graphics.is_some() && present.is_some() {
            break;
        }
    }

    (graphics, present)
}

fn create_logical_device_with_graphics_queue(
    instance: &Instance,
    device: vk::PhysicalDevice,
    queue_families_indices: QueueFamiliesIndices,
    device_extensions: &Vec<&'static CStr>,
) -> (Device, vk::Queue, vk::Queue) {
    let graphics_family_index = queue_families_indices.graphics;
    let present_family_index = queue_families_indices.present;
    let queue_priorities = [1.0f32];

    let queue_create_infos = {
        // Vulkan specs does not allow passing an array containing duplicated family indices.
        // And since the family for graphics and presentation could be the same we need to
        // deduplicate it.
        let mut indices = vec![graphics_family_index, present_family_index];
        indices.dedup();

        // Now we build an array of `DeviceQueueCreateInfo`.
        // One for each different family index.
        indices
            .iter()
            .map(|index| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(*index)
                    .queue_priorities(&queue_priorities)
                    .build()
            })
            .collect::<Vec<_>>()
    };

    let supported_extensions: HashSet<String> = unsafe {
        let extension_properties = instance
            .enumerate_device_extension_properties(device).unwrap();
        //dbg!("Extension properties:\n{:#?}", &extension_properties);
        extension_properties
            .iter()
            .map(|ext| {
                std::ffi::CStr::from_ptr(ext.extension_name.as_ptr() as *const c_char)
                    .to_string_lossy()
                    .as_ref()
                    .to_owned()
            })
            .collect()
    };

    let mut device_extensions_ptrs = vec![
        vk::ExtDescriptorIndexingFn::name().as_ptr(),
        vk::ExtScalarBlockLayoutFn::name().as_ptr(),
        vk::KhrMaintenance1Fn::name().as_ptr(),
        vk::KhrMaintenance2Fn::name().as_ptr(),
        vk::KhrMaintenance3Fn::name().as_ptr(),
        vk::KhrGetMemoryRequirements2Fn::name().as_ptr(),
        vk::KhrImagelessFramebufferFn::name().as_ptr(),
        vk::KhrImageFormatListFn::name().as_ptr(),
        vk::KhrDescriptorUpdateTemplateFn::name().as_ptr(),
        // Rust-GPU
        vk::KhrShaderFloat16Int8Fn::name().as_ptr(),
        // DLSS
        #[cfg(feature = "dlss")]
        {
            b"VK_NVX_binary_import\0".as_ptr() as *const i8
        },
        #[cfg(feature = "dlss")]
        {
            b"VK_KHR_push_descriptor\0".as_ptr() as *const i8
        },
        #[cfg(feature = "dlss")]
        vk::NvxImageViewHandleFn::name().as_ptr(),
    ];

    device_extensions_ptrs.push(ash::extensions::khr::Swapchain::name().as_ptr());

    let ray_tracing_extensions = [
        vk::KhrVulkanMemoryModelFn::name().as_ptr(), // used in ray tracing shaders
        vk::KhrPipelineLibraryFn::name().as_ptr(),   // rt dep
        vk::KhrDeferredHostOperationsFn::name().as_ptr(), // rt dep
        vk::KhrBufferDeviceAddressFn::name().as_ptr(), // rt dep
        vk::KhrAccelerationStructureFn::name().as_ptr(),
        vk::KhrRayTracingPipelineFn::name().as_ptr(),
    ];

    let ray_tracing_enabled = unsafe {
        ray_tracing_extensions.iter().all(|ext| {
            let ext = std::ffi::CStr::from_ptr(*ext).to_string_lossy();

            let supported = supported_extensions.contains(ext.as_ref());

            if !supported {
                dbg!("Ray tracing extension not supported: {}", ext);
            }

            supported
        })
    };

    if ray_tracing_enabled {
        dbg!("All ray tracing extensions are supported");
        device_extensions_ptrs.extend(ray_tracing_extensions.iter());
    }

    for ext in device_extensions {
        device_extensions_ptrs.push((*ext).as_ptr());
    }

    let device_features = vk::PhysicalDeviceFeatures::builder()
        .sampler_anisotropy(true)
        .shader_int64(true);

    let mut indexing_info = vk::PhysicalDeviceDescriptorIndexingFeatures::builder()
        .descriptor_binding_partially_bound(true)
        .runtime_descriptor_array(true)
        .build();
    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&device_extensions_ptrs)
        .enabled_features(&device_features)
        .push_next(&mut indexing_info);

    // Build device and queues
    let device = unsafe {
        instance
            .create_device(device, &device_create_info, None)
            .expect("Failed to create logical device.")
    };
    let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
    let present_queue = unsafe { device.get_device_queue(present_family_index, 0) };

    (device, graphics_queue, present_queue)
}

#[derive(Clone, Copy)]
pub struct QueueFamiliesIndices {
    pub graphics: u32,
    pub present: u32,
}

pub struct SharedContext {
    entry: Entry,
    instance: Instance,
    debug_utils_loader: DebugUtils,
    debug_call_back: vk::DebugUtilsMessengerEXT,
    device: Device,
    pdevice: vk::PhysicalDevice,
    allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
    pub queue_family_indices: QueueFamiliesIndices,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    pub acceleration_structure: khr::AccelerationStructure,
    pub ray_tracing: khr::RayTracingPipeline,
    pub ray_tracing_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
}

impl SharedContext {
    pub fn new(window: &mut Window, settings: &RendererSettings) -> Self {
        unsafe {
            let entry = Entry::load().unwrap();
            let app_name = CString::new("VulkanTriangle").unwrap();

            let mut layer_names = Vec::<CString>::new();
            if cfg!(debug_assertions) {
                layer_names.push(CString::new("VK_LAYER_KHRONOS_validation").unwrap());
                //layer_names.push(CString::new("VK_LAYER_LUNARG_api_dump").unwrap());
            }
            let layers_names_raw: Vec<*const i8> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let surface_extensions =
                ash_window::enumerate_required_extensions(window.handle()).unwrap();
            let mut extension_names_raw = surface_extensions
                .iter()
                .map(|ext| ext.as_ptr())
                .collect::<Vec<_>>();
            extension_names_raw.push(DebugUtils::name().as_ptr());

            for ext in &settings.extensions {
                extension_names_raw.push(ext.as_ptr());
            }

            let appinfo = vk::ApplicationInfo::builder()
                .application_name(&app_name)
                .application_version(0)
                .engine_name(&app_name)
                .engine_version(0)
                .api_version(vk::API_VERSION_1_2);

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&appinfo)
                .enabled_layer_names(&layers_names_raw)
                .enabled_extension_names(&extension_names_raw);

            let instance: Instance = entry
                .create_instance(&create_info, None)
                .expect("Instance creation error");

            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING, //| vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL)
                .pfn_user_callback(Some(vulkan_debug_callback));
            let debug_utils_loader = DebugUtils::new(&entry, &instance);
            let debug_call_back = debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap();

            window.create_surface(&entry, &instance);

            let pdevices = instance
                .enumerate_physical_devices()
                .expect("Physical device error");

            let pdevice = pdevices
                .iter()
                .map(|pdevice| {
                    instance
                        .get_physical_device_queue_family_properties(*pdevice)
                        .iter()
                        .enumerate()
                        .filter_map(|(index, ref info)| {
                            let supports_graphic_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && window.get_surface_support(*pdevice, index as u32);
                            if supports_graphic_and_surface {
                                Some(*pdevice)
                            } else {
                                None
                            }
                        })
                        .next()
                })
                .filter_map(|v| v)
                .next()
                .expect("Couldn't find suitable device.");

            //println!("{:?}", instance.get_physical_device_properties(pdevice));

            let (graphics, present) = find_queue_families(
                &instance,
                window.surface_loader(),
                window.surface(),
                pdevice,
            );
            let queue_family_indices = QueueFamiliesIndices {
                graphics: graphics.unwrap(),
                present: present.unwrap(),
            };
            let (device, graphics_queue, present_queue) = create_logical_device_with_graphics_queue(
                &instance,
                pdevice,
                queue_family_indices,
                &settings.device_extensions,
            );

            let allocator = Allocator::new(&AllocatorCreateDesc{
                instance: instance.clone(),
                device: device.clone(),
                physical_device: pdevice,
                debug_settings: Default::default(),
                buffer_device_address: true,  // TODO: check the BufferDeviceAddressFeatures struct.
            }).unwrap();

            let acceleration_structure = khr::AccelerationStructure::new(&instance, &device);
            let ray_tracing = khr::RayTracingPipeline::new(&instance, &device);
            let ray_tracing_properties = khr::RayTracingPipeline::get_properties(&instance, pdevice);

            SharedContext {
                entry,
                instance,
                debug_utils_loader,
                debug_call_back,
                device,
                pdevice,
                allocator: ManuallyDrop::new(Arc::new(Mutex::new(allocator))),
                queue_family_indices,
                graphics_queue,
                present_queue,
                acceleration_structure,
                ray_tracing,
                ray_tracing_properties,
            }
        }
    }

    pub fn entry(&self) -> &Entry {
        &self.entry
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.pdevice
    }

    pub fn get_physical_device_properties(&self) -> vk::PhysicalDeviceProperties {
        unsafe { self.instance.get_physical_device_properties(self.pdevice) }
    }

    pub fn get_physical_device_limits(&self) -> vk::PhysicalDeviceLimits {
        self.get_physical_device_properties().limits
    }

    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    pub fn present_queue(&self) -> vk::Queue {
        self.present_queue
    }

    pub fn allocator(&self) -> &Arc<Mutex<Allocator>> {
        &self.allocator
    }

    pub fn acceleration_structure(&self) -> &khr::AccelerationStructure {
        &self.acceleration_structure
    }

    pub fn ray_tracing(&self) -> &khr::RayTracingPipeline {
        &self.ray_tracing
    }

    pub unsafe fn ray_tracing_properties(&self) -> &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR {
        &self.ray_tracing_properties
    }

    pub fn queue_family_indices(&self) -> &QueueFamiliesIndices {
        &self.queue_family_indices
    }
}

impl Drop for SharedContext {
    fn drop(&mut self) {
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.allocator); // Explicitly drop before destruction of device and instance.
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_call_back, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub struct Context {
    shared_context: Arc<SharedContext>,
    frame_command_pools: Vec<CommandPool>,
    transient_command_pool: vk::CommandPool,
}

impl Context {
    pub fn new(shared_context: Arc<SharedContext>, swapchain_image_count: usize) -> Self {
        unsafe {
            let mut frame_command_pools = Vec::<CommandPool>::new();
            let graphics_index = shared_context.queue_family_indices.graphics;
            for _ in 0..swapchain_image_count {
                frame_command_pools.push(CommandPool::new(shared_context.clone(), graphics_index));
            }

            let pool_create_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                .queue_family_index(graphics_index);
            let transient_command_pool = shared_context
                .device()
                .create_command_pool(&pool_create_info, None)
                .unwrap();
            Context {
                shared_context,
                frame_command_pools,
                transient_command_pool,
            }
        }
    }

    pub fn entry(&self) -> &Entry {
        self.shared_context.entry()
    }

    pub fn instance(&self) -> &Instance {
        self.shared_context.instance()
    }

    pub fn device(&self) -> &Device {
        self.shared_context.device()
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.shared_context.physical_device()
    }

    pub fn get_physical_device_properties(&self) -> vk::PhysicalDeviceProperties {
        self.shared_context.get_physical_device_properties()
    }

    pub fn get_physical_device_limits(&self) -> vk::PhysicalDeviceLimits {
        self.shared_context.get_physical_device_limits()
    }

    pub fn present_queue(&self) -> vk::Queue {
        self.shared_context.present_queue()
    }

    pub fn graphics_queue(&self) -> vk::Queue {
        self.shared_context.graphics_queue()
    }

    pub fn allocator(&self) -> &Arc<Mutex<Allocator>> {
        self.shared_context.allocator()
    }

    pub fn acceleration_structure(&self) -> &khr::AccelerationStructure {
        self.shared_context.acceleration_structure()
    }

    pub fn ray_tracing(&self) -> &khr::RayTracingPipeline {
        self.shared_context.ray_tracing()
    }

    pub unsafe fn ray_tracing_properties(&self) -> &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR {
        self.shared_context.ray_tracing_properties()
    }

    pub fn shared(&self) -> &Arc<SharedContext> {
        &self.shared_context
    }

    pub fn begin_single_time_cmd(&self) -> vk::CommandBuffer {
        let create_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(self.transient_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);
        unsafe {
            let command_buffer = self
                .device()
                .allocate_command_buffers(&create_info)
                .unwrap()[0];
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device()
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();

            command_buffer
        }
    }

    pub fn end_single_time_cmd(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device().end_command_buffer(command_buffer).unwrap();

            let command_buffers = vec![command_buffer];
            let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);
            self.device()
                .queue_submit(
                    self.graphics_queue(),
                    &[submit_info.build()],
                    vk::Fence::null(),
                )
                .expect("queue submit failed.");

            self.device()
                .queue_wait_idle(self.graphics_queue())
                .unwrap();
            self.device()
                .free_command_buffers(self.transient_command_pool, &command_buffers)
        }
    }

    pub fn request_command_buffer(&self, frame_index: usize) -> vk::CommandBuffer {
        self.frame_command_pools[frame_index].reset();
        self.frame_command_pools[frame_index].request_command_buffer()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.device()
                .destroy_command_pool(self.transient_command_pool, None);
            self.frame_command_pools.clear();
        }
    }
}
