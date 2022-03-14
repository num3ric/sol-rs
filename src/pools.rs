use crate::{Resource, SharedContext};
use ash::{vk};
use std::cell::{Cell, RefCell};
use std::sync::Arc;

// Based on: https://github.com/KhronosGroup/Vulkan-Samples/blob/master/framework/semaphore_pool.h
pub struct SemaphorePool {
    shared_context: Arc<SharedContext>,
    semaphores: Vec<vk::Semaphore>,
    active_count: usize,
}

impl SemaphorePool {
    pub fn new(shared_context: Arc<SharedContext>) -> Self {
        SemaphorePool {
            shared_context,
            semaphores: Vec::new(),
            active_count: 0,
        }
    }

    pub fn request_semaphore(&mut self) -> vk::Semaphore {
        if self.active_count < self.semaphores.len() {
            let index = self.active_count;
            self.active_count = self.active_count + 1;
            return self.semaphores[index];
        } else {
            unsafe {
                let semaphore_create_info = vk::SemaphoreCreateInfo::default();
                let semaphore = self
                    .shared_context
                    .device()
                    .create_semaphore(&semaphore_create_info, None)
                    .unwrap();

                self.semaphores.push(semaphore.clone());
                return semaphore;
            }
        }
    }

    pub fn get_active_count(&self) -> usize {
        self.active_count
    }

    pub fn reset(&mut self) {
        self.active_count = 0;
    }
}

impl Drop for SemaphorePool {
    fn drop(&mut self) {
        self.reset();
        unsafe {
            self.semaphores.iter().for_each(|s| {
                self.shared_context.device().destroy_semaphore(*s, None);
            });
        }
    }
}

pub struct CommandPool {
    context: Arc<SharedContext>,
    pool: vk::CommandPool,
    command_buffers: RefCell<Vec<vk::CommandBuffer>>,
    active_count: Cell<usize>,
}

impl CommandPool {
    pub fn new(context: Arc<SharedContext>, queue_family_index: u32) -> Self {
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(queue_family_index);
        unsafe {
            let pool = context
                .device()
                .create_command_pool(&pool_create_info, None)
                .unwrap();
            CommandPool {
                context,
                pool,
                command_buffers: RefCell::new(Vec::new()),
                active_count: Cell::new(0),
            }
        }
    }

    pub fn reset(&self) {
        unsafe {
            self.context
                .device()
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::default())
                .expect("Reset command buffer failed.");

            self.active_count.set(0);
        }
    }

    pub fn request_command_buffer(&self) -> vk::CommandBuffer {
        let mut buffers = self.command_buffers.try_borrow_mut().unwrap();
        if self.active_count.get() < buffers.len() {
            let index = self.active_count.get();
            self.active_count.set(index + 1);
            return buffers[index];
        } else {
            unsafe {
                let create_info = vk::CommandBufferAllocateInfo::builder()
                    .command_buffer_count(1)
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY);
                let command_buffer = self
                    .context
                    .device()
                    .allocate_command_buffers(&create_info)
                    .unwrap()[0];

                buffers.push(command_buffer.clone());
                return command_buffer;
            }
        }
    }
}

impl Resource<vk::CommandPool> for CommandPool {
    fn handle(&self) -> vk::CommandPool {
        self.pool
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.command_buffers.get_mut().clear();
            self.context.device().destroy_command_pool(self.pool, None);
        }
    }
}
