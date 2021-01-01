use glam::*;
use winit::event::WindowEvent;

enum Modes {
    Examine,
    Fly,
    Walk,
    Trackball,
}
enum Actions {
    None,
    Orbit,
    Dolly,
    Pan,
    LookAround,
}
#[derive(Default, Debug, Clone, Copy)]
pub struct CameraInput {
    pub lmb: bool,
    pub mmb: bool,
    pub rmb: bool,
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
}

impl CameraInput {
    pub fn is_mouse_down(&self) -> bool {
        self.lmb || self.mmb || self.rmb
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Camera {
    input: CameraInput,
    position: Vec3,
    center: Vec3,
    up: Vec3,
    vfov: f32,
    z_near: f32,
    z_far: f32,
    view_matrix: Mat4,
    persp_matrix: Mat4,
    mouse_pos: Vec2,
    window_size: Vec2,
    speed: f32,
}

fn is_zero(value: f32) -> bool {
    value.abs() < f32::EPSILON
}

impl Camera {
    pub fn new(window_size: Vec2) -> Self {
        let mut camera = Camera {
            input: CameraInput::default(),
            position: Vec3::splat(10.0),
            center: Vec3::zero(),
            up: -Vec3::unit_y(),
            vfov: 35.0,
            z_near: 0.1,
            z_far: 1000.0,
            view_matrix: Mat4::identity(),
            persp_matrix: Mat4::identity(),
            mouse_pos: Vec2::zero(),
            window_size,
            speed: 30.0,
        };
        camera.update_persp();
        camera
    }

    pub fn from_view(view:glam::Mat4, yfov:f32, z_near:f32, z_far:f32) -> Self {
        let position = view * vec4(0.0,0.0,0.0,1.0);
        let camera = Camera {
            input: CameraInput::default(),
            position: position.into(),
            center: Vec3::zero(),
            up: -Vec3::unit_y(),
            vfov: yfov,
            z_near,
            z_far,
            view_matrix: view,
            persp_matrix: Mat4::identity(),
            mouse_pos: Vec2::zero(),
            window_size: vec2(1920.0, 1080.0),
            speed: 30.0,
        };
        camera
    }
}

impl Camera {
    fn update_view(&mut self) {
        self.view_matrix = Mat4::look_at_rh(self.position, self.center, self.up);
    }

    fn update_persp(&mut self) {
        let aspect = self.window_size.x / self.window_size.y;
        self.persp_matrix =
            Mat4::perspective_rh(self.vfov.to_radians(), aspect, self.z_near, self.z_far);
    }

    pub fn look_at(&mut self, eye: Vec3, center: Vec3, up: Vec3) {
        self.position = eye;
        self.center = center;
        self.up = up;
        self.update_view();
    }

    pub fn set_window_size(&mut self, window_size: Vec2) {
        self.window_size = window_size;
        self.update_persp();
    }

    pub fn set_mouse_pos(&mut self, x: f32, y: f32) {
        self.mouse_pos = vec2(x, y);
    }

    pub fn set_vfov(&mut self, vfov: f32) {
        self.vfov = vfov;
        self.update_persp();
    }

    pub fn mouse_move(&mut self, x: f32, y: f32, input: &CameraInput) -> bool {
        let mut moved = false;
        let mut action = Actions::None;
        if input.lmb {
            if ((input.ctrl) && (input.shift)) || input.alt {
                action = Actions::Orbit;
            } else if input.shift {
                action = Actions::Dolly;
            } else if input.ctrl {
                action = Actions::Pan;
            } else {
                action = Actions::LookAround;
            }
        } else if input.mmb {
            action = Actions::Pan;
        } else if input.rmb {
            action = Actions::Dolly;
        }

        let dx = (x - self.mouse_pos.x) / self.window_size.x;
        let dy = (y - self.mouse_pos.y) / self.window_size.y;
        match action {
            Actions::None => {}
            Actions::Orbit => {
                self.orbit(dx, -dy);
                moved = true;
            }
            Actions::Dolly => {
                self.dolly(0.0, dy);
                moved = true;
            }
            Actions::Pan => {
                self.pan(dx, -dy);
                moved = true;
            }
            Actions::LookAround => {}
        }
        if moved {
            self.update_view();
        }
        self.mouse_pos = vec2(x, y);

        moved
    }

    pub fn mouse_wheel(&mut self, value: i32) {
        let fval = value as f32;
        let dx = fval * fval.abs() / self.window_size.x;
        self.dolly(0.0, -dx * self.speed);
        self.update_view();
    }

    fn pan(&mut self, dx: f32, dy: f32) {
        let z = self.position - self.center;
        let length = z.length() / 0.785; // 45 degrees
        let z = z.normalize();
        let mut x = self.up.cross(z).normalize();
        let mut y = z.cross(x).normalize();

        x *= -dx * length;
        y *= dy * length;

        self.position += x + y;
        self.center += x + y;
    }

    fn orbit(&mut self, mut dx: f32, mut dy: f32) {
        if is_zero(dx) && is_zero(dy) {
            return;
        }

        // Full width will do a full turn
        dx *= std::f32::consts::TAU;
        dy *= std::f32::consts::TAU;

        // Get the camera
        let origin = self.center;
        let position = self.position;

        // Get the length of sight
        let mut center_to_eye = position - origin;
        let radius = center_to_eye.length();
        center_to_eye = center_to_eye.normalize();

        // Find the rotation around the UP axis (Y)
        let axe_z = center_to_eye;
        let rot_y = Mat4::from_axis_angle(self.up, -dx);

        // Apply the (Y) rotation to the eye-center vector
        let mut tmp = rot_y.mul_vec4(vec4(center_to_eye.x, center_to_eye.y, center_to_eye.z, 0.0));
        center_to_eye = vec3(tmp.x, tmp.y, tmp.z);

        // Find the rotation around the X vector: cross between eye-center and up (X)
        let axe_x = self.up.cross(axe_z).normalize();
        let rot_x = Mat4::from_axis_angle(axe_x, -dy);

        // Apply the (X) rotation to the eye-center vector
        tmp = rot_x.mul_vec4(vec4(center_to_eye.x, center_to_eye.y, center_to_eye.z, 0.0));
        let vect_rot = vec3(tmp.x, tmp.y, tmp.z);
        if vect_rot.x.signum() == center_to_eye.x.signum() {
            center_to_eye = vect_rot;
        }

        // Make the vector as long as it was originally
        center_to_eye *= radius;

        // Finding the new position
        self.position = origin + center_to_eye;
    }

    fn dolly(&mut self, dx: f32, dy: f32) {
        let mut z = self.center - self.position;
        let mut length = z.length();
        if is_zero(length) {
            return;
        }

        // Use the larger movement.
        let dd = if dx.abs() > dy.abs() { dx } else { -dy };
        let mut factor = self.speed * dd / length;

        // Adjust speed based on distance.
        length /= 10.0;
        length = length.max(0.001);
        factor *= length;

        // Don't move to or through the point of interest.
        if factor >= 1.0 {
            return;
        }

        z *= factor;
        self.position += z;
    }

    pub fn view_matrix(&self) -> Mat4 {
        self.view_matrix
    }

    pub fn perspective_matrix(&self) -> Mat4 {
        self.persp_matrix
    }
}

pub struct CameraManip {
    pub input: CameraInput,
    pub camera: Camera,
}

impl CameraManip {
    pub fn update(&mut self, window_event: &WindowEvent) -> bool {
        let mut moved = false;
        match window_event {
            WindowEvent::Resized(winit::dpi::PhysicalSize { width, height }) => {
                self.camera
                    .set_window_size(vec2(*width as f32, *height as f32));
            }
            WindowEvent::ModifiersChanged(m) => {
                self.input.alt = m.alt();
                self.input.ctrl = m.ctrl() || m.logo();
                self.input.shift = m.shift();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = vec2(position.x as f32, position.y as f32);
                if self.input.is_mouse_down() {
                    moved = self.camera.mouse_move(pos.x, pos.y, &self.input);
                } else {
                    self.camera.set_mouse_pos(pos.x, pos.y);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                match delta {
                    winit::event::MouseScrollDelta::PixelDelta(_) => {
                        //camera.mouse_wheel(d.x.max(d.y) as i32);
                    }
                    winit::event::MouseScrollDelta::LineDelta(_, y) => {
                        self.camera.mouse_wheel(*y as i32);
                        moved = true;
                    }
                };
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let is_down = match state {
                    winit::event::ElementState::Pressed => true,
                    winit::event::ElementState::Released => false,
                };
                match button {
                    winit::event::MouseButton::Left => {
                        self.input.lmb = is_down;
                    }
                    winit::event::MouseButton::Right => {
                        self.input.rmb = is_down;
                    }
                    winit::event::MouseButton::Middle => {
                        self.input.mmb = is_down;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        moved
    }
}
