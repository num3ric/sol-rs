use crate::{offset_of, Vertex};
use std::path::PathBuf;
use ash::vk;
use glam::{vec2, vec4};

#[derive(Clone, Debug, Copy, Default)]
pub struct BasicVertex {
    pub pos: glam::Vec4,
    pub color: glam::Vec4,
    pub uv: glam::Vec2,
}

pub fn find_asset(filename: &str) -> Option<PathBuf> {
    let mut file_path = std::env::current_exe().unwrap();
    for _ in 0..5 {
        match file_path.parent() {
            None => return None,
            Some(parent) => {
                let assets_folder = parent.join("assets");
                if assets_folder.exists() {
                    let asset_path = assets_folder.join(filename);
                    if asset_path.exists() {
                        return Some(asset_path);
                    }
                }
                file_path = parent.to_path_buf()
            }
        }
    }
    None
}

impl Vertex for BasicVertex {
    fn stride() -> u32 {
        std::mem::size_of::<BasicVertex>() as u32
    }

    fn format_offset() -> Vec<(vk::Format, u32)> {
        vec![
            (
                vk::Format::R32G32B32A32_SFLOAT,
                offset_of!(BasicVertex, pos) as u32,
            ),
            (
                vk::Format::R32G32B32A32_SFLOAT,
                offset_of!(BasicVertex, color) as u32,
            ),
            (
                vk::Format::R32G32_SFLOAT,
                offset_of!(BasicVertex, uv) as u32,
            ),
        ]
    }
}

fn get_cube_face_colors() -> [glam::Vec4;6] {
    [
        glam::vec4(0.549, 0.702, 0.412, 1.0),
        glam::vec4(0.957, 0.886, 0.522, 1.0),
        glam::vec4(0.957, 0.635, 0.349, 1.0),
        glam::vec4(0.180, 0.251, 0.341, 1.0),
        glam::vec4(0.737, 0.294, 0.318, 1.0),
        glam::vec4(0.357, 0.557, 0.490, 1.0),
    ]
}


pub fn colored_cube_vertices() -> [BasicVertex; 36] {
    let colors = get_cube_face_colors();
    // green face
    [
        BasicVertex {
            pos: vec4(-1.0, -1.0, 1.0, 1.0),
            color: colors[0],
            uv: vec2(1.0, 1.0),
        },
        BasicVertex {
            pos: vec4(-1.0, 1.0, 1.0, 1.0),
            color: colors[0],
            uv: vec2(0.0, 1.0),
        },
        BasicVertex {
            pos: vec4(1.0, -1.0, 1.0, 1.0),
            color: colors[0],
            uv: vec2(1.0, 0.0),
        },
        BasicVertex {
            pos: vec4(1.0, -1.0, 1.0, 1.0),
            color: colors[0],
            uv: vec2(1.0, 0.0),
        },
        BasicVertex {
            pos: vec4(-1.0, 1.0, 1.0, 1.0),
            color: colors[0],
            uv: vec2(0.0, 1.0),
        },
        BasicVertex {
            pos: vec4(1.0, 1.0, 1.0, 1.0),
            color: colors[0],
            uv: vec2(0.0, 0.0),
        },
        // yellow face
        BasicVertex {
            pos: vec4(-1.0, -1.0, -1.0, 1.0),
            color: colors[1],
            uv: vec2(1.0, 1.0),
        },
        BasicVertex {
            pos: vec4(1.0, -1.0, -1.0, 1.0),
            color: colors[1],
            uv: vec2(0.0, 1.0),
        },
        BasicVertex {
            pos: vec4(-1.0, 1.0, -1.0, 1.0),
            color: colors[1],
            uv: vec2(1.0, 0.0),
        },
        BasicVertex {
            pos: vec4(-1.0, 1.0, -1.0, 1.0),
            color: colors[1],
            uv: vec2(1.0, 0.0),
        },
        BasicVertex {
            pos: vec4(1.0, -1.0, -1.0, 1.0),
            color: colors[1],
            uv: vec2(0.0, 1.0),
        },
        BasicVertex {
            pos: vec4(1.0, 1.0, -1.0, 1.0),
            color: colors[1],
            uv: vec2(0.0, 0.0),
        },
        // orange face
        BasicVertex {
            pos: vec4(-1.0, 1.0, 1.0, 1.0),
            color: colors[2],
            uv: vec2(1.0, 1.0),
        },
        BasicVertex {
            pos: vec4(-1.0, -1.0, 1.0, 1.0),
            color: colors[2],
            uv: vec2(0.0, 1.0),
        },
        BasicVertex {
            pos: vec4(-1.0, 1.0, -1.0, 1.0),
            color: colors[2],
            uv: vec2(1.0, 0.0),
        },
        BasicVertex {
            pos: vec4(-1.0, 1.0, -1.0, 1.0),
            color: colors[2],
            uv: vec2(1.0, 0.0),
        },
        BasicVertex {
            pos: vec4(-1.0, -1.0, 1.0, 1.0),
            color: colors[2],
            uv: vec2(0.0, 1.0),
        },
        BasicVertex {
            pos: vec4(-1.0, -1.0, -1.0, 1.0),
            color: colors[2],
            uv: vec2(0.0, 0.0),
        },
        // dark blue face
        BasicVertex {
            pos: vec4(1.0, 1.0, 1.0, 1.0),
            color: colors[3],
            uv: vec2(1.0, 1.0),
        },
        BasicVertex {
            pos: vec4(1.0, 1.0, -1.0, 1.0),
            color: colors[3],
            uv: vec2(0.0, 1.0),
        },
        BasicVertex {
            pos: vec4(1.0, -1.0, 1.0, 1.0),
            color: colors[3],
            uv: vec2(1.0, 0.0),
        },
        BasicVertex {
            pos: vec4(1.0, -1.0, 1.0, 1.0),
            color: colors[3],
            uv: vec2(1.0, 0.0),
        },
        BasicVertex {
            pos: vec4(1.0, 1.0, -1.0, 1.0),
            color: colors[3],
            uv: vec2(0.0, 1.0),
        },
        BasicVertex {
            pos: vec4(1.0, -1.0, -1.0, 1.0),
            color: colors[3],
            uv: vec2(0.0, 0.0),
        },
        // red face
        BasicVertex {
            pos: vec4(1.0, 1.0, 1.0, 1.0),
            color: colors[4],
            uv: vec2(1.0, 1.0),
        },
        BasicVertex {
            pos: vec4(-1.0, 1.0, 1.0, 1.0),
            color: colors[4],
            uv: vec2(0.0, 1.0),
        },
        BasicVertex {
            pos: vec4(1.0, 1.0, -1.0, 1.0),
            color: colors[4],
            uv: vec2(1.0, 0.0),
        },
        BasicVertex {
            pos: vec4(1.0, 1.0, -1.0, 1.0),
            color: colors[4],
            uv: vec2(1.0, 0.0),
        },
        BasicVertex {
            pos: vec4(-1.0, 1.0, 1.0, 1.0),
            color: colors[4],
            uv: vec2(0.0, 1.0),
        },
        BasicVertex {
            pos: vec4(-1.0, 1.0, -1.0, 1.0),
            color: colors[4],
            uv: vec2(0.0, 0.0),
        },
        // wintergreen face
        BasicVertex {
            pos: vec4(1.0, -1.0, 1.0, 1.0),
            color: colors[5],
            uv: vec2(1.0, 1.0),
        },
        BasicVertex {
            pos: vec4(1.0, -1.0, -1.0, 1.0),
            color: colors[5],
            uv: vec2(0.0, 1.0),
        },
        BasicVertex {
            pos: vec4(-1.0, -1.0, 1.0, 1.0),
            color: colors[5],
            uv: vec2(1.0, 0.0),
        },
        BasicVertex {
            pos: vec4(-1.0, -1.0, 1.0, 1.0),
            color: colors[5],
            uv: vec2(1.0, 0.0),
        },
        BasicVertex {
            pos: vec4(1.0, -1.0, -1.0, 1.0),
            color: colors[5],
            uv: vec2(0.0, 1.0),
        },
        BasicVertex {
            pos: vec4(-1.0, -1.0, -1.0, 1.0),
            color: colors[5],
            uv: vec2(0.0, 0.0),
        },
    ]
}
