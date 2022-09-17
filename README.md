_! To prevent work policy violations, this project is now archived. !_

# sol-rs ☀

`sol-rs` is a small rendering toolkit for Vulkan, with a focus on real-time raytracing (which is not currently available via other APIs such as WebGPU). It hosts convenience wrappers but also exposes [ash](https://github.com/MaikKlein/ash) directly. Tested on Windows/NVIDIA, but the non-raytracing samples also work on Mac via MoltenVK.

However, this remains a personal sandbox for learning and experimentation so use at your own risk!

## Requirements
[LunarG Vulkan SDK](https://lunarg.com/vulkan-sdk/) installation.

On Windows, setting a `VULKAN_SDK` environment variable to the 1.3.204.1 SDK installation folder should prevent the need for building shaderc from source. Otherwise, additional installations such as [Ninja](https://ninja-build.org/) may be required. Please refer to the following [instructions](https://github.com/google/shaderc-rs/blob/master/README.md) if you run into shaderc-related issues.

## Examples

[![screenshot](https://i.imgur.com/kFc6nr3.png)](https://github.com/num3ric/sol-rs/blob/master/examples/5-pathtrace.rs)
`cargo run --release --example 5-pathtrace -- --model models/tunnel.gltf --sky`


[![screenshot](https://i.imgur.com/uW3Tm4e.png)](https://github.com/num3ric/sol-rs/blob/master/examples/5-pathtrace.rs)
`cargo run --release --example 5-pathtrace -- --model models/cornell.gltf`


[![screenshot](https://i.imgur.com/yC1x7EZ.png)](https://github.com/num3ric/sol-rs/blob/master/examples/4-ray-ao.rs)
`cargo run --release --example 4-ray-ao`


[![screenshot](https://i.imgur.com/R72zQ5N.png)](https://github.com/num3ric/sol-rs/blob/master/examples/1-cube.rs)
`cargo run --release --example 1-cube`


## References
* [Vulkan-Samples](https://github.com/KhronosGroup/Vulkan-Samples)
* [gltf-viewer-rs](https://github.com/adrien-ben/gltf-viewer-rs)
* [vkex](https://github.com/chaoticbob/vkex)
* [nvpro-samples](https://github.com/nvpro-samples)
* [metal-ray-tracer](https://sergeyreznik.github.io/metal-ray-tracer/index.html)
* [Google Dawn](https://dawn.googlesource.com/dawn/+/refs/heads/main)
* [Cinder](https://github.com/cinder/Cinder)
* [Nannou](https://github.com/nannou-org/nannou)
