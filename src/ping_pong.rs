//! GPU texture wrappers and double-buffered ("ping-pong") texture pairs.
//!
//! Why do fluid simulators need ping-pong textures?
//! In GPU compute shaders, hundreds of workgroups execute concurrently across the grid.
//! If a shader reads neighboring texels (e.g. during advection or Jacobi pressure iteration)
//! from the *same* texture it is currently writing to, a data race (read-after-write hazard)
//! occurs. WebGPU strictly forbids binding the same texture as both sampled input (`texture_2d`)
//! and storage output (`texture_storage_2d<..., write>`) in a single dispatch.
//!
//! [`PingPongTexture`] solves this by maintaining two identical textures (`read` and `write`).
//! Each compute pass reads from `read`, writes the updated field into `write`, and then calls
//! [`PingPongTexture::swap`] on the CPU (an $O(1)$ pointer swap) so `write` becomes `read`
//! for the next pass.

/// A single 2D GPU texture paired with its default [`wgpu::TextureView`].
pub struct GpuTexture {
    /// Underlying GPU texture allocation.
    pub texture: wgpu::Texture,
    /// View handle used when binding this texture to a [`wgpu::BindGroup`].
    pub view: wgpu::TextureView,
}

impl GpuTexture {
    /// Allocates a 2D GPU texture configured for both compute storage writes and shader sampling.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // TEXTURE_BINDING: allows reading via `textureLoad` or bilinear `textureSampleLevel`.
            // STORAGE_BINDING: allows writing via `textureStore` in compute shaders.
            // COPY_DST / COPY_SRC: allows clearing or uploading initial scenario data from the CPU.
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view }
    }
}

/// A pair of identical [`GpuTexture`]s (`read` and `write`) for hazard-free iterative compute passes.
pub struct PingPongTexture {
    /// Source texture bound as read-only (`texture_2d<f32>`) in the current compute pass.
    pub read: GpuTexture,
    /// Destination texture bound as write-only (`texture_storage_2d<..., write>`) in the current compute pass.
    pub write: GpuTexture,
    /// Grid width in texels.
    pub width: u32,
    /// Grid height in texels.
    pub height: u32,
}

impl PingPongTexture {
    /// Allocates two GPU textures (`{label}_a` and `{label}_b`) with identical dimensions and `format`.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> Self {
        let read = GpuTexture::new(device, width, height, format, &format!("{label}_a"));
        let write = GpuTexture::new(device, width, height, format, &format!("{label}_b"));
        Self {
            read,
            write,
            width,
            height,
        }
    }

    /// Swaps the `read` and `write` texture handles in $O(1)$ time on the CPU.
    ///
    /// Call this immediately after recording a compute pass that writes into `self.write`
    /// so subsequent passes read the newly computed values from `self.read`.
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.read, &mut self.write);
    }

    /// Zeroes out both the `read` and `write` textures on the GPU using `queue.write_texture`.
    ///
    /// Uses `write_texture` rather than `CommandEncoder::clear_texture` because `clear_texture`
    /// requires an optional native-only `CLEAR_TEXTURE` wgpu feature that is not supported on WebGPU.
    pub fn clear(&self, queue: &wgpu::Queue, bytes_per_pixel: u32) {
        let zero_data = vec![0u8; (self.width * self.height * bytes_per_pixel) as usize];
        for tex in [&self.read.texture, &self.write.texture] {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &zero_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.width * bytes_per_pixel),
                    rows_per_image: Some(self.height),
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}
