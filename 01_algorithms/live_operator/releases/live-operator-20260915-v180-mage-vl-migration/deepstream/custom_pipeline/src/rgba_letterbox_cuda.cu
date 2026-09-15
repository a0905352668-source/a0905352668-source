#include "rgba_letterbox_cuda.hpp"

#include <algorithm>
#include <stdexcept>

namespace jiankong::custom_pipeline {
namespace {

__device__ float interpolate(
    const std::uint8_t* source,
    std::size_t pitch,
    int source_width,
    int source_height,
    int crop_x,
    int crop_y,
    int crop_width,
    int crop_height,
    float source_x,
    float source_y,
    int channel) {
    int x0 = static_cast<int>(floorf(source_x));
    int y0 = static_cast<int>(floorf(source_y));
    const float tx = source_x - x0;
    const float ty = source_y - y0;
    x0 = max(0, min(crop_width - 1, x0));
    y0 = max(0, min(crop_height - 1, y0));
    const int x1 = max(0, min(crop_width - 1, x0 + 1));
    const int y1 = max(0, min(crop_height - 1, y0 + 1));
    const int gx0 = max(0, min(source_width - 1, crop_x + x0));
    const int gx1 = max(0, min(source_width - 1, crop_x + x1));
    const int gy0 = max(0, min(source_height - 1, crop_y + y0));
    const int gy1 = max(0, min(source_height - 1, crop_y + y1));
    const float v00 = static_cast<float>(
        source[static_cast<std::size_t>(gy0) * pitch + gx0 * 4 + channel]);
    const float v10 = static_cast<float>(
        source[static_cast<std::size_t>(gy0) * pitch + gx1 * 4 + channel]);
    const float v01 = static_cast<float>(
        source[static_cast<std::size_t>(gy1) * pitch + gx0 * 4 + channel]);
    const float v11 = static_cast<float>(
        source[static_cast<std::size_t>(gy1) * pitch + gx1 * 4 + channel]);
    return ((1.0F - tx) * (1.0F - ty) * v00 +
            tx * (1.0F - ty) * v10 +
            (1.0F - tx) * ty * v01 +
            tx * ty * v11) /
           255.0F;
}

__global__ void rgba_pitch_letterbox_kernel(
    const std::uint8_t* source,
    std::size_t pitch,
    int source_width,
    int source_height,
    int crop_x,
    int crop_y,
    int crop_width,
    int crop_height,
    float scale,
    int pad_x,
    int pad_y,
    int output_size,
    float* destination,
    int batch_index) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output_size || y >= output_size) return;
    const int resized_width = max(1, __float2int_rn(crop_width * scale));
    const int resized_height = max(1, __float2int_rn(crop_height * scale));
    const int area = output_size * output_size;
    float* output = destination + static_cast<std::size_t>(batch_index) * 3 * area;
    const int offset = y * output_size + x;
    float r = 114.0F / 255.0F;
    float g = 114.0F / 255.0F;
    float b = 114.0F / 255.0F;
    if (x >= pad_x && x < pad_x + resized_width &&
        y >= pad_y && y < pad_y + resized_height) {
        const float source_x = (x - pad_x + 0.5F) / scale - 0.5F;
        const float source_y = (y - pad_y + 0.5F) / scale - 0.5F;
        r = interpolate(source, pitch, source_width, source_height, crop_x, crop_y,
                        crop_width, crop_height, source_x, source_y, 0);
        g = interpolate(source, pitch, source_width, source_height, crop_x, crop_y,
                        crop_width, crop_height, source_x, source_y, 1);
        b = interpolate(source, pitch, source_width, source_height, crop_x, crop_y,
                        crop_width, crop_height, source_x, source_y, 2);
    }
    output[offset] = r;
    output[area + offset] = g;
    output[2 * area + offset] = b;
}

}  // namespace

LetterboxMeta launch_rgba_pitch_letterbox(
    const std::uint8_t* source_rgba,
    std::size_t source_pitch_bytes,
    int source_width,
    int source_height,
    int crop_x,
    int crop_y,
    int crop_width,
    int crop_height,
    int output_size,
    float* destination_chw,
    int batch_index,
    cudaStream_t stream) {
    if (source_rgba == nullptr || destination_chw == nullptr) {
        throw std::invalid_argument("CUDA letterbox buffers must not be null");
    }
    crop_x = std::clamp(crop_x, 0, source_width - 1);
    crop_y = std::clamp(crop_y, 0, source_height - 1);
    crop_width = std::clamp(crop_width, 1, source_width - crop_x);
    crop_height = std::clamp(crop_height, 1, source_height - crop_y);
    const LetterboxMeta meta = compute_letterbox_meta(crop_width, crop_height, output_size);
    const dim3 block(16, 16);
    const dim3 grid((output_size + block.x - 1) / block.x,
                    (output_size + block.y - 1) / block.y);
    rgba_pitch_letterbox_kernel<<<grid, block, 0, stream>>>(
        source_rgba, source_pitch_bytes, source_width, source_height,
        crop_x, crop_y, crop_width, crop_height, meta.scale, meta.pad_x, meta.pad_y,
        output_size, destination_chw, batch_index);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("RGBA letterbox kernel launch failed: ") +
                                 cudaGetErrorString(status));
    }
    return meta;
}

}  // namespace jiankong::custom_pipeline
