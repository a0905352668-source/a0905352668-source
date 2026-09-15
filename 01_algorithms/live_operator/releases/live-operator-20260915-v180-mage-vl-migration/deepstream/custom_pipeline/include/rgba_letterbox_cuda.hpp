#pragma once

#include "rgba_letterbox.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace jiankong::custom_pipeline {

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
    cudaStream_t stream);

}  // namespace jiankong::custom_pipeline
