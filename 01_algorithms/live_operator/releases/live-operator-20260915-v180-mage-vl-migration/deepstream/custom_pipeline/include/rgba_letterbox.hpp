#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace jiankong::custom_pipeline {

struct LetterboxMeta {
    float scale = 1.0F;
    int pad_x = 0;
    int pad_y = 0;
    int source_width = 0;
    int source_height = 0;
};

LetterboxMeta compute_letterbox_meta(
    int source_width,
    int source_height,
    int output_size);

void rgba_pitch_letterbox_reference(
    const std::uint8_t* source_rgba,
    std::size_t source_pitch_bytes,
    int source_width,
    int source_height,
    int crop_x,
    int crop_y,
    int crop_width,
    int crop_height,
    int output_size,
    float* destination_chw);

}  // namespace jiankong::custom_pipeline
