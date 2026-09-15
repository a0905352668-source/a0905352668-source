#include "rgba_letterbox.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace jiankong::custom_pipeline {
namespace {

float bilinear_channel(
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
    int x0 = static_cast<int>(std::floor(source_x));
    int y0 = static_cast<int>(std::floor(source_y));
    const float tx = source_x - x0;
    const float ty = source_y - y0;
    x0 = std::clamp(x0, 0, crop_width - 1);
    y0 = std::clamp(y0, 0, crop_height - 1);
    const int x1 = std::clamp(x0 + 1, 0, crop_width - 1);
    const int y1 = std::clamp(y0 + 1, 0, crop_height - 1);
    const int gx0 = std::clamp(crop_x + x0, 0, source_width - 1);
    const int gx1 = std::clamp(crop_x + x1, 0, source_width - 1);
    const int gy0 = std::clamp(crop_y + y0, 0, source_height - 1);
    const int gy1 = std::clamp(crop_y + y1, 0, source_height - 1);
    const auto sample = [&](int x, int y) {
        return static_cast<float>(source[static_cast<std::size_t>(y) * pitch + x * 4 + channel]);
    };
    return ((1.0F - tx) * (1.0F - ty) * sample(gx0, gy0) +
            tx * (1.0F - ty) * sample(gx1, gy0) +
            (1.0F - tx) * ty * sample(gx0, gy1) +
            tx * ty * sample(gx1, gy1)) /
           255.0F;
}

}  // namespace

LetterboxMeta compute_letterbox_meta(
    int source_width,
    int source_height,
    int output_size) {
    if (source_width <= 0 || source_height <= 0 || output_size <= 0) {
        throw std::invalid_argument("letterbox dimensions must be positive");
    }
    const float scale = std::min(
        output_size / static_cast<float>(source_width),
        output_size / static_cast<float>(source_height));
    const int resized_width = std::max(1, static_cast<int>(std::round(source_width * scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(source_height * scale)));
    return LetterboxMeta{
        scale,
        (output_size - resized_width) / 2,
        (output_size - resized_height) / 2,
        source_width,
        source_height,
    };
}

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
    float* destination_chw) {
    if (source_rgba == nullptr || destination_chw == nullptr) {
        throw std::invalid_argument("letterbox buffers must not be null");
    }
    crop_x = std::clamp(crop_x, 0, source_width - 1);
    crop_y = std::clamp(crop_y, 0, source_height - 1);
    crop_width = std::clamp(crop_width, 1, source_width - crop_x);
    crop_height = std::clamp(crop_height, 1, source_height - crop_y);
    if (source_pitch_bytes < static_cast<std::size_t>(source_width) * 4) {
        throw std::invalid_argument("RGBA pitch is smaller than one row");
    }
    const LetterboxMeta meta = compute_letterbox_meta(crop_width, crop_height, output_size);
    const int resized_width = std::max(1, static_cast<int>(std::round(crop_width * meta.scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(crop_height * meta.scale)));
    const int area = output_size * output_size;

    for (int y = 0; y < output_size; ++y) {
        for (int x = 0; x < output_size; ++x) {
            const int offset = y * output_size + x;
            float r = 114.0F / 255.0F;
            float g = 114.0F / 255.0F;
            float b = 114.0F / 255.0F;
            if (x >= meta.pad_x && x < meta.pad_x + resized_width &&
                y >= meta.pad_y && y < meta.pad_y + resized_height) {
                const float source_x = (x - meta.pad_x + 0.5F) / meta.scale - 0.5F;
                const float source_y = (y - meta.pad_y + 0.5F) / meta.scale - 0.5F;
                r = bilinear_channel(source_rgba, source_pitch_bytes, source_width, source_height,
                                     crop_x, crop_y, crop_width, crop_height, source_x, source_y, 0);
                g = bilinear_channel(source_rgba, source_pitch_bytes, source_width, source_height,
                                     crop_x, crop_y, crop_width, crop_height, source_x, source_y, 1);
                b = bilinear_channel(source_rgba, source_pitch_bytes, source_width, source_height,
                                     crop_x, crop_y, crop_width, crop_height, source_x, source_y, 2);
            }
            destination_chw[offset] = r;
            destination_chw[area + offset] = g;
            destination_chw[2 * area + offset] = b;
        }
    }
}

}  // namespace jiankong::custom_pipeline
