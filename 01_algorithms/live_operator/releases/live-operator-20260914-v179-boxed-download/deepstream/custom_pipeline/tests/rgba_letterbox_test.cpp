#include "rgba_letterbox.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_near(float actual, float expected, const char* message) {
    require(std::fabs(actual - expected) < 1.0e-5F, message);
}

void test_pitch_and_channel_order() {
    constexpr int width = 2;
    constexpr int height = 2;
    constexpr std::size_t pitch = 12;
    std::vector<std::uint8_t> rgba(pitch * height, 0xEE);
    const std::uint8_t pixels[2][2][4] = {
        {{255, 0, 0, 255}, {0, 255, 0, 255}},
        {{0, 0, 255, 255}, {255, 255, 255, 255}},
    };
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 4; ++c) {
                rgba[static_cast<std::size_t>(y) * pitch + x * 4 + c] = pixels[y][x][c];
            }
        }
    }
    std::vector<float> output(3 * width * height);
    jiankong::custom_pipeline::rgba_pitch_letterbox_reference(
        rgba.data(), pitch, width, height, 0, 0, width, height, width, output.data());
    require_near(output[0], 1.0F, "red channel order mismatch");
    require_near(output[width * height + 1], 1.0F, "green channel order mismatch");
    require_near(output[2 * width * height + 2], 1.0F, "blue channel order mismatch");
}

void test_letterbox_padding() {
    constexpr int width = 4;
    constexpr int height = 2;
    std::vector<std::uint8_t> rgba(width * height * 4, 255);
    std::vector<float> output(3 * width * width, 0.0F);
    const auto meta = jiankong::custom_pipeline::compute_letterbox_meta(width, height, width);
    require(meta.pad_y == 1, "expected symmetric vertical padding");
    jiankong::custom_pipeline::rgba_pitch_letterbox_reference(
        rgba.data(), width * 4, width, height, 0, 0, width, height, width, output.data());
    require_near(output[0], 114.0F / 255.0F, "top padding mismatch");
    require_near(output[width], 1.0F, "image body mismatch");
}

void test_crop_uses_global_pitch_coordinates() {
    constexpr int width = 4;
    constexpr int height = 2;
    std::vector<std::uint8_t> rgba(width * height * 4, 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 2; x < width; ++x) {
            rgba[(y * width + x) * 4] = 255;
        }
    }
    std::vector<float> output(3 * 2 * 2, 0.0F);
    jiankong::custom_pipeline::rgba_pitch_letterbox_reference(
        rgba.data(), width * 4, width, height, 2, 0, 2, 2, 2, output.data());
    for (int i = 0; i < 4; ++i) require_near(output[i], 1.0F, "crop offset mismatch");
}

}  // namespace

int main() {
    try {
        test_pitch_and_channel_order();
        test_letterbox_padding();
        test_crop_uses_global_pitch_coordinates();
        std::cout << "rgba_letterbox_test: PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "rgba_letterbox_test: FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
