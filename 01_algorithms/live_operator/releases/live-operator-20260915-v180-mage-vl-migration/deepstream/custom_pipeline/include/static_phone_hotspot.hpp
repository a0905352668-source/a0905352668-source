#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <utility>

namespace jiankong::custom_pipeline {

struct StaticHotspotCell {
    float heat = 0.0f;
    double last_confirmed_at = 0.0;
};

inline std::string static_hotspot_state_filename(
    const std::string& camera_name, std::size_t stream_index) {
    return camera_name + "__stream_" + std::to_string(stream_index) + ".json";
}

class CameraStaticHotspotMap {
public:
    CameraStaticHotspotMap(int frame_width, int frame_height)
        : frame_width_(std::max(0, frame_width)),
          frame_height_(std::max(0, frame_height)),
          cell_size_(frame_width_ > 0
                         ? std::max(kMinimumCellSizePx,
                                    kFrameWidthCellRatio * static_cast<float>(frame_width_))
                         : kMinimumCellSizePx) {}

    void confirm(float x, float y, double now) {
        const auto center = cell_for(x, y);
        if (!center.has_value() || !valid_time(now)) return;
        for_each_neighbour(*center, [&](const CellKey& key) {
            auto found = cells_.find(key);
            if (found != cells_.end() &&
                now - found->second.last_confirmed_at < kMinConfirmIntervalSeconds) {
                return;
            }
            auto& cell = found == cells_.end() ? cells_[key] : found->second;
            cell.heat = std::min(kMaximumHeat, effective_heat(cell, now) + kConfirmHeat);
            cell.last_confirmed_at = now;
        });
    }

    void penalize(float x, float y, double now) {
        const auto center = cell_for(x, y);
        if (!center.has_value() || !valid_time(now)) return;
        for_each_neighbour(*center, [&](const CellKey& key) {
            const auto found = cells_.find(key);
            if (found == cells_.end()) return;
            const float factor = decay_factor(found->second, now);
            const float penalized = std::min(
                kPenalizedMaximumHeat,
                std::max(0.0f, found->second.heat * factor - kPenaltyHeat));
            if (penalized <= 0.0f || factor <= 0.0f) {
                cells_.erase(found);
            } else {
                found->second.heat = std::min(kMaximumHeat, penalized / factor);
            }
        });
    }

    float score(float x, float y, double now) const {
        const auto key = cell_for(x, y);
        if (!key.has_value() || !valid_time(now)) return 0.0f;
        const auto found = cells_.find(*key);
        return found == cells_.end() ? 0.0f : effective_heat(found->second, now);
    }

    void decay(double now) {
        if (!valid_time(now)) return;
        for (auto it = cells_.begin(); it != cells_.end();) {
            if (effective_heat(it->second, now) <= 0.0f) {
                it = cells_.erase(it);
            } else {
                ++it;
            }
        }
    }

    const std::map<std::pair<int, int>, StaticHotspotCell>& cells() const {
        return cells_;
    }

    void restore(
        const std::map<std::pair<int, int>, StaticHotspotCell>& cells, double now) {
        cells_.clear();
        if (frame_width_ <= 0 || frame_height_ <= 0 || !valid_time(now)) return;
        for (const auto& [key, cell] : cells) {
            if (!valid_cell(key) || !std::isfinite(cell.heat) || cell.heat <= 0.0f ||
                !valid_time(cell.last_confirmed_at) ||
                cell.last_confirmed_at > now + kFutureTimestampToleranceSeconds ||
                now - cell.last_confirmed_at >= kExpirySeconds) {
                continue;
            }
            cells_[key] = StaticHotspotCell{
                std::min(cell.heat, kMaximumHeat), cell.last_confirmed_at};
        }
    }

private:
    using CellKey = std::pair<int, int>;

    static constexpr float kMinimumCellSizePx = 24.0f;
    static constexpr float kFrameWidthCellRatio = 0.012f;
    static constexpr float kConfirmHeat = 0.4f;
    static constexpr float kPenaltyHeat = 0.4f;
    static constexpr float kPenalizedMaximumHeat = 0.8f;
    static constexpr float kMaximumHeat = 4.0f;
    static constexpr double kMinConfirmIntervalSeconds = 3.0;
    static constexpr double kFutureTimestampToleranceSeconds = 1.0;
    static constexpr double kExpirySeconds = 24.0 * 60.0 * 60.0;

    static bool valid_time(double now) {
        return std::isfinite(now) && now >= 0.0;
    }

    bool valid_cell(const CellKey& key) const {
        if (key.first < 0 || key.second < 0) return false;
        const int columns = static_cast<int>(std::ceil(frame_width_ / cell_size_));
        const int rows = static_cast<int>(std::ceil(frame_height_ / cell_size_));
        return key.first < columns && key.second < rows;
    }

    std::optional<CellKey> cell_for(float x, float y) const {
        if (frame_width_ <= 0 || frame_height_ <= 0 || !std::isfinite(x) ||
            !std::isfinite(y) || x < 0.0f || y < 0.0f ||
            x >= static_cast<float>(frame_width_) || y >= static_cast<float>(frame_height_)) {
            return std::nullopt;
        }
        return CellKey{static_cast<int>(std::floor(x / cell_size_)),
                       static_cast<int>(std::floor(y / cell_size_))};
    }

    template <typename Function>
    void for_each_neighbour(const CellKey& center, Function&& function) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const CellKey key{center.first + dx, center.second + dy};
                if (valid_cell(key)) function(key);
            }
        }
    }

    static float effective_heat(const StaticHotspotCell& cell, double now) {
        if (!std::isfinite(cell.heat) || cell.heat <= 0.0f ||
            !valid_time(cell.last_confirmed_at) || !valid_time(now)) {
            return 0.0f;
        }
        return cell.heat * decay_factor(cell, now);
    }

    static float decay_factor(const StaticHotspotCell& cell, double now) {
        if (!valid_time(cell.last_confirmed_at) || !valid_time(now)) return 0.0f;
        const double age = std::max(0.0, now - cell.last_confirmed_at);
        if (age >= kExpirySeconds) return 0.0f;
        return static_cast<float>(1.0 - age / kExpirySeconds);
    }

    int frame_width_ = 0;
    int frame_height_ = 0;
    float cell_size_ = kMinimumCellSizePx;
    std::map<CellKey, StaticHotspotCell> cells_;
};

}  // namespace jiankong::custom_pipeline
