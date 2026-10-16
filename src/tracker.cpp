#include "tracker.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

// Image-processing helpers shared with the optimized feature extractor.
namespace FeatureExtractor
{
    Image gaussian_blur(const Image &img, float sigma);
    std::pair<Image, Image> image_gradients(const Image &img, float sigma);
}

using namespace FeatureExtractor;
using namespace Common;
namespace Tracker
{
    namespace
    {
        bool point_inside_border(const Image &img, float x, float y,
                                 int borderx, int bordery)
        {
            return x >= borderx && x <= img.width - 1 - borderx &&
                   y >= bordery && y <= img.height - 1 - bordery;
        }

        void validate_pyramids(const ImagePyramids &p1, const ImagePyramids &p2)
        {
            const auto level_count = p1.image.levels.size();
            if (level_count == 0 || p2.image.levels.size() != level_count ||
                p1.gradx.levels.size() != level_count ||
                p1.grady.levels.size() != level_count ||
                p2.gradx.levels.size() != level_count ||
                p2.grady.levels.size() != level_count ||
                p1.image.subsampling != p2.image.subsampling)
            {
                throw std::invalid_argument("Incompatible image pyramids");
            }

            for (std::size_t level = 0; level < level_count; ++level)
            {
                const Image &a = p1.image.levels[level];
                const Image &b = p2.image.levels[level];
                if (a.width != b.width || a.height != b.height ||
                    p1.gradx.levels[level].width != a.width ||
                    p1.gradx.levels[level].height != a.height ||
                    p1.grady.levels[level].width != a.width ||
                    p1.grady.levels[level].height != a.height ||
                    p2.gradx.levels[level].width != b.width ||
                    p2.gradx.levels[level].height != b.height ||
                    p2.grady.levels[level].width != b.width ||
                    p2.grady.levels[level].height != b.height)
                {
                    throw std::invalid_argument("Incompatible image pyramids");
                }
            }
        }

        std::vector<TrackResult> track_prepared(
            const ImagePyramids &p1,
            const ImagePyramids &p2,
            const std::vector<TrackResult> &positions,
            const Config &config);

        bool is_active(TrackStatus status)
        {
            return status == TrackStatus::Tracked ||
                   status == TrackStatus::Replaced;
        }

        void replace_lost_positions(
            const ImagePyramids &current,
            std::vector<TrackResult> &positions,
            const Config &config)
        {
            if (std::none_of(
                    positions.begin(), positions.end(),
                    [](const TrackResult &position)
                    { return !is_active(position.status); }))
            {
                return;
            }

            std::vector<Feature> features;
            features.reserve(positions.size());
            for (const TrackResult &position : positions)
            {
                if (is_active(position.status))
                {
                    features.push_back(
                        {static_cast<int>(position.x),
                         static_cast<int>(position.y),
                         0.0f});
                }
                else
                {
                    features.push_back({-1, -1, -1.0f});
                }
            }

            replace_lost_features(
                current.image.levels.front(),
                current.gradx.levels.front(),
                current.grady.levels.front(),
                features,
                config);

            for (std::size_t i = 0; i < positions.size(); ++i)
            {
                if (!is_active(positions[i].status) && features[i].score >= 0.0f)
                {
                    positions[i] = {
                        static_cast<float>(features[i].x),
                        static_cast<float>(features[i].y),
                        TrackStatus::Replaced};
                }
            }
        }
    }

    Image compute_smoothed_image(const Image &img, float sigma)
    {
        return FeatureExtractor::gaussian_blur(img, sigma);
    }
    Pyramid compute_pyramid(const Image &img, float sigma_fact,
                            int subsampling, int num_levels)
    {
        if (subsampling != 2 && subsampling != 4 && subsampling != 8 &&
            subsampling != 16 && subsampling != 32)
            throw std::invalid_argument("Unsupported pyramid subsampling");
        if (num_levels < 1)
            throw std::invalid_argument("Pyramid needs at least one level");
        Pyramid pyramid{subsampling, {img}};
        const int offset = subsampling / 2;
        for (int level = 1; level < num_levels; ++level)
        {
            const Image &previous = pyramid.levels.back();
            const int width = previous.width / subsampling;
            const int height = previous.height / subsampling;
            if (width < 1 || height < 1)
                throw std::invalid_argument("Too many pyramid levels for image size");
            Image smoothed = compute_smoothed_image(previous, subsampling * sigma_fact);
            Image next{width, height};
            next.data.resize(static_cast<std::size_t>(width) * height);
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    next.data[y * width + x] =
                        smoothed.data[(subsampling * y + offset) * previous.width +
                                      (subsampling * x + offset)];
            pyramid.levels.push_back(std::move(next));
        }
        return pyramid;
    }
    ImagePyramids compute_image_pyramids(const Image &img, const Config &config)
    {
        ImagePyramids result;
        Image smoothed = compute_smoothed_image(img, compute_smooth_sigma(config));
        result.image = compute_pyramid(
            smoothed, config.pyramid_sigma_fact,
            config.pyramid_subsampling, config.num_pyramid_levels);
        result.gradx.subsampling = result.image.subsampling;
        result.grady.subsampling = result.image.subsampling;
        for (const Image &level : result.image.levels)
        {
            auto [gx, gy] = FeatureExtractor::image_gradients(level, config.sigma);
            result.gradx.levels.push_back(std::move(gx));
            result.grady.levels.push_back(std::move(gy));
        }
        return result;
    }
    float interpolate(const Image &img, float x, float y)
    {
        assert(x >= 0.0f && y >= 0.0f);
        assert(x < img.width - 1 && y < img.height - 1);
        const int ix = static_cast<int>(x);
        const int iy = static_cast<int>(y);
        const float ax = x - ix;
        const float ay = y - iy;
        const float a = img.data[iy * img.width + ix];
        const float b = img.data[iy * img.width + ix + 1];
        const float c = img.data[(iy + 1) * img.width + ix];
        const float d = img.data[(iy + 1) * img.width + ix + 1];
        return (1 - ax) * (1 - ay) * a + ax * (1 - ay) * b +
               (1 - ax) * ay * c + ax * ay * d;
    }
    bool patch_fits(const Image &img, float x, float y,
                    int half_width, int half_height)
    {
        return x - half_width >= 0.0f && y - half_height >= 0.0f &&
               x + half_width < img.width - 1 &&
               y + half_height < img.height - 1;
    }

    TrackResult track_feature(float x1, float y1, float x2, float y2,
                              const Image &img1, const Image &gradx1,
                              const Image &grady1, const Image &img2,
                              const Image &gradx2, const Image &grady2,
                              const Config &config)
    {
        const int hw = config.window_width / 2;
        const int hh = config.window_height / 2;
        const int count = config.window_width * config.window_height;
        assert(config.window_width >= 3 && config.window_width % 2 == 1);
        assert(config.window_height >= 3 && config.window_height % 2 == 1);
        bool converged = false;
        for (int iteration = 0; iteration < config.max_iterations; ++iteration)
        {
            if (!patch_fits(img1, x1, y1, hw, hh) ||
                !patch_fits(img2, x2, y2, hw, hh))
                return {x2, y2, TrackStatus::OutOfBounds};
            float alpha = 1.0f;
            float beta = 0.0f;
            if (config.light_insensitive)
            {
                float sum1 = 0.0f, sum2 = 0.0f;
                float square1 = 0.0f, square2 = 0.0f;
                for (int j = -hh; j <= hh; ++j)
                    for (int i = -hw; i <= hw; ++i)
                    {
                        const float a = interpolate(img1, x1 + i, y1 + j);
                        const float b = interpolate(img2, x2 + i, y2 + j);
                        sum1 += a;
                        sum2 += b;
                        square1 += a * a;
                        square2 += b * b;
                    }
                if (square2 <= 0.0f)
                    return {x2, y2, TrackStatus::SmallDeterminant};
                alpha = std::sqrt(square1 / square2);
                beta = (sum1 - alpha * sum2) / count;
            }
            float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
            float ex = 0.0f, ey = 0.0f;
            for (int j = -hh; j <= hh; ++j)
                for (int i = -hw; i <= hw; ++i)
                {
                    const float ax = x1 + i, ay = y1 + j;
                    const float bx = x2 + i, by = y2 + j;
                    const float diff = alpha * interpolate(img2, bx, by) + beta -
                                       interpolate(img1, ax, ay);
                    const float gx = interpolate(gradx1, ax, ay) +
                                     alpha * interpolate(gradx2, bx, by);
                    const float gy = interpolate(grady1, ax, ay) +
                                     alpha * interpolate(grady2, bx, by);
                    gxx += gx * gx;
                    gxy += gx * gy;
                    gyy += gy * gy;
                    ex += diff * gx;
                    ey += diff * gy;
                }
            const float det = gxx * gyy - gxy * gxy;
            if (!std::isfinite(det) || det < config.min_determinant)
                return {x2, y2, TrackStatus::SmallDeterminant};
            ex *= config.step_factor;
            ey *= config.step_factor;
            const float dx = (gyy * ex - gxy * ey) / det;
            const float dy = (gxx * ey - gxy * ex) / det;
            x2 += dx;
            y2 += dy;
            if (!std::isfinite(x2) || !std::isfinite(y2))
                return {x2, y2, TrackStatus::OutOfBounds};
            if (std::abs(dx) < config.min_displacement &&
                std::abs(dy) < config.min_displacement)
            {
                converged = true;
                break;
            }
        }
        if (!patch_fits(img2, x2, y2, hw, hh))
            return {x2, y2, TrackStatus::OutOfBounds};
        if (!converged)
            return {x2, y2, TrackStatus::MaxIterations};
        float residue = 0.0f;
        float alpha = 1.0f, beta = 0.0f;
        if (config.light_insensitive)
        {
            float sum1 = 0.0f, sum2 = 0.0f;
            float square1 = 0.0f, square2 = 0.0f;
            for (int j = -hh; j <= hh; ++j)
                for (int i = -hw; i <= hw; ++i)
                {
                    const float a = interpolate(img1, x1 + i, y1 + j);
                    const float b = interpolate(img2, x2 + i, y2 + j);
                    sum1 += a;
                    sum2 += b;
                    square1 += a * a;
                    square2 += b * b;
                }
            if (square2 <= 0.0f)
                return {x2, y2, TrackStatus::LargeResidue};
            alpha = std::sqrt(square1 / square2);
            beta = (sum1 - alpha * sum2) / count;
        }
        for (int j = -hh; j <= hh; ++j)
            for (int i = -hw; i <= hw; ++i)
                residue += std::abs(interpolate(img1, x1 + i, y1 + j) -
                                    (alpha * interpolate(img2, x2 + i, y2 + j) + beta));
        if (residue / count > config.max_residue)
            return {x2, y2, TrackStatus::LargeResidue};
        return {x2, y2, TrackStatus::Tracked};
    }
    namespace
    {
        std::vector<TrackResult> track_prepared(
            const ImagePyramids &p1,
            const ImagePyramids &p2,
            const std::vector<TrackResult> &positions,
            const Config &config)
        {
            validate_pyramids(p1, p2);
            const int subsampling = p1.image.subsampling;
            const int last_level = static_cast<int>(p1.image.levels.size()) - 1;
            std::vector<TrackResult> results;
            results.reserve(positions.size());
            for (const TrackResult &position : positions)
            {
                if (!is_active(position.status))
                {
                    results.push_back(position);
                    continue;
                }

                float x1 = position.x;
                float y1 = position.y;
                for (int level = 0; level < last_level; ++level)
                {
                    x1 /= subsampling;
                    y1 /= subsampling;
                }
                float x2 = x1, y2 = y1;
                TrackResult tracked{x2, y2, TrackStatus::Tracked};
                for (int level = last_level; level >= 0; --level)
                {
                    if (level != last_level)
                    {
                        x1 *= subsampling;
                        y1 *= subsampling;
                        x2 *= subsampling;
                        y2 *= subsampling;
                    }
                    tracked = track_feature(
                        x1, y1, x2, y2,
                        p1.image.levels[level], p1.gradx.levels[level], p1.grady.levels[level],
                        p2.image.levels[level], p2.gradx.levels[level], p2.grady.levels[level],
                        config);
                    x2 = tracked.x;
                    y2 = tracked.y;
                    if (tracked.status == TrackStatus::OutOfBounds ||
                        tracked.status == TrackStatus::SmallDeterminant)
                        break;
                }
                if (!point_inside_border(
                        p2.image.levels.front(), tracked.x, tracked.y,
                        config.borderx, config.bordery))
                {
                    tracked.status = TrackStatus::OutOfBounds;
                }
                if (tracked.status != TrackStatus::Tracked)
                    tracked.x = tracked.y = -1.0f;
                results.push_back(tracked);
            }
            return results;
        }
    }

    std::vector<TrackResult> track_features(
        const Image &img1, const Image &img2,
        const std::vector<Feature> &features,
        const Config &config)
    {
        if (img1.width != img2.width || img1.height != img2.height)
            throw std::invalid_argument("Images must have equal dimensions");

        const auto p1 = compute_image_pyramids(img1, config);
        const auto p2 = compute_image_pyramids(img2, config);
        std::vector<TrackResult> positions;
        positions.reserve(features.size());
        for (const Feature &feature : features)
        {
            positions.push_back(
                {static_cast<float>(feature.x),
                 static_cast<float>(feature.y),
                 TrackStatus::Tracked});
        }
        return track_prepared(p1, p2, positions, config);
    }

    std::vector<Track> track(const std::vector<Image> &imgs, const Config &config)
    {
        if (imgs.empty())
            return {};

        const Image &first = imgs.front();
        for (const Image &img : imgs)
        {
            if (img.width != first.width || img.height != first.height)
                throw std::invalid_argument("Images must have equal dimensions");
        }

        const auto features = select_good_features(first, config);
        std::vector<TrackResult> positions;
        positions.reserve(features.size());

        std::vector<Track> tracks(features.size());
        for (std::size_t i = 0; i < features.size(); ++i)
        {
            const TrackResult initial{
                static_cast<float>(features[i].x),
                static_cast<float>(features[i].y),
                TrackStatus::Tracked};
            positions.push_back(initial);
            tracks[i].positions.reserve(imgs.size());
            tracks[i].positions.push_back(initial);
        }

        if (imgs.size() == 1 || tracks.empty())
            return tracks;

        auto previous = compute_image_pyramids(first, config);
        for (std::size_t frame = 1; frame < imgs.size(); ++frame)
        {
            auto current = compute_image_pyramids(imgs[frame], config);
            positions = track_prepared(previous, current, positions, config);
            replace_lost_positions(current, positions, config);

            for (std::size_t feature = 0; feature < tracks.size(); ++feature)
                tracks[feature].positions.push_back(positions[feature]);

            previous = std::move(current);
        }
        return tracks;
    }
}
