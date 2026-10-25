#include "tracker.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace KLT
{
    namespace
    {
        struct CachedKernels
        {
            float sigma{std::numeric_limits<float>::quiet_NaN()};
            Kernels kernels;

            const Kernels &get(float requested_sigma)
            {
                if (sigma != requested_sigma)
                {
                    Kernels replacement = make_kernels(requested_sigma);
                    kernels = std::move(replacement);
                    sigma = requested_sigma;
                }
                return kernels;
            }
        };

        struct Scratch
        {
            Image filter_temporary;
            Image pyramid_smoothed;
            Image gradient_temporary;

            ImagePyramids pyramids[2];
            CachedKernels smoothing_kernels;
            CachedKernels pyramid_kernels;
            CachedKernels gradient_kernels;

            std::vector<float> reference_image;
            std::vector<float> reference_gradx;
            std::vector<float> reference_grady;
            std::vector<float> current_image;
            std::vector<Feature> replacement_features;
        };

        Scratch &scratch()
        {
            static thread_local Scratch value;
            return value;
        }

        inline float interpolate_unchecked(const Image &img, float x, float y)
        {
            const int ix = static_cast<int>(x);
            const int iy = static_cast<int>(y);
            const float ax = x - static_cast<float>(ix);
            const float ay = y - static_cast<float>(iy);
            const std::size_t top =
                static_cast<std::size_t>(iy) * img.width + ix;
            const std::size_t bottom = top + img.width;

            const float top_value =
                img.data[top] + ax * (img.data[top + 1] - img.data[top]);
            const float bottom_value =
                img.data[bottom] +
                ax * (img.data[bottom + 1] - img.data[bottom]);
            return top_value + ay * (bottom_value - top_value);
        }

        bool point_inside_border(
            const Image &img,
            float x,
            float y,
            int borderx,
            int bordery)
        {
            return x >= borderx && x <= img.width - 1 - borderx &&
                   y >= bordery && y <= img.height - 1 - bordery;
        }

        bool is_active(TrackStatus status)
        {
            return status == TrackStatus::Tracked ||
                   status == TrackStatus::Replaced;
        }

        void validate_pyramids(
            const ImagePyramids &first,
            const ImagePyramids &second)
        {
            const std::size_t level_count = first.image.levels.size();
            if (level_count == 0 ||
                second.image.levels.size() != level_count ||
                first.gradx.levels.size() != level_count ||
                first.grady.levels.size() != level_count ||
                second.gradx.levels.size() != level_count ||
                second.grady.levels.size() != level_count ||
                first.image.subsampling != second.image.subsampling)
            {
                throw std::invalid_argument("Incompatible image pyramids");
            }

            for (std::size_t level = 0; level < level_count; ++level)
            {
                const Image &a = first.image.levels[level];
                const Image &b = second.image.levels[level];
                if (a.width != b.width || a.height != b.height ||
                    first.gradx.levels[level].width != a.width ||
                    first.gradx.levels[level].height != a.height ||
                    first.grady.levels[level].width != a.width ||
                    first.grady.levels[level].height != a.height ||
                    second.gradx.levels[level].width != b.width ||
                    second.gradx.levels[level].height != b.height ||
                    second.grady.levels[level].width != b.width ||
                    second.grady.levels[level].height != b.height)
                {
                    throw std::invalid_argument("Incompatible image pyramids");
                }
            }
        }

        void validate_pyramid_settings(int subsampling, int num_levels)
        {
            if (subsampling != 2 && subsampling != 4 && subsampling != 8 &&
                subsampling != 16 && subsampling != 32)
            {
                throw std::invalid_argument("Unsupported pyramid subsampling");
            }
            if (num_levels < 1)
                throw std::invalid_argument("Pyramid needs at least one level");
        }

        void resize_grey_image(Image &img, int width, int height)
        {
            img.pixel_format = PixelFormat::GREY;
            resize_image(img, width, height);
        }

        void copy_image(const Image &source, Image &destination)
        {
            destination.pixel_format = source.pixel_format;
            resize_image(destination, source.width, source.height);
            std::copy(
                source.data.begin(), source.data.end(),
                destination.data.begin());
        }

        void build_remaining_pyramid_levels(
            Pyramid &pyramid,
            Scratch &workspace,
            const std::vector<float> &kernel)
        {
            const int subsampling = pyramid.subsampling;
            const int offset = subsampling / 2;

            for (std::size_t level = 1; level < pyramid.levels.size(); ++level)
            {
                const Image &previous = pyramid.levels[level - 1];
                const int width = previous.width / subsampling;
                const int height = previous.height / subsampling;
                if (width < 1 || height < 1)
                {
                    throw std::invalid_argument(
                        "Too many pyramid levels for image size");
                }

                detail::gaussian_blur_into(
                    previous,
                    workspace.filter_temporary,
                    workspace.pyramid_smoothed,
                    kernel);

                Image &next = pyramid.levels[level];
                resize_grey_image(next, width, height);
                const float *const smoothed =
                    workspace.pyramid_smoothed.data.data();
                float *const output = next.data.data();

                for (int y = 0; y < height; ++y)
                {
                    const std::size_t source_row =
                        static_cast<std::size_t>(subsampling * y + offset) *
                        previous.width;
                    const std::size_t output_row =
                        static_cast<std::size_t>(y) * width;
                    for (int x = 0; x < width; ++x)
                    {
                        output[output_row + x] =
                            smoothed[source_row + subsampling * x + offset];
                    }
                }
            }
        }

        void compute_pyramid_into(
            const Image &img,
            Pyramid &pyramid,
            float sigma_fact,
            int subsampling,
            int num_levels,
            Scratch &workspace)
        {
            validate_pyramid_settings(subsampling, num_levels);
            pyramid.subsampling = subsampling;
            pyramid.levels.resize(static_cast<std::size_t>(num_levels));
            copy_image(img, pyramid.levels.front());

            const auto &kernels = workspace.pyramid_kernels.get(
                static_cast<float>(subsampling) * sigma_fact);
            build_remaining_pyramid_levels(
                pyramid, workspace, kernels.gauss);
        }

        void compute_image_pyramids_into(
            const Image &img,
            ImagePyramids &result,
            const Config &config,
            Scratch &workspace)
        {
            validate_pyramid_settings(
                config.pyramid_subsampling,
                config.num_pyramid_levels);

            const std::size_t level_count =
                static_cast<std::size_t>(config.num_pyramid_levels);
            result.image.subsampling = config.pyramid_subsampling;
            result.gradx.subsampling = config.pyramid_subsampling;
            result.grady.subsampling = config.pyramid_subsampling;
            result.image.levels.resize(level_count);
            result.gradx.levels.resize(level_count);
            result.grady.levels.resize(level_count);

            const auto &smoothing = workspace.smoothing_kernels.get(
                compute_smooth_sigma(config));
            detail::gaussian_blur_into(
                img,
                workspace.filter_temporary,
                result.image.levels.front(),
                smoothing.gauss);

            const auto &pyramid = workspace.pyramid_kernels.get(
                static_cast<float>(config.pyramid_subsampling) *
                config.pyramid_sigma_fact);
            build_remaining_pyramid_levels(
                result.image, workspace, pyramid.gauss);

            const auto &gradients =
                workspace.gradient_kernels.get(config.sigma);
            for (std::size_t level = 0; level < level_count; ++level)
            {
                detail::image_gradients_into(
                    result.image.levels[level],
                    workspace.gradient_temporary,
                    result.gradx.levels[level],
                    result.grady.levels[level],
                    gradients);
            }
        }

        void cache_reference_patch(
            float x,
            float y,
            const Image &img,
            const Image &gradx,
            const Image &grady,
            int half_width,
            int half_height,
            Scratch &workspace)
        {
            const std::size_t count = static_cast<std::size_t>(
                (2 * half_width + 1) * (2 * half_height + 1));
            workspace.reference_image.resize(count);
            workspace.reference_gradx.resize(count);
            workspace.reference_grady.resize(count);

            std::size_t index = 0;
            for (int j = -half_height; j <= half_height; ++j)
            {
                for (int i = -half_width; i <= half_width; ++i)
                {
                    const float px = x + static_cast<float>(i);
                    const float py = y + static_cast<float>(j);
                    workspace.reference_image[index] =
                        interpolate_unchecked(img, px, py);
                    workspace.reference_gradx[index] =
                        interpolate_unchecked(gradx, px, py);
                    workspace.reference_grady[index] =
                        interpolate_unchecked(grady, px, py);
                    ++index;
                }
            }
        }

        void sample_image_patch(
            const Image &img,
            float x,
            float y,
            int half_width,
            int half_height,
            std::vector<float> &values,
            float &sum,
            float &square_sum)
        {
            const std::size_t count = static_cast<std::size_t>(
                (2 * half_width + 1) * (2 * half_height + 1));
            values.resize(count);
            sum = 0.0f;
            square_sum = 0.0f;

            std::size_t index = 0;
            for (int j = -half_height; j <= half_height; ++j)
            {
                for (int i = -half_width; i <= half_width; ++i)
                {
                    const float value = interpolate_unchecked(
                        img,
                        x + static_cast<float>(i),
                        y + static_cast<float>(j));
                    values[index++] = value;
                    sum += value;
                    square_sum += value * value;
                }
            }
        }

        TrackResult track_feature_impl(
            float x1,
            float y1,
            float x2,
            float y2,
            const Image &img1,
            const Image &gradx1,
            const Image &grady1,
            const Image &img2,
            const Image &gradx2,
            const Image &grady2,
            const Config &config,
            Scratch &workspace)
        {
            const int half_width = config.window_width / 2;
            const int half_height = config.window_height / 2;
            const int count = config.window_width * config.window_height;
            assert(config.window_width >= 3 &&
                   config.window_width % 2 == 1);
            assert(config.window_height >= 3 &&
                   config.window_height % 2 == 1);

            if (!patch_fits(img1, x1, y1, half_width, half_height) ||
                !patch_fits(img2, x2, y2, half_width, half_height))
            {
                return {x2, y2, TrackStatus::OutOfBounds};
            }

            cache_reference_patch(
                x1, y1, img1, gradx1, grady1,
                half_width, half_height, workspace);

            float reference_sum = 0.0f;
            float reference_square_sum = 0.0f;
            if (config.light_insensitive)
            {
                for (float value : workspace.reference_image)
                {
                    reference_sum += value;
                    reference_square_sum += value * value;
                }
            }

            bool converged = false;
            for (int iteration = 0;
                 iteration < config.max_iterations;
                 ++iteration)
            {
                if (!patch_fits(
                        img2, x2, y2, half_width, half_height))
                {
                    return {x2, y2, TrackStatus::OutOfBounds};
                }

                float alpha = 1.0f;
                float beta = 0.0f;
                if (config.light_insensitive)
                {
                    float current_sum;
                    float current_square_sum;
                    sample_image_patch(
                        img2, x2, y2, half_width, half_height,
                        workspace.current_image,
                        current_sum, current_square_sum);
                    if (current_square_sum <= 0.0f)
                    {
                        return {x2, y2, TrackStatus::SmallDeterminant};
                    }
                    alpha = std::sqrt(
                        reference_square_sum / current_square_sum);
                    beta = (reference_sum - alpha * current_sum) / count;
                }

                float gxx = 0.0f;
                float gxy = 0.0f;
                float gyy = 0.0f;
                float ex = 0.0f;
                float ey = 0.0f;
                std::size_t index = 0;
                for (int j = -half_height; j <= half_height; ++j)
                {
                    for (int i = -half_width; i <= half_width; ++i)
                    {
                        const float bx = x2 + static_cast<float>(i);
                        const float by = y2 + static_cast<float>(j);
                        const float current = config.light_insensitive
                                                  ? workspace.current_image[index]
                                                  : interpolate_unchecked(
                                                        img2, bx, by);
                        const float difference =
                            alpha * current + beta -
                            workspace.reference_image[index];
                        const float gx =
                            workspace.reference_gradx[index] +
                            alpha * interpolate_unchecked(gradx2, bx, by);
                        const float gy =
                            workspace.reference_grady[index] +
                            alpha * interpolate_unchecked(grady2, bx, by);

                        gxx += gx * gx;
                        gxy += gx * gy;
                        gyy += gy * gy;
                        ex += difference * gx;
                        ey += difference * gy;
                        ++index;
                    }
                }

                const float determinant = gxx * gyy - gxy * gxy;
                if (!std::isfinite(determinant) ||
                    determinant < config.min_determinant)
                {
                    return {x2, y2, TrackStatus::SmallDeterminant};
                }

                ex *= config.step_factor;
                ey *= config.step_factor;
                const float dx =
                    (gyy * ex - gxy * ey) / determinant;
                const float dy =
                    (gxx * ey - gxy * ex) / determinant;
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

            if (!patch_fits(img2, x2, y2, half_width, half_height))
                return {x2, y2, TrackStatus::OutOfBounds};
            if (!converged)
                return {x2, y2, TrackStatus::MaxIterations};

            float alpha = 1.0f;
            float beta = 0.0f;
            if (config.light_insensitive)
            {
                float current_sum;
                float current_square_sum;
                sample_image_patch(
                    img2, x2, y2, half_width, half_height,
                    workspace.current_image,
                    current_sum, current_square_sum);
                if (current_square_sum <= 0.0f)
                    return {x2, y2, TrackStatus::LargeResidue};
                alpha = std::sqrt(
                    reference_square_sum / current_square_sum);
                beta = (reference_sum - alpha * current_sum) / count;
            }

            float residue = 0.0f;
            std::size_t index = 0;
            for (int j = -half_height; j <= half_height; ++j)
            {
                for (int i = -half_width; i <= half_width; ++i)
                {
                    const float current = config.light_insensitive
                                              ? workspace.current_image[index]
                                              : interpolate_unchecked(
                                                    img2,
                                                    x2 + static_cast<float>(i),
                                                    y2 + static_cast<float>(j));
                    residue += std::abs(
                        workspace.reference_image[index] -
                        (alpha * current + beta));
                    ++index;
                }
            }

            if (residue / count > config.max_residue)
                return {x2, y2, TrackStatus::LargeResidue};
            return {x2, y2, TrackStatus::Tracked};
        }

        void track_prepared_into(
            const ImagePyramids &first,
            const ImagePyramids &second,
            const std::vector<TrackResult> &positions,
            std::vector<TrackResult> &results,
            const Config &config,
            Scratch &workspace)
        {
            validate_pyramids(first, second);
            const int subsampling = first.image.subsampling;
            const int last_level =
                static_cast<int>(first.image.levels.size()) - 1;

            int coarsest_scale = 1;
            for (int level = 0; level < last_level; ++level)
                coarsest_scale *= subsampling;

            results.resize(positions.size());
            for (std::size_t feature = 0;
                 feature < positions.size();
                 ++feature)
            {
                const TrackResult &position = positions[feature];
                if (!is_active(position.status))
                {
                    results[feature] = position;
                    continue;
                }

                float x1 = position.x / coarsest_scale;
                float y1 = position.y / coarsest_scale;
                float x2 = x1;
                float y2 = y1;
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

                    tracked = track_feature_impl(
                        x1, y1, x2, y2,
                        first.image.levels[level],
                        first.gradx.levels[level],
                        first.grady.levels[level],
                        second.image.levels[level],
                        second.gradx.levels[level],
                        second.grady.levels[level],
                        config,
                        workspace);
                    x2 = tracked.x;
                    y2 = tracked.y;

                    if (tracked.status == TrackStatus::OutOfBounds ||
                        tracked.status == TrackStatus::SmallDeterminant)
                    {
                        break;
                    }
                }

                if (!point_inside_border(
                        second.image.levels.front(),
                        tracked.x,
                        tracked.y,
                        config.borderx,
                        config.bordery))
                {
                    tracked.status = TrackStatus::OutOfBounds;
                }
                if (tracked.status != TrackStatus::Tracked)
                    tracked.x = tracked.y = -1.0f;
                results[feature] = tracked;
            }
        }

        void replace_lost_positions(
            const ImagePyramids &current,
            std::vector<TrackResult> &positions,
            const Config &config,
            Scratch &workspace)
        {
            if (std::none_of(
                    positions.begin(), positions.end(),
                    [](const TrackResult &position)
                    { return !is_active(position.status); }))
            {
                return;
            }

            auto &features = workspace.replacement_features;
            features.clear();
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
                if (!is_active(positions[i].status) &&
                    features[i].score >= 0.0f)
                {
                    positions[i] = {
                        static_cast<float>(features[i].x),
                        static_cast<float>(features[i].y),
                        TrackStatus::Replaced};
                }
            }
        }
    }

    Pyramid compute_pyramid(
        const Image &img,
        float sigma_fact,
        int subsampling,
        int num_levels)
    {
        Pyramid result;
        compute_pyramid_into(
            img, result, sigma_fact, subsampling, num_levels, scratch());
        return result;
    }

    ImagePyramids compute_image_pyramids(
        const Image &img,
        const Config &config)
    {
        ImagePyramids result;
        compute_image_pyramids_into(img, result, config, scratch());
        return result;
    }

    float interpolate(const Image &img, float x, float y)
    {
        assert(x >= 0.0f && y >= 0.0f);
        assert(x < img.width - 1 && y < img.height - 1);
        return interpolate_unchecked(img, x, y);
    }

    bool patch_fits(
        const Image &img,
        float x,
        float y,
        int half_width,
        int half_height)
    {
        return x - half_width >= 0.0f &&
               y - half_height >= 0.0f &&
               x + half_width < img.width - 1 &&
               y + half_height < img.height - 1;
    }

    TrackResult track_feature(
        float x1,
        float y1,
        float x2,
        float y2,
        const Image &img1,
        const Image &gradx1,
        const Image &grady1,
        const Image &img2,
        const Image &gradx2,
        const Image &grady2,
        const Config &config)
    {
        return track_feature_impl(
            x1, y1, x2, y2,
            img1, gradx1, grady1,
            img2, gradx2, grady2,
            config, scratch());
    }

    std::vector<TrackResult> track_features(
        const Image &img1,
        const Image &img2,
        const std::vector<Feature> &features,
        const Config &config)
    {
        if (img1.width != img2.width || img1.height != img2.height)
            throw std::invalid_argument("Images must have equal dimensions");

        Scratch &workspace = scratch();
        compute_image_pyramids_into(
            img1, workspace.pyramids[0], config, workspace);
        compute_image_pyramids_into(
            img2, workspace.pyramids[1], config, workspace);

        std::vector<TrackResult> positions;
        positions.reserve(features.size());
        for (const Feature &feature : features)
        {
            positions.push_back(
                {static_cast<float>(feature.x),
                 static_cast<float>(feature.y),
                 TrackStatus::Tracked});
        }

        std::vector<TrackResult> results;
        track_prepared_into(
            workspace.pyramids[0],
            workspace.pyramids[1],
            positions,
            results,
            config,
            workspace);
        return results;
    }

    std::vector<Track> track(
        const std::vector<Image> &imgs,
        const Config &config)
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

        Scratch &workspace = scratch();
        ImagePyramids *previous = &workspace.pyramids[0];
        ImagePyramids *current = &workspace.pyramids[1];
        compute_image_pyramids_into(first, *previous, config, workspace);

        std::vector<TrackResult> next_positions;
        next_positions.reserve(positions.size());
        for (std::size_t frame = 1; frame < imgs.size(); ++frame)
        {
            compute_image_pyramids_into(
                imgs[frame], *current, config, workspace);
            track_prepared_into(
                *previous,
                *current,
                positions,
                next_positions,
                config,
                workspace);
            positions.swap(next_positions);
            replace_lost_positions(
                *current, positions, config, workspace);

            for (std::size_t feature = 0;
                 feature < tracks.size();
                 ++feature)
            {
                tracks[feature].positions.push_back(positions[feature]);
            }

            std::swap(previous, current);
        }
        return tracks;
    }
}
