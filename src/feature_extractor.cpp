#include "feature_extractor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <vector>

namespace KLT
{
    namespace
    {
        struct Scratch
        {
            Image blur_tmp{0, 0};
            Image smoothed{0, 0};
            Image tmp{0, 0};
            Image gradx{0, 0};
            Image grady{0, 0};
            std::vector<float> col_xx, col_xy, col_yy;
            std::vector<Feature> candidates;
            std::vector<std::uint8_t> occupied;
        };

        // Per-thread reusable scratch buffers, without global shared mutable state.
        Scratch &scratch()
        {
            static thread_local Scratch s;
            return s;
        }

        void compute_gradients(const Image &img, Scratch &s, const Kernels &k)
        {
            detail::image_gradients_into(
                img, s.tmp, s.gradx, s.grady, k);
        }

        // An exact sliding box sum of gradient products, apart from floating-point
        // addition order. Complexity: O(width * height), not O(pixels * window area).
        void find_candidates(
            const Image &img,
            const Image &gradx,
            const Image &grady,
            const Config &cfg,
            Scratch &s)
        {
            const int w = img.width;
            const int h = img.height;
            const int hw = cfg.window_width / 2;
            const int hh = cfg.window_height / 2;
            const int borderx = std::max(cfg.borderx, hw);
            const int bordery = std::max(cfg.bordery, hh);
            const float threshold = std::max(1.0f, static_cast<float>(cfg.min_eigenvalue));

            s.candidates.clear();
            if (cfg.window_width > w || cfg.window_height > h ||
                borderx >= w - borderx || bordery >= h - bordery)
                return;

            s.col_xx.assign(static_cast<std::size_t>(w), 0.0f);
            s.col_xy.assign(static_cast<std::size_t>(w), 0.0f);
            s.col_yy.assign(static_cast<std::size_t>(w), 0.0f);

            const float *const gx = gradx.data.data();
            const float *const gy = grady.data.data();

            // Initial vertical sums for the first row inside the valid border.
            for (int yy = bordery - hh; yy <= bordery + hh; ++yy)
            {
                const std::size_t base = static_cast<std::size_t>(yy) * w;
                for (int xx = 0; xx < w; ++xx)
                {
                    const float a = gx[base + xx];
                    const float b = gy[base + xx];
                    s.col_xx[xx] += a * a;
                    s.col_xy[xx] += a * b;
                    s.col_yy[xx] += b * b;
                }
            }

            for (int y = bordery; y < h - bordery; ++y)
            {
                if (y != bordery)
                {
                    const std::size_t old_base = static_cast<std::size_t>(y - hh - 1) * w;
                    const std::size_t new_base = static_cast<std::size_t>(y + hh) * w;
                    for (int xx = 0; xx < w; ++xx)
                    {
                        const float ax = gx[old_base + xx], ay = gy[old_base + xx];
                        const float bx = gx[new_base + xx], by = gy[new_base + xx];
                        s.col_xx[xx] += bx * bx - ax * ax;
                        s.col_xy[xx] += bx * by - ax * ay;
                        s.col_yy[xx] += by * by - ay * ay;
                    }
                }

                float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
                for (int xx = borderx - hw; xx <= borderx + hw; ++xx)
                {
                    gxx += s.col_xx[xx];
                    gxy += s.col_xy[xx];
                    gyy += s.col_yy[xx];
                }

                for (int x = borderx; x < w - borderx; ++x)
                {
                    if (x != borderx)
                    {
                        const int leave = x - hw - 1;
                        const int enter = x + hw;
                        gxx += s.col_xx[enter] - s.col_xx[leave];
                        gxy += s.col_xy[enter] - s.col_xy[leave];
                        gyy += s.col_yy[enter] - s.col_yy[leave];
                    }

                    // For PSD Z, lambda_min(Z) >= t iff (Z-tI) is PSD.
                    // This rejects flat regions and 1-D edges without a sqrt().
                    if (gxx < threshold || gyy < threshold)
                        continue;
                    const float xx = gxx - threshold;
                    const float yy = gyy - threshold;
                    if (xx * yy < gxy * gxy)
                        continue;

                    const float diff = gxx - gyy;
                    const float score = 0.5f *
                                        (gxx + gyy - std::sqrt(diff * diff + 4.0f * gxy * gxy));
                    if (score >= threshold)
                        s.candidates.push_back({x, y, score});
                }
            }
        }

        void mark_occupied(
            std::vector<std::uint8_t> &occupied,
            int width,
            int height,
            int x,
            int y,
            int radius)
        {
            if (radius < 0)
                return;

            const int xmin = std::max(0, x - radius);
            const int xmax = std::min(width - 1, x + radius);
            const int ymin = std::max(0, y - radius);
            const int ymax = std::min(height - 1, y + radius);
            for (int yy = ymin; yy <= ymax; ++yy)
            {
                auto *row = occupied.data() + static_cast<std::size_t>(yy) * width;
                std::fill(row + xmin, row + xmax + 1, static_cast<std::uint8_t>(1));
            }
        }

        const auto lower_score = [](const Feature &a, const Feature &b)
        {
            return a.score < b.score;
        };

        std::vector<Feature> choose_features(const Image &img, const Config &cfg, Scratch &s)
        {
            std::vector<Feature> selected;
            selected.reserve(std::min(s.candidates.size(),
                                      static_cast<std::size_t>(cfg.max_features)));
            s.occupied.assign(static_cast<std::size_t>(img.width) * img.height, 0);

            // Heap: O(n) construction; stop popping once max_features have been selected.
            // Avoids sorting all candidates when many more are available than needed.
            std::make_heap(s.candidates.begin(), s.candidates.end(), lower_score);

            while (!s.candidates.empty() &&
                   selected.size() < static_cast<std::size_t>(cfg.max_features))
            {
                std::pop_heap(s.candidates.begin(), s.candidates.end(), lower_score);
                Feature candidate = s.candidates.back();
                s.candidates.pop_back();

                const int x = candidate.x;
                const int y = candidate.y;
                if (s.occupied[static_cast<std::size_t>(y) * img.width + x])
                    continue;
                selected.push_back(candidate);

                // Original KLT _fillFeaturemap uses a square of radius mindist - 1.
                mark_occupied(
                    s.occupied, img.width, img.height,
                    x, y, cfg.min_dist - 1);
            }
            return selected;
        }

        void replace_features(
            const Image &img,
            const Config &cfg,
            Scratch &s,
            std::vector<Feature> &features)
        {
            s.occupied.assign(static_cast<std::size_t>(img.width) * img.height, 0);
            const int radius = cfg.min_dist - 1;
            for (const Feature &feature : features)
            {
                if (feature.score >= 0.0f)
                {
                    mark_occupied(
                        s.occupied, img.width, img.height,
                        feature.x, feature.y, radius);
                }
            }

            std::make_heap(s.candidates.begin(), s.candidates.end(), lower_score);
            std::size_t lost_index = 0;
            while (!s.candidates.empty())
            {
                while (lost_index < features.size() &&
                       features[lost_index].score >= 0.0f)
                {
                    ++lost_index;
                }
                if (lost_index == features.size())
                    break;

                std::pop_heap(s.candidates.begin(), s.candidates.end(), lower_score);
                const Feature candidate = s.candidates.back();
                s.candidates.pop_back();

                const std::size_t pos =
                    static_cast<std::size_t>(candidate.y) * img.width + candidate.x;
                if (s.occupied[pos])
                    continue;

                features[lost_index] = candidate;
                mark_occupied(
                    s.occupied, img.width, img.height,
                    candidate.x, candidate.y, radius);
                ++lost_index;
            }
        }

        void validate_gradients(
            const Image &img,
            const Image &gradx,
            const Image &grady)
        {
            const auto pixel_count =
                static_cast<std::size_t>(img.width) * img.height;
            if (gradx.width != img.width || gradx.height != img.height ||
                grady.width != img.width || grady.height != img.height ||
                gradx.data.size() != pixel_count ||
                grady.data.size() != pixel_count)
            {
                throw std::invalid_argument(
                    "Gradient images must match the source image");
            }
        }

    } // anonymous namespace

    std::vector<Feature> select_good_features(const Image &img, const Config &config)
    {
        assert(config.window_width >= 3 && config.window_width % 2 == 1);
        assert(config.window_height >= 3 && config.window_height % 2 == 1);
        assert(config.min_dist >= 0 && config.max_features >= 0);
        assert(img.width > 0 && img.height > 0);
        assert(img.data.size() == static_cast<std::size_t>(img.width) * img.height);

        if (config.max_features == 0 ||
            config.window_width > img.width || config.window_height > img.height)
            return {};

        auto &s = scratch();
        const auto smooth_kernels = make_kernels(compute_smooth_sigma(config));
        detail::gaussian_blur_into(
            img, s.blur_tmp, s.smoothed, smooth_kernels.gauss);
        compute_gradients(s.smoothed, s, make_kernels(config.sigma));
        find_candidates(img, s.gradx, s.grady, config, s);
        return choose_features(img, config, s);
    }

    void select_good_features(
        const Image &img,
        const Image &gradx,
        const Image &grady,
        std::vector<Feature> &features,
        const Config &config)
    {
        assert(config.window_width >= 3 && config.window_width % 2 == 1);
        assert(config.window_height >= 3 && config.window_height % 2 == 1);
        assert(config.min_dist >= 0);
        assert(img.width > 0 && img.height > 0);
        assert(img.data.size() == static_cast<std::size_t>(img.width) * img.height);

        if (std::none_of(
                features.begin(), features.end(),
                [](const Feature &feature)
                { return feature.score < 0.0f; }))
        {
            return;
        }

        validate_gradients(img, gradx, grady);
        if (config.window_width > img.width || config.window_height > img.height)
        {
            return;
        }

        auto &s = scratch();
        find_candidates(img, gradx, grady, config, s);
        replace_features(img, config, s, features);
    }

    void replace_lost_features(
        const Image &img,
        std::vector<Feature> &features,
        const Config &config)
    {
        assert(config.window_width >= 3 && config.window_width % 2 == 1);
        assert(config.window_height >= 3 && config.window_height % 2 == 1);
        assert(config.min_dist >= 0);
        assert(img.width > 0 && img.height > 0);
        assert(img.data.size() == static_cast<std::size_t>(img.width) * img.height);

        if (std::none_of(
                features.begin(), features.end(),
                [](const Feature &feature)
                { return feature.score < 0.0f; }))
        {
            return;
        }

        auto &s = scratch();
        const auto smooth_kernels = make_kernels(compute_smooth_sigma(config));
        detail::gaussian_blur_into(
            img, s.blur_tmp, s.smoothed, smooth_kernels.gauss);
        compute_gradients(s.smoothed, s, make_kernels(config.sigma));
        replace_lost_features(s.smoothed, s.gradx, s.grady, features, config);
    }

    void replace_lost_features(
        const Image &img,
        const Image &gradx,
        const Image &grady,
        std::vector<Feature> &features,
        const Config &config)
    {
        assert(config.window_width >= 3 && config.window_width % 2 == 1);
        assert(config.window_height >= 3 && config.window_height % 2 == 1);
        assert(config.min_dist >= 0);
        assert(img.width > 0 && img.height > 0);
        assert(img.data.size() == static_cast<std::size_t>(img.width) * img.height);

        if (std::none_of(
                features.begin(), features.end(),
                [](const Feature &feature)
                { return feature.score < 0.0f; }))
        {
            return;
        }

        select_good_features(img, gradx, grady, features, config);
    }

    void draw_features(
        const Image &img,
        Image &output,
        const std::vector<Feature> &features,
        const Color &color,
        int r)
    {
        if (img.pixel_format == PixelFormat::GREY)
            convert_grey_2_rgb(img, output);
        else
            output = img;

        for (const auto &f : features)
        {
            draw_rect(
                output,
                std::max(0, f.x - r),
                std::max(0, f.y - r),
                std::min(output.width, f.x + r + 1),
                std::min(output.height, f.y + r + 1),
                color);
        }
    }

} // namespace KLT
