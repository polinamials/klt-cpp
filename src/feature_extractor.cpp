#include "feature_extractor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <vector>

using namespace IO;
using namespace Common;

namespace FeatureExtractor
{
    namespace
    {

        constexpr double gauss_cutoff = 3.0348542587702925;
        constexpr double deriv_cutoff = 3.571606252443768;

        struct Kernels
        {
            std::vector<float> gauss;
            std::vector<float> deriv;
        };

        Kernels make_kernels(float sigma)
        {
            if (!(sigma > 0.0f) || !std::isfinite(sigma))
                throw std::invalid_argument("sigma must be finite and positive");

            const int rg = static_cast<int>(gauss_cutoff * sigma);
            const int rd = static_cast<int>(deriv_cutoff * sigma);
            if (rd == 0)
                throw std::invalid_argument("sigma is too small for a derivative kernel");

            Kernels k;
            k.gauss.resize(static_cast<std::size_t>(rg) + 1);
            k.deriv.resize(static_cast<std::size_t>(rd) + 1);
            // Store only the center and positive half of each symmetric/antisymmetric kernel.
            // k.gauss[i] = G(+i); k.deriv[i] = G'(+i), so k.deriv[i] < 0 for i > 0.
            const float inv_two_sigma_sq = 0.5f / (sigma * sigma);
            float gauss_sum = 0.0f;
            float deriv_moment = 0.0f;
            for (int i = 0; i <= std::max(rg, rd); ++i)
            {
                const float g = std::exp(-static_cast<float>(i * i) * inv_two_sigma_sq);
                if (i <= rg)
                {
                    k.gauss[i] = g;
                    gauss_sum += (i == 0 ? 1.0f : 2.0f) * g;
                }
                if (i <= rd)
                {
                    k.deriv[i] = -static_cast<float>(i) * g;
                    deriv_moment += 2.0f * static_cast<float>(i * i) * g;
                }
            }
            for (float &v : k.gauss)
                v /= gauss_sum;
            for (float &v : k.deriv)
                v /= deriv_moment;
            return k;
        }

        // The caller's original implementation uses correlation (sample at x+k),
        // not mathematically reversed convolution. Keep that convention, including
        // the sign of the Gaussian derivative, for compatibility with the tracker.
        // Exploit kernel symmetry: one multiply for each +/- pair.
        template <bool Vertical, bool Derivative>
        void filter_axis(const Image &src, Image &dst, const std::vector<float> &kernel)
        {
            const int w = src.width;
            const int h = src.height;
            const int radius = static_cast<int>(kernel.size()) - 1;
            const int stride = Vertical ? w : 1;
            const float *const in = src.data.data();
            float *const out = dst.data.data();

            for (int y = 0; y < h; ++y)
            {
                const bool interior_row = y >= radius && y + radius < h;
                for (int x = 0; x < w; ++x)
                {
                    const std::size_t pos = static_cast<std::size_t>(y) * w + x;
                    float sum = Derivative ? 0.0f : kernel[0] * in[pos];
                    const bool interior = Vertical ? interior_row : (x >= radius && x + radius < w);

                    if (interior)
                    {
                        for (int i = 1; i <= radius; ++i)
                        {
                            const std::size_t offset = static_cast<std::size_t>(i) * stride;
                            if constexpr (Derivative)
                                sum += kernel[i] * (in[pos + offset] - in[pos - offset]);
                            else
                                sum += kernel[i] * (in[pos + offset] + in[pos - offset]);
                        }
                    }
                    else
                    {
                        // Match Image::get() clamping at image boundaries.
                        for (int i = 1; i <= radius; ++i)
                        {
                            float plus, minus;
                            if constexpr (Vertical)
                            {
                                plus = in[static_cast<std::size_t>(std::min(h - 1, y + i)) * w + x];
                                minus = in[static_cast<std::size_t>(std::max(0, y - i)) * w + x];
                            }
                            else
                            {
                                plus = in[static_cast<std::size_t>(y) * w + std::min(w - 1, x + i)];
                                minus = in[static_cast<std::size_t>(y) * w + std::max(0, x - i)];
                            }
                            if constexpr (Derivative)
                                sum += kernel[i] * (plus - minus);
                            else
                                sum += kernel[i] * (plus + minus);
                        }
                    }
                    out[pos] = sum;
                }
            }
        }

        struct Scratch
        {
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
            resize_image(s.tmp, img.width, img.height);
            resize_image(s.gradx, img.width, img.height);
            resize_image(s.grady, img.width, img.height);

            // x gradient: derivative horizontally, Gaussian smoothing vertically.
            filter_axis<false, true>(img, s.tmp, k.deriv);
            filter_axis<true, false>(s.tmp, s.gradx, k.gauss);

            // y gradient: derivative vertically, Gaussian smoothing horizontally.
            filter_axis<true, true>(img, s.tmp, k.deriv);
            filter_axis<false, false>(s.tmp, s.grady, k.gauss);
        }

        // An exact sliding box sum of gradient products, apart from floating-point
        // addition order. Complexity: O(width * height), not O(pixels * window area).
        void find_candidates(const Image &img, const Config &cfg, Scratch &s)
        {
            const int w = img.width;
            const int h = img.height;
            const int hw = cfg.window_width / 2;
            const int hh = cfg.window_height / 2;
            const float threshold = std::max(1.0f, static_cast<float>(cfg.min_eigenvalue));

            s.candidates.clear();
            if (cfg.window_width > w || cfg.window_height > h)
                return;

            s.col_xx.assign(static_cast<std::size_t>(w), 0.0f);
            s.col_xy.assign(static_cast<std::size_t>(w), 0.0f);
            s.col_yy.assign(static_cast<std::size_t>(w), 0.0f);

            const float *const gx = s.gradx.data.data();
            const float *const gy = s.grady.data.data();

            // Initial vertical sums (window rows 0..2*hh inclusive).
            for (int yy = 0; yy < cfg.window_height; ++yy)
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

            for (int y = hh; y < h - hh; ++y)
            {
                if (y != hh)
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
                for (int xx = 0; xx < cfg.window_width; ++xx)
                {
                    gxx += s.col_xx[xx];
                    gxy += s.col_xy[xx];
                    gyy += s.col_yy[xx];
                }

                for (int x = hw; x < w - hw; ++x)
                {
                    if (x != hw)
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

        std::vector<Feature> choose_features(const Image &img, const Config &cfg, Scratch &s)
        {
            std::vector<Feature> selected;
            selected.reserve(std::min(s.candidates.size(),
                                      static_cast<std::size_t>(cfg.max_features)));
            s.occupied.assign(static_cast<std::size_t>(img.width) * img.height, 0);

            const auto lower_score = [](const Feature &a, const Feature &b)
            {
                return a.score < b.score;
            };
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
                const int radius = cfg.min_dist - 1;
                if (radius >= 0)
                {
                    const int xmin = std::max(0, x - radius);
                    const int xmax = std::min(img.width - 1, x + radius);
                    const int ymin = std::max(0, y - radius);
                    const int ymax = std::min(img.height - 1, y + radius);
                    for (int yy = ymin; yy <= ymax; ++yy)
                    {
                        auto *row = s.occupied.data() + static_cast<std::size_t>(yy) * img.width;
                        std::fill(row + xmin, row + xmax + 1, static_cast<std::uint8_t>(1));
                    }
                }
            }
            return selected;
        }

    } // anonymous namespace

    Image gaussian_blur(const Image &img, float sigma)
    {
        const auto k = make_kernels(sigma);
        Image tmp{img.width, img.height};
        Image result{img.width, img.height};
        resize_image(tmp, img.width, img.height);
        resize_image(result, img.width, img.height);
        filter_axis<false, false>(img, tmp, k.gauss);
        filter_axis<true, false>(tmp, result, k.gauss);
        return result;
    }

    std::pair<Image, Image> image_gradients(const Image &img, float sigma)
    {
        Scratch local;
        compute_gradients(img, local, make_kernels(sigma));
        return {std::move(local.gradx), std::move(local.grady)};
    }

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
        compute_gradients(img, s, make_kernels(config.sigma));
        find_candidates(img, config, s);
        return choose_features(img, config, s);
    }

} // namespace FeatureExtractor
