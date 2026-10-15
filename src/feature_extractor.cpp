#include "feature_extractor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace FeatureExtractor
{
    std::pair<std::vector<float>, std::vector<float>> compute_kernels(float sigma)
    {
        if (!std::isfinite(sigma) || sigma <= 0.0f)
            throw std::invalid_argument("sigma must be positive and finite");

        constexpr double gaussian_cutoff = 3.0348542587702925;
        constexpr double derivative_cutoff = 3.571606252443768;

        const int gauss_radius = static_cast<int>(gaussian_cutoff * sigma);
        const int deriv_radius = static_cast<int>(derivative_cutoff * sigma);

        if (deriv_radius == 0)
            throw std::invalid_argument("sigma is too small");

        auto make_kernel = [&](int radius, bool derivative)
        {
            std::vector<float> kernel(2 * radius + 1);
            float sum = 0.0f;

            for (int i = -radius; i <= radius; ++i)
            {
                const float g = std::exp(-float(i * i) / (2.0f * sigma * sigma));
                const float value = derivative ? -i * g : g;
                kernel[i + radius] = value;
                sum += derivative ? -i * value : value;
            }

            for (float &value : kernel)
                value /= sum;

            return kernel;
        };

        return {make_kernel(gauss_radius, false), make_kernel(deriv_radius, true)};
    }

    Image convolve(const Image &img, const std::vector<float> &kernel, bool horizontal)
    {
        assert(!kernel.empty() && kernel.size() % 2 == 1);

        Image output{img.width, img.height};
        output.data.resize(static_cast<std::size_t>(img.width) * img.height);
        const int radius = static_cast<int>(kernel.size() / 2);

        for (int y = 0; y < img.height; ++y)
            for (int x = 0; x < img.width; ++x)
            {
                float sum = 0.0f;
                for (int k = -radius; k <= radius; ++k)
                {
                    const int dx = horizontal ? k : 0;
                    const int dy = horizontal ? 0 : k;
                    sum += img.get(x + dx, y + dy) * kernel[k + radius];
                }
                output.data[y * img.width + x] = sum;
            }

        return output;
    }

    Image convolve_horiz(const Image &img, const std::vector<float> &kernel)
    {
        return convolve(img, kernel, true);
    }

    Image convolve_vert(const Image &img, const std::vector<float> &kernel)
    {
        return convolve(img, kernel, false);
    }

    Image convolve_separate(const Image &img, const std::vector<float> &kernel_horiz,
                            const std::vector<float> &kernel_vert)
    {
        return convolve_vert(convolve_horiz(img, kernel_horiz), kernel_vert);
    }

    std::pair<Image, Image> compute_gradients(const Image &img, float sigma)
    {
        auto [gauss, deriv] = compute_kernels(sigma);
        return {convolve_separate(img, deriv, gauss),
                convolve_separate(img, gauss, deriv)};
    }

    float min_eigenvalue(float gxx, float gxy, float gyy)
    {
        return 0.5f * (gxx + gyy -
                       std::sqrt((gxx - gyy) * (gxx - gyy) + 4.0f * gxy * gxy));
    }

    std::vector<Feature> select_good_features(const Image &img,
                                              const Config &config)
    {
        assert(config.window_width >= 3 && config.window_width % 2 == 1);
        assert(config.window_height >= 3 && config.window_height % 2 == 1);
        assert(config.min_dist >= 0);
        assert(config.max_features >= 0);
        assert(img.width > 0 && img.height > 0);
        assert(img.data.size() == static_cast<std::size_t>(img.width) * img.height);

        if (config.max_features == 0)
            return {};

        auto [gradx, grady] = compute_gradients(img, config.sigma);
        const int window_hw = config.window_width / 2;
        const int window_hh = config.window_height / 2;
        std::vector<Feature> candidates;

        for (int y = window_hh; y < img.height - window_hh; ++y)
            for (int x = window_hw; x < img.width - window_hw; ++x)
            {
                float gxx = 0.0f;
                float gxy = 0.0f;
                float gyy = 0.0f;

                for (int yy = y - window_hh; yy <= y + window_hh; ++yy)
                    for (int xx = x - window_hw; xx <= x + window_hw; ++xx)
                    {
                        const float gx = gradx.data[yy * img.width + xx];
                        const float gy = grady.data[yy * img.width + xx];
                        gxx += gx * gx;
                        gxy += gx * gy;
                        gyy += gy * gy;
                    }

                const float score = min_eigenvalue(gxx, gxy, gyy);
                if (score >= config.min_eigenvalue)
                    candidates.push_back({x, y, score});
            }

        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &a, const auto &b)
                  { return a.score > b.score; });

        std::vector<Feature> features;
        features.reserve(config.max_features);
        std::vector<std::uint8_t> occupied(
            static_cast<std::size_t>(img.width) * img.height, 0);
        const int dist = config.min_dist;
        const int dist_sq = dist * dist;

        for (const auto &candidate : candidates)
        {
            const int x = candidate.x;
            const int y = candidate.y;
            if (occupied[y * img.width + x])
                continue;

            features.push_back(candidate);
            if (features.size() >= static_cast<std::size_t>(config.max_features))
                break;

            for (int dy = -dist; dy <= dist; ++dy)
            {
                const int yy = y + dy;
                if (yy < 0 || yy >= img.height)
                    continue;

                for (int dx = -dist; dx <= dist; ++dx)
                {
                    const int xx = x + dx;
                    if (xx >= 0 && xx < img.width && dx * dx + dy * dy <= dist_sq)
                        occupied[yy * img.width + xx] = 1;
                }
            }
        }

        return features;
    }
}