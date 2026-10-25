#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace KLT
{
    namespace
    {
        // KLT uses correlation (sample at x+k), including the derivative sign.
        // Store only half of each symmetric/antisymmetric kernel and process the
        // +/- samples together.
        template <bool Vertical, bool Derivative>
        void filter_axis(
            const Image &src,
            Image &dst,
            const std::vector<float> &kernel)
        {
            const int width = src.width;
            const int height = src.height;
            const int radius = static_cast<int>(kernel.size()) - 1;
            const int stride = Vertical ? width : 1;
            const float *const input = src.data.data();
            float *const output = dst.data.data();

            for (int y = 0; y < height; ++y)
            {
                const bool interior_row =
                    y >= radius && y + radius < height;
                for (int x = 0; x < width; ++x)
                {
                    const std::size_t position =
                        static_cast<std::size_t>(y) * width + x;
                    float sum = Derivative
                                    ? 0.0f
                                    : kernel[0] * input[position];
                    const bool interior = Vertical
                                              ? interior_row
                                              : x >= radius &&
                                                    x + radius < width;

                    if (interior)
                    {
                        for (int i = 1; i <= radius; ++i)
                        {
                            const std::size_t offset =
                                static_cast<std::size_t>(i) * stride;
                            if constexpr (Derivative)
                            {
                                sum += kernel[i] *
                                       (input[position + offset] -
                                        input[position - offset]);
                            }
                            else
                            {
                                sum += kernel[i] *
                                       (input[position + offset] +
                                        input[position - offset]);
                            }
                        }
                    }
                    else
                    {
                        for (int i = 1; i <= radius; ++i)
                        {
                            float plus;
                            float minus;
                            if constexpr (Vertical)
                            {
                                plus = input[
                                    static_cast<std::size_t>(
                                        std::min(height - 1, y + i)) *
                                        width +
                                    x];
                                minus = input[
                                    static_cast<std::size_t>(
                                        std::max(0, y - i)) *
                                        width +
                                    x];
                            }
                            else
                            {
                                plus = input[
                                    static_cast<std::size_t>(y) * width +
                                    std::min(width - 1, x + i)];
                                minus = input[
                                    static_cast<std::size_t>(y) * width +
                                    std::max(0, x - i)];
                            }

                            if constexpr (Derivative)
                                sum += kernel[i] * (plus - minus);
                            else
                                sum += kernel[i] * (plus + minus);
                        }
                    }
                    output[position] = sum;
                }
            }
        }
    }

    namespace detail
    {
        void gaussian_blur_into(
            const Image &img,
            Image &temporary,
            Image &result,
            const std::vector<float> &kernel)
        {
            resize_image(temporary, img.width, img.height);
            resize_image(result, img.width, img.height);
            filter_axis<false, false>(img, temporary, kernel);
            filter_axis<true, false>(temporary, result, kernel);
        }

        void image_gradients_into(
            const Image &img,
            Image &temporary,
            Image &gradx,
            Image &grady,
            const Kernels &kernels)
        {
            resize_image(temporary, img.width, img.height);
            resize_image(gradx, img.width, img.height);
            resize_image(grady, img.width, img.height);

            filter_axis<false, true>(img, temporary, kernels.deriv);
            filter_axis<true, false>(temporary, gradx, kernels.gauss);

            filter_axis<true, true>(img, temporary, kernels.deriv);
            filter_axis<false, false>(temporary, grady, kernels.gauss);
        }
    }

    int get_gauss_radius(float sigma)
    {
        return static_cast<int>(3.0348542587702925 * sigma);
    }

    int get_deriv_radius(float sigma)
    {
        return static_cast<int>(3.571606252443768 * sigma);
    }

    Kernels make_kernels(float sigma)
    {
        if (!(sigma > 0.0f) || !std::isfinite(sigma))
            throw std::invalid_argument("sigma must be finite and positive");

        const int rg = get_gauss_radius(sigma);
        const int rd = get_deriv_radius(sigma);
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

    float compute_smooth_sigma(const Config &config)
    {
        return config.smooth_sigma_fact *
               std::max(config.window_width, config.window_height);
    }

    Image gaussian_blur(const Image &img, float sigma)
    {
        Image temporary;
        Image result;
        detail::gaussian_blur_into(
            img, temporary, result, make_kernels(sigma).gauss);
        return result;
    }

    std::pair<Image, Image> image_gradients(const Image &img, float sigma)
    {
        Image temporary;
        Image gradx;
        Image grady;
        detail::image_gradients_into(
            img, temporary, gradx, grady, make_kernels(sigma));
        return {std::move(gradx), std::move(grady)};
    }

    void update_pyramid_settings(Config &config, int search_range)
    {
        auto window_halfwidth = std::min(config.window_width, config.window_height) / 2.0f;

        const float subsampling = static_cast<float>(search_range) / window_halfwidth;

        if (subsampling < 1.0)
        { /* 1.0 = 0+1 */
            config.num_pyramid_levels = 1;
        }
        else if (subsampling <= 3.0)
        { /* 3.0 = 2+1 */
            config.num_pyramid_levels = 2;
            config.pyramid_subsampling = 2;
        }
        else if (subsampling <= 5.0)
        { /* 5.0 = 4+1 */
            config.num_pyramid_levels = 2;
            config.pyramid_subsampling = 4;
        }
        else if (subsampling <= 9.0)
        { /* 9.0 = 8+1 */
            config.num_pyramid_levels = 2;
            config.pyramid_subsampling = 8;
        }
        else
        {
            /* The following lines are derived from the formula:
            search_range =
            window_halfwidth * \sum_{i=0}^{num_pyramid_levels-1} 8^i,
            which is the same as:
            search_range =
            window_halfwidth * (8^num_pyramid_levels - 1)/(8 - 1).
            Then, the value is rounded up to the nearest integer. */
            const float val = static_cast<float>(
                std::log(7.0f * subsampling + 1.0f) / std::log(8.0f));
            config.num_pyramid_levels = static_cast<int>(val + 0.99f);
            config.pyramid_subsampling = 8;
        }
    }

    void update_borders(Config &config)
    {
        // get_gauss_radius() already returns a half-width.  The original C
        // implementation obtains a full kernel width and divides that by two.
        const int smooth_gauss_hw = get_gauss_radius(compute_smooth_sigma(config));
        const int pyramid_gauss_hw =
            get_gauss_radius(config.pyramid_sigma_fact * config.pyramid_subsampling);
        /* Compute the # of invalid pixels at each level of the pyramid.
        n_invalid_pixels is computed with respect to the ith level
        of the pyramid.  So, e.g., if n_invalid_pixels = 5 after
        the first iteration, then there are 5 invalid pixels in
        level 1, which translated means 5*subsampling invalid pixels
        in the original level 0. */
        int n_invalid_pixels = smooth_gauss_hw;
        for (int i = 1; i < config.num_pyramid_levels; ++i)
        {
            float val = ((float)n_invalid_pixels + pyramid_gauss_hw) / config.pyramid_subsampling;
            n_invalid_pixels = (int)(val + 0.99); /* Round up */
        }

        /* ss_power = ss^(num_levels-1) */
        int ss_power = 1;
        for (int i = 1; i < config.num_pyramid_levels; i++)
            ss_power *= config.pyramid_subsampling;

        /* Compute border by translating invalid pixels back into */
        /* original image */
        int window_hw = std::max(config.window_width, config.window_height) / 2;
        int border = (n_invalid_pixels + window_hw) * ss_power;

        config.borderx = border;
        config.bordery = border;
    }

    Config::Config()
    {
        update_pyramid_settings(*this, 15);
        update_borders(*this);
    }

    Config create_config(int search_range)
    {
        Config config{};
        update_pyramid_settings(config, search_range);
        update_borders(config);
        return config;
    }
}
