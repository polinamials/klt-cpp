#pragma once

#include "image.hpp"

#include <utility>
#include <vector>

namespace KLT
{
    struct Config
    {
        Config();

        float sigma{1.0f};
        int window_width{7};
        int window_height{7};
        int min_dist{10};
        float min_eigenvalue{1.0f};
        int max_features{500};
        float smooth_sigma_fact{0.1f};
        float pyramid_sigma_fact{0.9f};
        int num_pyramid_levels{1};
        int pyramid_subsampling{2};
        float step_factor{1.0f};
        int max_iterations{10};
        float min_determinant{0.01f};
        float min_displacement{0.1f};
        float max_residue{10.0f};
        bool light_insensitive{false};
        int borderx{0};
        int bordery{0};
    };

    struct Kernels
    {
        std::vector<float> gauss;
        std::vector<float> deriv;
    };

    Kernels make_kernels(float sigma);
    float compute_smooth_sigma(const Config &config);

    Image gaussian_blur(const Image &img, float sigma);
    std::pair<Image, Image> image_gradients(const Image &img, float sigma);

    namespace detail
    {
        void gaussian_blur_into(
            const Image &img,
            Image &temporary,
            Image &result,
            const std::vector<float> &kernel);

        void image_gradients_into(
            const Image &img,
            Image &temporary,
            Image &gradx,
            Image &grady,
            const Kernels &kernels);
    }

    int get_gauss_radius(float sigma);
    int get_deriv_radius(float sigma);

    void update_pyramid_settings(Config &config, int search_range);
    void update_borders(Config &config);

    Config create_config(int search_range = 15);
}
