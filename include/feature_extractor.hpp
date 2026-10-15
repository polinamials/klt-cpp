#pragma once

#include <vector>
#include <utility>

#include "image.hpp"
#include "common.hpp"

namespace FeatureExtractor
{
    struct Feature
    {
        int x;
        int y;
        float score;
    };

    std::vector<Feature> select_good_features(const IO::Image &img, const Common::Config &config);

    IO::Image gaussian_blur(const IO::Image &img, float sigma);

    std::pair<IO::Image, IO::Image> image_gradients(const IO::Image &img, float sigma);
}
