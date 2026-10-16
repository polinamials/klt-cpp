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

    // Fills lost feature slots using a caller-provided smoothed image and
    // matching gradients.
    void select_good_features(
        const IO::Image &img,
        const IO::Image &gradx,
        const IO::Image &grady,
        std::vector<Feature> &features,
        const Common::Config &config);

    void replace_lost_features(
        const IO::Image &img,
        std::vector<Feature> &features,
        const Common::Config &config);

    // Reuses level zero of the current image and gradient pyramids.
    void replace_lost_features(
        const IO::Image &img,
        const IO::Image &gradx,
        const IO::Image &grady,
        std::vector<Feature> &features,
        const Common::Config &config);

    IO::Image gaussian_blur(const IO::Image &img, float sigma);

    std::pair<IO::Image, IO::Image> image_gradients(const IO::Image &img, float sigma);

    void draw_features(const IO::Image &img, IO::Image &output, const std::vector<Feature> &features, const IO::Color &color, int r = 1);

}
