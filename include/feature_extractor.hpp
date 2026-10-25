#pragma once

#include <vector>
#include "image.hpp"
#include "common.hpp"

namespace KLT
{
    struct Feature
    {
        int x;
        int y;
        float score;
    };

    std::vector<Feature> select_good_features(
        const Image &img,
        const Config &config);

    // Fills lost feature slots using a caller-provided smoothed image and
    // matching gradients.
    void select_good_features(
        const Image &img,
        const Image &gradx,
        const Image &grady,
        std::vector<Feature> &features,
        const Config &config);

    void replace_lost_features(
        const Image &img,
        std::vector<Feature> &features,
        const Config &config);

    // Reuses level zero of the current image and gradient pyramids.
    void replace_lost_features(
        const Image &img,
        const Image &gradx,
        const Image &grady,
        std::vector<Feature> &features,
        const Config &config);

    void draw_features(
        const Image &img,
        Image &output,
        const std::vector<Feature> &features,
        const Color &color,
        int radius = 1);

}
