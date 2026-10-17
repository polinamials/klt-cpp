#pragma once

#include "common.hpp"
#include "image.hpp"
#include "feature_extractor.hpp"

#include <vector>

using namespace FeatureExtractor;
using namespace IO;
using namespace Common;

namespace Tracker
{
    struct Pyramid
    {
        int subsampling{2};
        std::vector<Image> levels;
    };

    Image compute_smoothed_image(const Image &img, float sigma);

    Pyramid compute_pyramid(const Image &img, float sigma_fact,
                            int subsampling = 2, int num_levels = 2);

    struct ImagePyramids
    {
        Pyramid image;
        Pyramid gradx;
        Pyramid grady;
    };

    ImagePyramids compute_image_pyramids(const Image &img, const Config &config);
    float interpolate(const Image &img, float x, float y);

    enum class TrackStatus
    {
        Tracked,
        SmallDeterminant,
        MaxIterations,
        OutOfBounds,
        LargeResidue,
        Replaced
    };

    struct TrackResult
    {
        float x = 0.0f;
        float y = 0.0f;
        TrackStatus status = TrackStatus::Tracked;
    };

    struct Track
    {
        // One entry per input image, preserving the feature's subpixel position.
        std::vector<TrackResult> positions;
    };

    bool patch_fits(const Image &img, float x, float y,
                    int half_width, int half_height);

    TrackResult track_feature(float x1, float y1, float x2, float y2,
                              const Image &img1, const Image &gradx1,
                              const Image &grady1, const Image &img2,
                              const Image &gradx2, const Image &grady2,
                              const Config &config);

    // Tracks selected features between two images.
    std::vector<TrackResult> track_features(
        const Image &img1, const Image &img2,
        const std::vector<Feature> &features,
        const Config &config);

    // Selects features in the first image and follows those same features through
    // the sequence. Each returned Track contains one result per input image.
    std::vector<Track> track(const std::vector<Image> &imgs, const Config &config);
}
