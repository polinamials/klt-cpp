#pragma once

#include <vector>
#include <utility>

#include "image.hpp"
#include "common.hpp"

struct FeatureCandidate
{
    int x;
    int y;
    float score;
};

std::pair<std::vector<float>, std::vector<float>> compute_kernels(float sigma = 1.0f);

Image convolve(const Image &img, const std::vector<float> &kernel, bool horizontal);

Image convolve_horiz(const Image &img, const std::vector<float> &kernel);

Image convolve_vert(const Image &img, const std::vector<float> &kernel);

Image convolve_separate(const Image &img, const std::vector<float> &kernel_horiz,
                        const std::vector<float> &kernel_vert);

std::pair<Image, Image> compute_gradients(const Image &img, float sigma = 1.0f);

float min_eigenvalue(float gxx, float gxy, float gyy);

std::vector<FeatureCandidate> select_good_features(const Image &img,
                                                   const Config &config);
