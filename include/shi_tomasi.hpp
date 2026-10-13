#pragma once

#include <vector>
#include <utility>

#include "image.hpp"

struct FeatureCandidate
{
    int x;
    int y;
    float score;
};

struct Config
{
    float sigma{1.0f};
    int window_width{5};
    int window_height{5};
    int min_dist{10};
    float min_eigenvalue{1.0f};
    int max_features{500};
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
