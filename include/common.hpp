#pragma once

struct Config
{
    float sigma{1.0f};
    int window_width{5};
    int window_height{5};
    int min_dist{10};
    float min_eigenvalue{1.0f};
    int max_features{500};
    float smooth_sigma_fact{0.1f};
    float pyramid_sigma_fact{0.9f};
    float step_factor{1.0f};
    int max_iterations{10};
    float min_determinant{0.01f};
    float min_displacement{0.1f};
    float max_residue{10.0f};
    bool light_insensitive{false};

};