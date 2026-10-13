#pragma once

#include <vector>
#include <string>

struct Image
{
    int width{};
    int height{};
    std::vector<float> data{};

    float get(int x, int y) const;
};

Image make_image(int width, int height, float value = 255.0f);

void draw_rect(Image& img, int x0, int y0, int x1, int y1, float value);

void save_pgm(const Image &img, const std::string &path);

Image load_pgm(const std::string &path);
