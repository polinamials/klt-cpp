#include "image.hpp"

#include <algorithm>

float Image::get(int x, int y) const
{
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    return data[y * width + x];
}

Image make_image(int width, int height, float value)
{
    Image img{width, height};
    img.data.resize(static_cast<std::size_t>(width) * height, value);
    return img;
}

void draw_rect(Image& img, int x0, int y0, int x1, int y1, float value)
{
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x)
            img.data[y * img.width + x] = value;
}
