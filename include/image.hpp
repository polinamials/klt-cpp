#pragma once

#include <vector>
#include <string>

namespace IO
{
    enum class PixelFormat
    {
        GREY,
        RGB
    };

    struct Color
    {
        float r{};
        float g{};
        float b{};

        Color(float r_in, float g_in, float b_in) : r(r_in), g(g_in), b(b_in) {}
    };

    struct Image
    {
        int width{};
        int height{};
        std::vector<float> data{};
        PixelFormat pixel_format{PixelFormat::GREY};

        float get(int x, int y) const;

        bool operator==(const Image &) const = default;
    };

    Image make_image(int width, int height, Color &color);

    Image make_image(int width, int height, float value = 255.0f);

    void draw_rect(Image &img, int x0, int y0, int x1, int y1, const Color &color);

    void draw_rect(Image &img, int x0, int y0, int x1, int y1, float value);

    void save_img(const Image &img, const std::string &path);

    Image load_img(const std::string &path);

    void resize_image(Image &img, int w, int h);

    void convert_grey_2_rgb(const Image &img, Image &output);
}