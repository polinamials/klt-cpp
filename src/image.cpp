#include "image.hpp"

#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

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

void draw_rect(Image &img, int x0, int y0, int x1, int y1, float value)
{
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x)
            img.data[y * img.width + x] = value;
}

void save_pgm(const Image &img, const std::string &path)
{
    std::ofstream file(path, std::ios::binary);

    file << "P5\n"
         << img.width << " " << img.height << "\n255\n";

    for (float pixel : img.data)
    {
        const auto value = static_cast<unsigned char>(
            std::clamp(pixel, 0.0f, 255.0f));

        file.write(reinterpret_cast<const char *>(&value), 1);
    }
}

Image load_pgm(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file: " + path);

    std::string magic;
    int width, height, maxval;

    file >> magic >> width >> height >> maxval;

    if (!file || magic != "P5" || width <= 0 || height <= 0 || maxval != 255)
        throw std::runtime_error("Invalid PGM file");

    file.get();

    std::vector<unsigned char> pixels(
        static_cast<size_t>(width) * height);

    file.read(reinterpret_cast<char *>(pixels.data()), pixels.size());

    if (!file)
        throw std::runtime_error("Failed to read PGM pixel data");

    Image img{width, height};
    img.data.assign(pixels.begin(), pixels.end());

    return img;
}
