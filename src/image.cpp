#include "image.hpp"

#include <algorithm>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>
#include <limits>
#include <cctype>

namespace KLT
{
    namespace
    {
        std::string pixel_2_magic(const PixelFormat &pixel_format)
        {
            switch (pixel_format)
            {
            case PixelFormat::GREY:
                return "P5";
            case PixelFormat::RGB:
                return "P6";
            default:
                throw std::runtime_error("invalid pixel format");
            }
        }

        PixelFormat magic_2_pixel(const std::string &magic)
        {
            if (magic == "P5")
                return PixelFormat::GREY;
            else if (magic == "P6")
                return PixelFormat::RGB;
            else
                throw std::runtime_error("invalid magic number");
        }

        void skip_comments(std::istream &file)
        {
            while (true)
            {
                file >> std::ws;

                if (file.peek() != '#')
                    break;

                file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
        }
    }

    float Image::get(int x, int y) const
    {
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
        return data[y * width + x];
    }

    Image make_image(int width, int height, const Color &color)
    {
        Image img{width, height};
        img.pixel_format = PixelFormat::RGB;
        img.data.resize(static_cast<std::size_t>(width) * height * 3);
        for (int i = 0; i < width * height; ++i)
        {
            img.data[3 * i] = color.r;
            img.data[3 * i + 1] = color.g;
            img.data[3 * i + 2] = color.b;
        }
        return img;
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

    void draw_rect(Image &img, int x0, int y0, int x1, int y1, const Color &color)
    {
        for (int y = y0; y < y1; ++y)
        {
            for (int x = x0; x < x1; ++x)
            {
                const int idx = 3 * (y * img.width + x);

                img.data[idx] = color.r;
                img.data[idx + 1] = color.g;
                img.data[idx + 2] = color.b;
            }
        }
    }

    void save_img(const Image &img, const std::string &path)
    {
        std::ofstream file(path, std::ios::binary);

        std::string magic = pixel_2_magic(img.pixel_format);

        file << magic << "\n"
             << img.width << " " << img.height << "\n255\n";

        for (float pixel : img.data)
        {
            const auto value = static_cast<unsigned char>(
                std::clamp(pixel, 0.0f, 255.0f));

            file.write(reinterpret_cast<const char *>(&value), 1);
        }
    }

    Image load_img(const std::string &path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error("Cannot open file: " + path);

        std::string magic;
        int width, height, maxval;

        file >> magic;

        skip_comments(file);
        file >> width;

        skip_comments(file);
        file >> height;

        skip_comments(file);
        file >> maxval;

        auto pixel_format = magic_2_pixel(magic);

        if (!file || width <= 0 || height <= 0 || maxval != 255)
            throw std::runtime_error("Invalid image file");

        // Consume the single whitespace character separating
        // the header from the binary pixel data.
        const int separator = file.get();
        if (separator == EOF || !std::isspace(static_cast<unsigned char>(separator)))
            throw std::runtime_error("Invalid PNM header separator");

        const int channels = pixel_format == PixelFormat::RGB ? 3 : 1;

        std::vector<unsigned char> pixels(
            static_cast<size_t>(width) * height * channels);

        file.read(
            reinterpret_cast<char *>(pixels.data()),
            static_cast<std::streamsize>(pixels.size()));

        if (!file)
            throw std::runtime_error("Failed to read image pixel data");

        Image img{width, height};
        img.pixel_format = pixel_format;
        img.data.assign(pixels.begin(), pixels.end());

        return img;
    }

    void resize_image(Image &img, int w, int h)
    {
        img.width = w;
        img.height = h;

        const int channels = img.pixel_format == PixelFormat::RGB ? 3 : 1;
        img.data.resize(static_cast<std::size_t>(w) * h * channels);
    }

    void convert_grey_2_rgb(const Image &img, Image &output)
    {
        output.pixel_format = PixelFormat::RGB;
        resize_image(output, img.width, img.height);
        for (size_t i = 0; i < img.data.size(); ++i)
        {
            output.data[3 * i] = img.data[i];
            output.data[3 * i + 1] = img.data[i];
            output.data[3 * i + 2] = img.data[i];
        }
    }

}
