#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "image.hpp"
#include "shi_tomasi.hpp"

#include <string>
#include <string_view>
#include <vector>
#include <filesystem>
#include <iostream>

std::string g_image_path;

int main(int argc, char **argv)
{
    doctest::Context context;

    constexpr std::string_view prefix = "--image_path=";

    std::vector<char *> doctest_args{argv[0]};

    for (int i = 1; i < argc; ++i)
    {
        std::string_view arg = argv[i];

        if (arg.starts_with(prefix))
            g_image_path = arg.substr(prefix.size());
        else
            doctest_args.push_back(argv[i]);
    }

    context.applyCommandLine(
        static_cast<int>(doctest_args.size()),
        doctest_args.data());

    return context.run();
}

Image test_image()
{
    auto img = make_image(360, 240, 255.0);
    draw_rect(img, 10, 10, 25, 35, 0);
    draw_rect(img, 100, 110, 200, 220, 0);

    return img;
}

const std::vector<std::pair<int, int>> expected_features()
{
    return {
        {10, 10}, {25, 10}, {10, 35}, {25, 35}, {100, 110}, {200, 110}, {100, 220}, {200, 220}};
}

// Image tests
TEST_CASE("save load image")
{
    auto img = test_image();
    std::string filepath = g_image_path + "/test.pgm";

    save_pgm(img, filepath);
    CHECK(std::filesystem::exists(std::filesystem::path(filepath)));
    auto img_loaded = load_pgm(filepath);
    CHECK(img_loaded == img);
}

// Feature extraction tests
TEST_CASE("extract features")
{
    auto img = test_image();
    Config config{};
    auto features = select_good_features(img, config);

    const auto expected = expected_features();

    constexpr int tolerance = 3;

    REQUIRE(features.size() == expected.size());

    for (const auto &[x, y] : expected)
    {
        const bool found = std::any_of(
            features.begin(), features.end(),
            [&](const FeatureCandidate &feature)
            {
                return std::abs(feature.x - x) <= tolerance &&
                       std::abs(feature.y - y) <= tolerance;
            });

        CHECK_MESSAGE(found, "Feature not detected near (", x, ", ", y, ")");
    }
}