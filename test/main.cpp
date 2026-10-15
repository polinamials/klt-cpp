#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "image.hpp"
#include "feature_extractor.hpp"
#include "tracker.hpp"

#include <string>
#include <string_view>
#include <vector>
#include <filesystem>
#include <iostream>

std::string g_image_path;

using namespace Tracker;

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

Image test_image(const int x = 0, const int y = 0)
{
    auto img = make_image(360, 240, 255.0);
    draw_rect(img, 10 + x, 10 + y, 25 + x, 35 + y, 0);
    draw_rect(img, 100 + x, 110 + y, 200 + x, 220 + y, 0);

    return img;
}

Image test_image_rgb(const int x = 0, const int y = 0)
{
    auto bg = Color{255.0f, 255.0f, 255.0f};
    auto img = make_image(360, 240, bg);
    auto c1 = Color(255.0f, 0.0f,0.0f);
    draw_rect(img, 10 + x, 10 + y, 25 + x, 35 + y, c1);
    auto c2 = Color(0.0f,0.0f,255.0f);
    draw_rect(img, 100 + x, 110 + y, 200 + x, 220 + y, c2);

    return img;
}

const std::vector<std::pair<int, int>> expected_features(int x = 0, int y = 0)
{
    return {
        {10 + x, 10 + y}, {25 + x, 10 + y}, {10 + x, 35 + y}, {25 + x, 35 + y}, {100 + x, 110 + y}, {200 + x, 110 + y}, {100 + x, 220 + y}, {200 + x, 220 + y}};
}

// Image tests
TEST_CASE("save load pgm")
{
    auto img = test_image();
    std::string filepath = g_image_path + "/test.pgm";

    save_img(img, filepath);
    CHECK(std::filesystem::exists(std::filesystem::path(filepath)));
    auto img_loaded = load_img(filepath);
    CHECK(img_loaded == img);
}

TEST_CASE("save load ppm")
{
    auto img = test_image_rgb();
    std::string filepath = g_image_path + "/test.ppm";

    save_img(img, filepath);
    CHECK(std::filesystem::exists(std::filesystem::path(filepath)));
    auto img_loaded = load_img(filepath);
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
            [&](const Feature &feature)
            {
                return std::abs(feature.x - x) <= tolerance &&
                       std::abs(feature.y - y) <= tolerance;
            });

        CHECK_MESSAGE(found, "Feature not detected near (", x, ", ", y, ")");
    }
}
TEST_CASE("draw features")
{
    auto img = test_image();
    Config config{};
    auto features = select_good_features(img, config);

    const auto expected = expected_features();

    REQUIRE(features.size() == expected.size());

    Image output;
    auto c = Color(255.0f,0.0f,0.0f);
    draw_features(img, output, features, c);
    std::string filepath = g_image_path + "/features.ppm";
    save_img(output, filepath);
}

TEST_CASE("real image")
{
    auto img = load_img(g_image_path + "/img0.pgm");
    Config config{};
    config.max_features = 100;
    auto features = select_good_features(img, config);

    Image output;
    auto c = Color(255.0f,0.0f,0.0f);
    draw_features(img, output, features, c);
    save_img(output, g_image_path + "/img0_features.ppm");
}


// Tracking tests

TEST_CASE("track features")
{
    constexpr float dx = 3.0f;
    constexpr float dy = 2.0f;
    constexpr float tolerance = 0.05f;

    auto img1 = test_image();
    auto img2 = test_image(3, 2);

    Config config{};

    auto features = select_good_features(img1, config);
    auto results = track_features(img1, img2, features, config);

    REQUIRE(results.size() == features.size());
    REQUIRE_FALSE(features.empty());

    for (size_t i = 0; i < features.size(); ++i)
    {
        // std::cout << "Original feature: (" << features[i].x << ", " << features[i].y << ")\n";
        // std::cout << "Tracked feature: (" << results[i].x << ", " << results[i].y << ")\n\n";

        REQUIRE(results[i].status == TrackStatus::Tracked);

        CHECK(results[i].x == doctest::Approx(features[i].x + dx).epsilon(tolerance));
        CHECK(results[i].y == doctest::Approx(features[i].y + dy).epsilon(tolerance));
    }
}