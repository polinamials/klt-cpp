#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "image.hpp"
#include "feature_extractor.hpp"
#include "tracker.hpp"
#include "orig_klt_values.hpp"

#include <string>
#include <string_view>
#include <vector>
#include <filesystem>
#include <iostream>
#include <array>

using namespace KLT;

std::string g_image_path{"images"};
std::string g_output_path{"output"};

constexpr int g_test_image_width{320};
constexpr int g_test_image_height{240};
constexpr float g_test_grey_background{255.0f};
const Color g_test_rgb_background{255.0f, 255.0f, 255.0f};

struct TestRectangle
{
    int x1;
    int y1;
    int x2;
    int y2;
    float grey;
    Color rgb;
};

const std::array<TestRectangle, 2> g_test_rectangles{{
    {40, 40, 55, 65, 0.0f, {255.0f, 0.0f, 0.0f}},
    {100, 110, 200, 200, 0.0f, {0.0f, 0.0f, 255.0f}},
}};

int main(int argc, char **argv)
{
    doctest::Context context;

    constexpr std::string_view image_prefix = "--image_path=";
    constexpr std::string_view output_prefix = "--output_path=";
    std::vector<char *> doctest_args{argv[0]};

    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if (arg.starts_with(image_prefix))
            g_image_path = arg.substr(image_prefix.size());
        else if (arg.starts_with(output_prefix))
            g_output_path = arg.substr(output_prefix.size());
        else
            doctest_args.push_back(argv[i]);
    }

    std::filesystem::create_directories(g_output_path);
    context.applyCommandLine(
        static_cast<int>(doctest_args.size()), doctest_args.data());
    return context.run();
}

// Helpers
Image test_image(const int x = 0, const int y = 0)
{
    auto img = make_image(
        g_test_image_width, g_test_image_height, g_test_grey_background);

    for (const auto &rect : g_test_rectangles)
    {
        draw_rect(
            img,
            rect.x1 + x, rect.y1 + y,
            rect.x2 + x, rect.y2 + y,
            rect.grey);
    }

    return img;
}

Image test_image_rgb(const int x = 0, const int y = 0)
{
    auto bg = g_test_rgb_background;
    auto img = make_image(g_test_image_width, g_test_image_height, bg);

    for (const auto &rect : g_test_rectangles)
    {
        draw_rect(
            img,
            rect.x1 + x, rect.y1 + y,
            rect.x2 + x, rect.y2 + y,
            rect.rgb);
    }

    return img;
}

std::vector<Feature> expected_features(int x = 0, int y = 0)
{
    std::vector<Feature> features;
    features.reserve(g_test_rectangles.size() * 4);

    for (const auto &rect : g_test_rectangles)
    {
        features.push_back({rect.x1 + x, rect.y1 + y, 0.0f});
        features.push_back({rect.x2 + x, rect.y1 + y, 0.0f});
        features.push_back({rect.x1 + x, rect.y2 + y, 0.0f});
        features.push_back({rect.x2 + x, rect.y2 + y, 0.0f});
    }

    return features;
}

const char *track_status_name(TrackStatus status)
{
    switch (status)
    {
    case TrackStatus::Tracked:
        return "Tracked";
    case TrackStatus::SmallDeterminant:
        return "SmallDeterminant";
    case TrackStatus::OutOfBounds:
        return "OutOfBounds";
    case TrackStatus::LargeResidue:
        return "LargeResidue";
    case TrackStatus::MaxIterations:
        return "MaxIterations";
    case TrackStatus::Replaced:
        return "Replaced";
    }

    return "Unknown";
}

TEST_SUITE("ConfigTests")
{
    TEST_CASE("DefaultPyramidAndBorderMatchOriginalKLT")
    {
        const Config config = create_config();

        CHECK(config.window_width == 7);
        CHECK(config.window_height == 7);
        CHECK(config.num_pyramid_levels == 2);
        CHECK(config.pyramid_subsampling == 4);
        CHECK(config.borderx == 24);
        CHECK(config.bordery == 24);
    }

    TEST_CASE("FeatureSelectionRespectsBorders")
    {
        auto img = make_image(120, 100, 255.0f);
        draw_rect(img, 5, 5, 20, 20, 0.0f);
        draw_rect(img, 40, 40, 60, 60, 0.0f);

        const Config config = create_config();
        const auto features = select_good_features(img, config);

        REQUIRE_FALSE(features.empty());
        for (const auto &feature : features)
        {
            CHECK(feature.x >= config.borderx);
            CHECK(feature.x <= img.width - 1 - config.borderx);
            CHECK(feature.y >= config.bordery);
            CHECK(feature.y <= img.height - 1 - config.bordery);
        }
    }
}

// Image tests
TEST_SUITE("ImageTests")
{
    TEST_CASE("SaveLoadPgm")
    {
        auto img = test_image();
        std::string filepath = g_output_path + "/test.pgm";

        save_img(img, filepath);
        CHECK(std::filesystem::exists(std::filesystem::path(filepath)));
        auto img_loaded = load_img(filepath);
        CHECK(img_loaded == img);
    }

    TEST_CASE("SaveLoadPpm")
    {
        auto img = test_image_rgb();
        std::string filepath = g_output_path + "/test.ppm";

        save_img(img, filepath);
        CHECK(std::filesystem::exists(std::filesystem::path(filepath)));
        auto img_loaded = load_img(filepath);
        CHECK(img_loaded == img);
    }
}

TEST_SUITE("FeatureExtractor")
{
    TEST_CASE("ExtractFeaturesSynthetic")
    {
        auto img = test_image();
        Config config{};
        auto features = select_good_features(img, config);
        const auto expected = expected_features();

        constexpr int tolerance = 3;

        REQUIRE(features.size() == expected.size());

        for (const auto &expected_feature : expected)
        {
            const bool found = std::any_of(
                features.begin(), features.end(),
                [&](const Feature &feature)
                {
                    return std::abs(feature.x - expected_feature.x) <= tolerance &&
                           std::abs(feature.y - expected_feature.y) <= tolerance;
                });

            CHECK_MESSAGE(
                found,
                "Feature not detected near (",
                expected_feature.x, ", ", expected_feature.y, ")");
        }
    }
    TEST_CASE("DrawFeatures")
    {
        auto img = test_image();
        Config config{};
        auto features = select_good_features(img, config);

        const auto expected = expected_features();

        REQUIRE(features.size() == expected.size());

        Image output;
        auto c = Color(255.0f, 0.0f, 0.0f);
        draw_features(img, output, features, c);
        std::string filepath = g_output_path + "/test_features.ppm";
        save_img(output, filepath);
    }

    TEST_CASE("RealImage")
    {
        // Compares to the values of the original
        auto img = load_img(g_image_path + "/img0.pgm");
        Config config = create_config();

        config.max_features = 100;
        auto features = select_good_features(img, config);

        REQUIRE(features.size() == orig_img0_max100.size());

        constexpr int pos_tolerance = 1;
        constexpr float score_tolerance = 1.0f;
        std::vector<bool> matched(features.size(), false);

        for (size_t i = 0; i < orig_img0_max100.size(); ++i)
        {
            const auto &expected = orig_img0_max100[i];
            bool found = false;

            for (size_t j = 0; j < features.size(); ++j)
            {
                if (matched[j])
                    continue;

                const auto &actual = features[j];
                const bool matches =
                    std::abs(actual.x - expected.x) <= pos_tolerance &&
                    std::abs(actual.y - expected.y) <= pos_tolerance &&
                    std::abs(actual.score - expected.score) <= score_tolerance;

                if (matches)
                {
                    // std::cout
                    //     << "[" << i << "] Original: ("
                    //     << expected.x << ", " << expected.y << ", "
                    //     << expected.score << "), Current: "
                    //     << actual.x << ", " << actual.y << ", "
                    //     << actual.score << ")\n"
                    //     << std::endl;

                    matched[j] = true;
                    found = true;
                    break;
                }
            }

            REQUIRE_MESSAGE(
                found,
                "No match for original feature ", i,
                " at (", expected.x, ", ", expected.y,
                ") with score ", expected.score);
        }

        Image output;
        auto c = Color(255.0f, 0.0f, 0.0f);
        draw_features(img, output, features, c);
        save_img(output, g_output_path + "/img0_features.ppm");
    }
}

TEST_SUITE("Tracking")
{
    TEST_CASE("TrackFeatures")
    {
        constexpr float dx = 3.0f;
        constexpr float dy = 2.0f;
        constexpr float tolerance = 0.05f;

        auto img0 = test_image();
        auto img1 = test_image(3, 2);

        Config config{};

        auto features = select_good_features(img0, config);
        auto results = track_features(img0, img1, features, config);

        REQUIRE(results.size() == features.size());
        REQUIRE_FALSE(features.empty());

        for (size_t i = 0; i < features.size(); ++i)
        {
            // std::cout << "Original feature: (" << features[i].x << ", " << features[i].y << ")\n";
            // std::cout << "Tracked feature: (" << results[i].x << ", " << results[i].y << ")\n\n";

            REQUIRE(results[i].status == TrackStatus::Tracked);

            CHECK(std::abs(results[i].x - (features[i].x + dx)) <= tolerance);
            CHECK(std::abs(results[i].y - (features[i].y + dy)) <= tolerance);
        }
    }

    TEST_CASE("TrackReal")
    {
        auto img0 = load_img(g_image_path + "/img0.pgm");
        auto img1 = load_img(g_image_path + "/img1.pgm");
        Config config = create_config();

        config.max_features = 100;
        auto features = select_good_features(img0, config);
        auto results = track_features(img0, img1, features, config);

        REQUIRE(features.size() == orig_img0_max100.size());
        REQUIRE(results.size() == orig_img1_tracked.size());
        REQUIRE_FALSE(features.empty());

        constexpr int feature_position_tolerance = 1;
        constexpr float feature_score_tolerance = 1.0f;
        constexpr float tracked_position_tolerance = 0.1f;
        std::vector<bool> matched(features.size(), false);

        for (size_t i = 0; i < orig_img0_max100.size(); ++i)
        {
            const auto &original_feature = orig_img0_max100[i];
            size_t current_index = features.size();

            for (size_t j = 0; j < features.size(); ++j)
            {
                if (matched[j])
                    continue;

                const auto &current_feature = features[j];
                const bool matches =
                    std::abs(current_feature.x - original_feature.x) <=
                        feature_position_tolerance &&
                    std::abs(current_feature.y - original_feature.y) <=
                        feature_position_tolerance &&
                    std::abs(current_feature.score - original_feature.score) <=
                        feature_score_tolerance;

                if (matches)
                {
                    current_index = j;
                    matched[j] = true;
                    break;
                }
            }

            REQUIRE_MESSAGE(
                current_index != features.size(),
                "No current img0 feature matches original feature ", i,
                " at (", original_feature.x, ", ", original_feature.y,
                ") with score ", original_feature.score);

            const auto &expected = orig_img1_tracked[i];
            const auto &actual = results[current_index];

            // std::cout
            //     << "[" << i << "] Original: ("
            //     << expected.x << ", " << expected.y << ", "
            //     << track_status_name(expected.status) << "), Current["
            //     << current_index << "]: ("
            //     << actual.x << ", " << actual.y << ", "
            //     << track_status_name(actual.status) << ")\n"
            //     << std::endl;

            REQUIRE_MESSAGE(
                actual.status == expected.status,
                "Tracking status differs for original feature ", i,
                ": expected ", track_status_name(expected.status),
                ", got ", track_status_name(actual.status));
            CHECK_MESSAGE(
                std::abs(actual.x - expected.x) <= tracked_position_tolerance,
                "Tracked x differs for original feature ", i);
            CHECK_MESSAGE(
                std::abs(actual.y - expected.y) <= tracked_position_tolerance,
                "Tracked y differs for original feature ", i);
        }
    }

    TEST_CASE("TrackedFeaturesOutsideBorder")
    {
        auto img = make_image(120, 100, 255.0f);
        draw_rect(img, 20, 40, 40, 60, 0.0f);

        const Config config = create_config();
        const std::vector<Feature> features{{20, 40, 1.0f}};
        const auto results = track_features(img, img, features, config);

        REQUIRE(results.size() == 1);
        CHECK(results[0].status == TrackStatus::OutOfBounds);
        CHECK(results[0].x == -1.0f);
        CHECK(results[0].y == -1.0f);
    }

    TEST_CASE("TrackMultiple")
    {
        const std::vector<Image> imgs{
            test_image(),
            test_image(3, 2),
            test_image(-1, 1),
            test_image(2, 4),
            test_image(0, -1)};
        const std::vector<std::vector<Feature>> expected{
            expected_features(),
            expected_features(3, 2),
            expected_features(-1, 1),
            expected_features(2, 4),
            expected_features(0, -1)};
        const Config config = create_config();
        const auto tracks = track(imgs, config);

        constexpr float initial_position_tolerance = 3.0f;
        constexpr float tracked_position_tolerance = 0.1f;

        REQUIRE(tracks.size() == expected.front().size());
        std::vector<bool> matched(expected.front().size(), false);

        for (size_t track_index = 0; track_index < tracks.size(); ++track_index)
        {
            const auto &positions = tracks[track_index].positions;
            REQUIRE(positions.size() == imgs.size());

            size_t expected_index = expected.front().size();
            for (size_t i = 0; i < expected.front().size(); ++i)
            {
                if (matched[i])
                    continue;

                const auto &initial = positions.front();
                const auto &expected_initial = expected.front()[i];
                if (std::abs(initial.x - expected_initial.x) <=
                        initial_position_tolerance &&
                    std::abs(initial.y - expected_initial.y) <=
                        initial_position_tolerance)
                {
                    expected_index = i;
                    matched[i] = true;
                    break;
                }
            }

            REQUIRE_MESSAGE(
                expected_index != expected.front().size(),
                "No synthetic corner matches track ", track_index,
                " starting at (", positions.front().x, ", ",
                positions.front().y, ")");

            const float corner_offset_x =
                positions.front().x - expected.front()[expected_index].x;
            const float corner_offset_y =
                positions.front().y - expected.front()[expected_index].y;

            for (size_t frame = 0; frame < imgs.size(); ++frame)
            {
                const auto &actual = positions[frame];
                const auto &expected_position = expected[frame][expected_index];
                const float expected_x = expected_position.x + corner_offset_x;
                const float expected_y = expected_position.y + corner_offset_y;

                // std::cout
                //     << "[Track " << track_index << ", frame " << frame
                //     << "] Expected: (" << expected_x << ", " << expected_y
                //     << ", Tracked), Actual: (" << actual.x << ", "
                //     << actual.y << ", " << track_status_name(actual.status)
                //     << ")\n";

                REQUIRE_MESSAGE(
                    actual.status == TrackStatus::Tracked,
                    "Track ", track_index, " was not tracked in frame ", frame,
                    ": ", track_status_name(actual.status));
                CHECK_MESSAGE(
                    std::abs(actual.x - expected_x) <=
                        tracked_position_tolerance,
                    "Track ", track_index, " has incorrect x in frame ", frame);
                CHECK_MESSAGE(
                    std::abs(actual.y - expected_y) <=
                        tracked_position_tolerance,
                    "Track ", track_index, " has incorrect y in frame ", frame);
            }
        }
    }
}
