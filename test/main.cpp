#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "shi_tomasi.hpp"
#include "image.hpp"

TEST_CASE("create image")
{
    auto img = make_image(360, 240, 255.0);
    draw_rect(img, 10, 10, 25, 35, 0);
    draw_rect(img, 100, 110, 200, 220, 0);

    Config config{};
    auto result = select_good_features(img, config);

    CHECK_FALSE(result.empty());
    CHECK(result.size() == 8);
}
