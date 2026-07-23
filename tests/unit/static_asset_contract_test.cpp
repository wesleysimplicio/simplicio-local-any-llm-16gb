#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "net/static_asset.h"

namespace {

class StaticWebRoot {
public:
  StaticWebRoot() {
    root_ = std::filesystem::temp_directory_path() / "us4-static-web-test";
    std::filesystem::create_directories(root_ / "assets");
    std::ofstream(root_ / "index.html") << "<main>US4</main>";
    std::ofstream(root_ / "assets" / "app.js") << "export default 1";
  }
  ~StaticWebRoot() { std::filesystem::remove_all(root_); }
  std::string Path() const { return root_.string(); }

private:
  std::filesystem::path root_;
};

} // namespace

TEST(StaticAssetContractTest, ServesIndexAndTypedAssets) {
  StaticWebRoot root;
  const auto index = us4::LoadStaticAsset(root.Path(), "/");
  const auto script =
      us4::LoadStaticAsset(root.Path(), "/assets/app.js?version=1");
  ASSERT_TRUE(index.has_value());
  ASSERT_TRUE(script.has_value());
  EXPECT_EQ(index->contentType, "text/html; charset=utf-8");
  EXPECT_EQ(script->contentType, "text/javascript; charset=utf-8");
}

TEST(StaticAssetContractTest, RejectsTraversalAndOversizedFiles) {
  StaticWebRoot root;
  EXPECT_FALSE(us4::LoadStaticAsset(root.Path(), "/../secret").has_value());
  EXPECT_FALSE(us4::LoadStaticAsset(root.Path(), "/%2e%2e/secret").has_value());
  EXPECT_FALSE(us4::LoadStaticAsset(root.Path(), "/index.html", 2).has_value());
}
