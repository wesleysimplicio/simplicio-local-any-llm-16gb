#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace us4 {

class SsdColdStore {
 public:
  explicit SsdColdStore(std::filesystem::path root = "build/kv-cold-store");

  bool Flush(const std::string& key, const std::vector<float>& values);
  std::optional<std::vector<float>> Restore(const std::string& key) const;

 private:
  std::filesystem::path root_;
};

}  // namespace us4
