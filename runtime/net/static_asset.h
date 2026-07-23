#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace us4 {

struct StaticAsset {
  std::string contentType;
  std::string body;
};

std::optional<StaticAsset>
LoadStaticAsset(const std::string &webRoot, const std::string &requestPath,
                std::size_t maxBytes = 8U * 1024U * 1024U);

} // namespace us4
