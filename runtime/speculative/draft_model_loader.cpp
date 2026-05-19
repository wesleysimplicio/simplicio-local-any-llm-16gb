#include "speculative/draft_model_loader.h"

#include <cctype>
#include <sstream>
#include <string_view>

namespace us4 {

namespace {

std::string Trim(std::string_view value) {
  std::size_t begin = 0;
  std::size_t end = value.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return std::string(value.substr(begin, end - begin));
}

} // namespace

std::optional<DraftModelDescriptor>
ParseDraftModelManifestBody(const std::string &manifestBody) {
  DraftModelDescriptor descriptor;
  std::istringstream stream(manifestBody);
  std::string line;
  while (std::getline(stream, line)) {
    const auto trimmed = Trim(line);
    if (trimmed.empty() || trimmed.front() == '#') {
      continue;
    }
    const auto eq = trimmed.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    const std::string key = Trim(trimmed.substr(0, eq));
    const std::string value = Trim(trimmed.substr(eq + 1));
    if (key == "family") {
      descriptor.family = value;
    } else if (key == "model_id") {
      descriptor.modelId = value;
    } else if (key == "tokenizer_hash") {
      descriptor.tokenizerHash = value;
    } else if (key == "file") {
      descriptor.filePath = value;
    } else if (key == "weight_format") {
      descriptor.weightFormat = value;
    }
  }
  if (descriptor.family.empty() || descriptor.modelId.empty() ||
      descriptor.tokenizerHash.empty() || descriptor.filePath.empty() ||
      descriptor.weightFormat.empty()) {
    return std::nullopt;
  }
  return descriptor;
}

} // namespace us4
