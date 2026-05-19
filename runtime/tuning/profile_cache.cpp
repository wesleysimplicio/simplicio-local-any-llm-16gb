#include "tuning/profile_cache.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string_view>
#include <vector>

namespace us4 {

bool operator==(const ProfileCacheKey &lhs, const ProfileCacheKey &rhs) {
  return lhs.chip == rhs.chip && lhs.modelId == rhs.modelId;
}

std::size_t
ProfileCacheKeyHash::operator()(const ProfileCacheKey &key) const noexcept {
  const std::size_t chipHash = std::hash<std::string>{}(key.chip);
  const std::size_t modelHash = std::hash<std::string>{}(key.modelId);
  return chipHash ^ (modelHash * 131U);
}

void ProfileCache::Store(const ProfileCacheKey &key,
                         const AutoTunerProfile &profile) {
  profiles_[key] = profile;
}

std::optional<AutoTunerProfile>
ProfileCache::Lookup(const ProfileCacheKey &key) const {
  const auto it = profiles_.find(key);
  if (it == profiles_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::size_t ProfileCache::Size() const { return profiles_.size(); }

std::string ProfileCache::Serialize() const {
  std::vector<std::pair<ProfileCacheKey, AutoTunerProfile>> entries(
      profiles_.begin(), profiles_.end());
  std::sort(entries.begin(), entries.end(),
            [](const std::pair<ProfileCacheKey, AutoTunerProfile> &lhs,
               const std::pair<ProfileCacheKey, AutoTunerProfile> &rhs) {
              if (lhs.first.chip != rhs.first.chip) {
                return lhs.first.chip < rhs.first.chip;
              }
              return lhs.first.modelId < rhs.first.modelId;
            });
  std::ostringstream stream;
  for (const auto &entry : entries) {
    stream << "chip=" << entry.first.chip
           << ";model=" << entry.first.modelId
           << ";tile_rows=" << entry.second.tileRows
           << ";tile_cols=" << entry.second.tileCols
           << ";batch=" << entry.second.batchSize
           << ";latency_ms=" << entry.second.estimatedLatencyMs << "\n";
  }
  return stream.str();
}

bool ProfileCache::Load(const std::string &body) {
  std::istringstream stream(body);
  std::string line;
  std::unordered_map<ProfileCacheKey, AutoTunerProfile, ProfileCacheKeyHash>
      next;
  while (std::getline(stream, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    ProfileCacheKey key;
    AutoTunerProfile profile;
    std::istringstream parts(line);
    std::string part;
    while (std::getline(parts, part, ';')) {
      const auto eq = part.find('=');
      if (eq == std::string::npos) {
        continue;
      }
      const std::string name = part.substr(0, eq);
      const std::string value = part.substr(eq + 1);
      if (name == "chip") {
        key.chip = value;
        profile.chip = value;
      } else if (name == "model") {
        key.modelId = value;
      } else if (name == "tile_rows") {
        profile.tileRows = static_cast<std::size_t>(std::strtoul(value.c_str(), nullptr, 10));
      } else if (name == "tile_cols") {
        profile.tileCols = static_cast<std::size_t>(std::strtoul(value.c_str(), nullptr, 10));
      } else if (name == "batch") {
        profile.batchSize = static_cast<std::size_t>(std::strtoul(value.c_str(), nullptr, 10));
      } else if (name == "latency_ms") {
        profile.estimatedLatencyMs = std::strtof(value.c_str(), nullptr);
      }
    }
    if (key.chip.empty() || key.modelId.empty()) {
      return false;
    }
    next[key] = profile;
  }
  profiles_ = std::move(next);
  return true;
}

} // namespace us4
