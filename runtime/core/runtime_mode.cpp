#include "core/runtime_mode.h"

#include <cctype>

namespace us4 {

namespace {

constexpr unsigned long long kOneGiB = 1024ULL * 1024ULL * 1024ULL;

std::string Normalize(std::string_view value) {
  std::string normalized;
  normalized.reserve(value.size());
  for (char ch : value) {
    if (ch == '-' || ch == ' ') {
      normalized.push_back('_');
    } else {
      normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
  }
  return normalized;
}

}  // namespace

std::string_view ToString(RuntimeMode mode) {
  switch (mode) {
    case RuntimeMode::kFull:
      return "FULL";
    case RuntimeMode::kBalancedPlus:
      return "BALANCED_PLUS";
    case RuntimeMode::kDegraded:
      return "DEGRADED";
    case RuntimeMode::kUltraLow:
      return "ULTRA_LOW";
    case RuntimeMode::kMicro:
      return "MICRO";
    case RuntimeMode::kMicroPlus:
      return "MICRO_PLUS";
    case RuntimeMode::kNano:
      return "NANO";
  }
  return "NANO";
}

std::optional<RuntimeMode> ParseRuntimeMode(std::string_view value) {
  const std::string normalized = Normalize(value);
  if (normalized == "full") {
    return RuntimeMode::kFull;
  }
  if (normalized == "balanced_plus" || normalized == "balancedplus") {
    return RuntimeMode::kBalancedPlus;
  }
  if (normalized == "degraded") {
    return RuntimeMode::kDegraded;
  }
  if (normalized == "ultra_low" || normalized == "ultralow") {
    return RuntimeMode::kUltraLow;
  }
  if (normalized == "micro") {
    return RuntimeMode::kMicro;
  }
  if (normalized == "micro_plus" || normalized == "microplus") {
    return RuntimeMode::kMicroPlus;
  }
  if (normalized == "nano") {
    return RuntimeMode::kNano;
  }
  return std::nullopt;
}

RuntimeMode SelectRuntimeModeFromMemoryGiB(unsigned long long memory_gib) {
  if (memory_gib >= 128ULL) {
    return RuntimeMode::kFull;
  }
  if (memory_gib >= 96ULL) {
    return RuntimeMode::kBalancedPlus;
  }
  if (memory_gib >= 64ULL) {
    return RuntimeMode::kDegraded;
  }
  if (memory_gib >= 48ULL) {
    return RuntimeMode::kUltraLow;
  }
  if (memory_gib >= 32ULL) {
    return RuntimeMode::kMicro;
  }
  if (memory_gib >= 24ULL) {
    return RuntimeMode::kMicroPlus;
  }
  return RuntimeMode::kNano;
}

int RuntimeModeRank(const RuntimeMode mode) {
  switch (mode) {
    case RuntimeMode::kNano:
      return 0;
    case RuntimeMode::kMicroPlus:
      return 1;
    case RuntimeMode::kMicro:
      return 2;
    case RuntimeMode::kUltraLow:
      return 3;
    case RuntimeMode::kDegraded:
      return 4;
    case RuntimeMode::kBalancedPlus:
      return 5;
    case RuntimeMode::kFull:
      return 6;
  }
  return 0;
}

RuntimeMode MaxRuntimeMode(const RuntimeMode lhs, const RuntimeMode rhs) {
  return RuntimeModeRank(lhs) >= RuntimeModeRank(rhs) ? lhs : rhs;
}

}  // namespace us4
