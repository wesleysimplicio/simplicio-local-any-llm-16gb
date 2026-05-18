#include "core/hardware_probe.h"

#include <cstdlib>
#include <string_view>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace us4 {

namespace {

bool ReadBoolEnv(const char *name, const bool fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr) {
    return fallback;
  }
  return std::string_view(value) == "1" || std::string_view(value) == "true" ||
         std::string_view(value) == "TRUE";
}

unsigned int ReadUnsignedEnv(const char *name, const unsigned int fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr) {
    return fallback;
  }
  return static_cast<unsigned int>(std::strtoul(value, nullptr, 10));
}

unsigned long long DetectMemoryGiB() {
#if defined(__APPLE__)
  std::uint64_t memory_bytes = 0;
  size_t size = sizeof(memory_bytes);
  if (sysctlbyname("hw.memsize", &memory_bytes, &size, nullptr, 0) == 0 &&
      memory_bytes > 0) {
    return memory_bytes / (1024ULL * 1024ULL * 1024ULL);
  }
#endif
  const char *from_env = std::getenv("US4_MEMORY_GIB");
  if (from_env != nullptr) {
    return std::strtoull(from_env, nullptr, 10);
  }
  return 16ULL;
}

std::string DetectPlatform() {
#if defined(_WIN32)
  return "windows";
#elif defined(__APPLE__)
  return "apple";
#elif defined(__linux__)
  return "linux";
#else
  return "unknown";
#endif
}

std::string DetectArchitecture() {
#if defined(__aarch64__) || defined(_M_ARM64)
  return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
  return "x64";
#else
  return "unknown";
#endif
}

std::string DetectChip(bool is_apple_silicon) {
#if defined(__APPLE__) && defined(__aarch64__)
  char buffer[256] = {};
  size_t size = sizeof(buffer);
  if (sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nullptr, 0) ==
          0 &&
      buffer[0] != '\0') {
    return std::string(buffer);
  }
  return "apple-silicon";
#else
  if (is_apple_silicon) {
    return "apple-silicon";
  }
  return "generic-host";
#endif
}

bool DetectAneSupport(const bool isAppleSilicon, const std::string &chip) {
  if (ReadBoolEnv("US4_HAS_ANE", false)) {
    return true;
  }
  if (!isAppleSilicon) {
    return false;
  }
  return chip.find("M5") != std::string::npos ||
         chip.find("apple-m5") != std::string::npos;
}

} // namespace

HardwareProbeResult HardwareProbe::Detect() {
  HardwareProbeResult result;
  result.platform = DetectPlatform();
  result.architecture = DetectArchitecture();
  result.isAppleSilicon =
      (result.platform == "apple" && result.architecture == "arm64");
  result.unifiedMemoryGiB = DetectMemoryGiB();
  result.hasMlx = result.isAppleSilicon;
  result.hasMetal = result.isAppleSilicon;
  result.hasNeon = (result.architecture == "arm64");
  result.neonVectorBits =
      ReadUnsignedEnv("US4_NEON_VECTOR_BITS", result.hasNeon ? 128U : 0U);
  result.hasPerformanceCores =
      ReadBoolEnv("US4_HAS_PERFORMANCE_CORES", result.isAppleSilicon);
  result.hasEfficiencyCores =
      ReadBoolEnv("US4_HAS_EFFICIENCY_CORES", result.isAppleSilicon);
  result.chip = DetectChip(result.isAppleSilicon);
  result.hasAne = DetectAneSupport(result.isAppleSilicon, result.chip);
  result.supportsCoreMl =
      result.hasAne || ReadBoolEnv("US4_SUPPORTS_COREML", false);
  result.recommendedMode =
      SelectRuntimeModeFromMemoryGiB(result.unifiedMemoryGiB);
  return result;
}

} // namespace us4
