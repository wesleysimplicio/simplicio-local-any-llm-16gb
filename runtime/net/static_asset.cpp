#include "net/static_asset.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <system_error>

namespace us4 {

namespace {

bool IsWithin(const std::filesystem::path &root,
              const std::filesystem::path &candidate) {
  auto rootPart = root.begin();
  auto candidatePart = candidate.begin();
  while (rootPart != root.end()) {
    if (candidatePart == candidate.end() || *rootPart != *candidatePart) {
      return false;
    }
    ++rootPart;
    ++candidatePart;
  }
  return true;
}

std::string ContentType(const std::filesystem::path &path) {
  const std::string extension = path.extension().string();
  if (extension == ".html") return "text/html; charset=utf-8";
  if (extension == ".js") return "text/javascript; charset=utf-8";
  if (extension == ".css") return "text/css; charset=utf-8";
  if (extension == ".json") return "application/json";
  if (extension == ".svg") return "image/svg+xml";
  if (extension == ".png") return "image/png";
  if (extension == ".ico") return "image/x-icon";
  return "application/octet-stream";
}

} // namespace

std::optional<StaticAsset>
LoadStaticAsset(const std::string &webRoot, const std::string &requestPath,
                const std::size_t maxBytes) {
  if (webRoot.empty() || requestPath.empty() || maxBytes == 0U ||
      requestPath.find('\\') != std::string::npos ||
      requestPath.find('%') != std::string::npos ||
      requestPath.find('\0') != std::string::npos) {
    return std::nullopt;
  }
  const std::size_t query = requestPath.find('?');
  std::string relative = requestPath.substr(0, query);
  if (relative == "/") relative = "/index.html";
  if (relative.front() != '/' || relative.find("..") != std::string::npos) {
    return std::nullopt;
  }
  relative.erase(0, 1);

  std::error_code error;
  const std::filesystem::path root =
      std::filesystem::weakly_canonical(webRoot, error);
  if (error || !std::filesystem::is_directory(root, error)) {
    return std::nullopt;
  }
  std::filesystem::path candidate =
      std::filesystem::weakly_canonical(root / relative, error);
  if (error || !IsWithin(root, candidate) ||
      !std::filesystem::is_regular_file(candidate, error)) {
    return std::nullopt;
  }
  const std::uintmax_t size = std::filesystem::file_size(candidate, error);
  if (error || size > maxBytes) {
    return std::nullopt;
  }
  std::ifstream input(candidate, std::ios::binary);
  if (!input) return std::nullopt;
  std::string body((std::istreambuf_iterator<char>(input)),
                   std::istreambuf_iterator<char>());
  if (body.size() != size) return std::nullopt;
  return StaticAsset{ContentType(candidate), std::move(body)};
}

} // namespace us4
