#include "core/safetensors_reader.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

#include "core/json_value.h"

namespace us4 {

namespace {
// A JSON number is a double; safetensors offsets/shapes are logically
// non-negative integers. Reject NaN/Inf/negative/out-of-range values here
// instead of letting a malformed or adversarial header silently produce
// undefined behavior at the static_cast<size_t> call site.
bool SafeDoubleToSize(const double value, std::size_t *out) {
  if (!std::isfinite(value) || value < 0.0 ||
      value > static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
    return false;
  }
  *out = static_cast<std::size_t>(value);
  return true;
}
} // namespace

std::optional<SafetensorsReader>
SafetensorsReader::Open(const std::filesystem::path &path, std::string *error) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    if (error != nullptr) {
      *error = "safetensors file not found at " + path.string();
    }
    return std::nullopt;
  }

  std::uint64_t headerLength = 0;
  file.read(reinterpret_cast<char *>(&headerLength), sizeof(headerLength));
  if (!file || headerLength == 0 || headerLength > (1ULL << 30)) {
    if (error != nullptr) {
      *error = "safetensors file has no valid 8-byte header length prefix "
               "(placeholder/non-binary file, not real tensor data)";
    }
    return std::nullopt;
  }

  std::string headerJson(headerLength, '\0');
  file.read(headerJson.data(), static_cast<std::streamsize>(headerLength));
  if (!file) {
    if (error != nullptr) {
      *error = "safetensors header is truncated";
    }
    return std::nullopt;
  }

  JsonValue header;
  try {
    header = JsonValue::Parse(headerJson);
  } catch (const std::exception &ex) {
    if (error != nullptr) {
      *error = std::string("failed to parse safetensors header: ") + ex.what();
    }
    return std::nullopt;
  }

  if (!header.IsObject()) {
    if (error != nullptr) {
      *error = "safetensors header is not a JSON object";
    }
    return std::nullopt;
  }

  SafetensorsReader reader;
  reader.path_ = path;
  reader.dataSectionStart_ = sizeof(headerLength) + headerLength;

  for (const auto &[name, entry] : header.Entries()) {
    if (name == "__metadata__" || !entry.IsObject()) {
      continue;
    }
    TensorInfo info;
    if (entry.Has("dtype") && entry["dtype"].IsString()) {
      info.dtype = entry["dtype"].AsString();
    }
    bool shapeValid = true;
    if (entry.Has("shape") && entry["shape"].IsArray()) {
      for (const JsonValue &dim : entry["shape"].AsArray()) {
        std::size_t dimValue = 0;
        if (!SafeDoubleToSize(dim.AsNumber(), &dimValue)) {
          shapeValid = false;
          break;
        }
        info.shape.push_back(dimValue);
      }
    }
    bool offsetsValid = false;
    if (shapeValid && entry.Has("data_offsets") &&
        entry["data_offsets"].IsArray() &&
        entry["data_offsets"].AsArray().size() == 2) {
      std::size_t begin = 0;
      std::size_t end = 0;
      if (SafeDoubleToSize(entry["data_offsets"].AsArray()[0].AsNumber(),
                           &begin) &&
          SafeDoubleToSize(entry["data_offsets"].AsArray()[1].AsNumber(),
                           &end) &&
          end >= begin) {
        info.byteOffsetBegin = begin;
        info.byteOffsetEnd = end;
        offsetsValid = true;
      }
    }
    if (!shapeValid || !offsetsValid) {
      continue; // drop tensors with malformed/adversarial metadata instead
                // of carrying invalid offsets forward into ReadFloat32.
    }
    reader.tensors_.emplace(name, std::move(info));
  }

  if (reader.tensors_.empty()) {
    if (error != nullptr) {
      *error = "safetensors header contains no real tensor entries";
    }
    return std::nullopt;
  }

  return reader;
}

namespace {

// bfloat16 is the upper 16 bits of an IEEE-754 float32 (sign + 8-bit
// exponent + 7 mantissa bits, with the low 16 mantissa bits truncated).
// Widening back to float32 is exact for the bits it kept -- just place
// them in the high half and zero the rest.
float Bf16ToFloat32(const std::uint16_t bits) {
  std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &widened, sizeof(value));
  return value;
}

} // namespace

std::vector<float> SafetensorsReader::ReadFloat32(const std::string &name,
                                                  std::string *error) const {
  const TensorInfo *info = Find(name);
  if (info == nullptr) {
    if (error != nullptr) {
      *error = "tensor not found: " + name;
    }
    return {};
  }
  const bool isBf16 = info->dtype == "BF16";
  if (info->dtype != "F32" && !isBf16) {
    if (error != nullptr) {
      *error =
          "tensor " + name + " is not F32/BF16 (dtype=" + info->dtype + ")";
    }
    return {};
  }

  const std::size_t elementSize =
      isBf16 ? sizeof(std::uint16_t) : sizeof(float);
  const std::size_t byteLength = info->byteOffsetEnd - info->byteOffsetBegin;
  if (byteLength % elementSize != 0) {
    if (error != nullptr) {
      *error = "tensor " + name + " byte length is not " +
               (isBf16 ? "bfloat16" : "float32") + "-aligned";
    }
    return {};
  }

  std::error_code fileSizeError;
  const std::uintmax_t fileSize =
      std::filesystem::file_size(path_, fileSizeError);
  if (fileSizeError || dataSectionStart_ + info->byteOffsetEnd > fileSize) {
    if (error != nullptr) {
      *error =
          "tensor " + name + " data_offsets extend past the end of the file";
    }
    return {};
  }

  std::ifstream file(path_, std::ios::binary);
  if (!file) {
    if (error != nullptr) {
      *error = "unable to reopen safetensors file " + path_.string();
    }
    return {};
  }
  file.seekg(
      static_cast<std::streamoff>(dataSectionStart_ + info->byteOffsetBegin));

  const std::size_t elementCount = byteLength / elementSize;
  if (!isBf16) {
    std::vector<float> values(elementCount);
    file.read(reinterpret_cast<char *>(values.data()),
              static_cast<std::streamsize>(byteLength));
    if (!file) {
      if (error != nullptr) {
        *error = "tensor " + name + " data is truncated in file body";
      }
      return {};
    }
    return values;
  }

  std::vector<std::uint16_t> rawBits(elementCount);
  file.read(reinterpret_cast<char *>(rawBits.data()),
            static_cast<std::streamsize>(byteLength));
  if (!file) {
    if (error != nullptr) {
      *error = "tensor " + name + " data is truncated in file body";
    }
    return {};
  }
  std::vector<float> values(elementCount);
  for (std::size_t index = 0; index < elementCount; ++index) {
    values[index] = Bf16ToFloat32(rawBits[index]);
  }
  return values;
}

} // namespace us4
