#pragma once

#include <optional>
#include <string>

namespace us4 {

// DraftModelLoader contract.
//
// A draft model is a small companion model used for speculative decoding. The
// loader needs to keep the tokenizer assumption explicit so the verifier can
// trust that draft tokens map cleanly into the target model's vocabulary.

struct DraftModelDescriptor {
  std::string family;
  std::string modelId;
  std::string tokenizerHash;
  std::string filePath;
  std::string weightFormat;
};

// Parse a draft-model manifest body. The contract is intentionally simple so
// tests can exercise the schema without touching disk.
std::optional<DraftModelDescriptor>
ParseDraftModelManifestBody(const std::string &manifestBody);

}  // namespace us4
