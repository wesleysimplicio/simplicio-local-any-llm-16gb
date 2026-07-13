#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace us4 {

// Real byte-pair-encoding tokenizer, loaded from a HuggingFace `tokenizers`
// style tokenizer.json (model.type == "BPE", model.vocab, model.merges).
// This replaces the naive whitespace/lowercase splitter for any model asset
// that ships a genuine vocab+merges tokenizer.json.
class BpeTokenizer {
public:
  // Returns std::nullopt if `path` does not exist or does not contain a real
  // BPE vocab+merges definition (e.g. a placeholder/manifest-only file).
  // `error`, when provided, receives a human-readable reason for the miss so
  // callers can surface an explicit fallback instead of a silent one.
  static std::optional<BpeTokenizer>
  LoadFromFile(const std::filesystem::path &path, std::string *error = nullptr);

  // Splits `text` on whitespace, then BPE-merges each word using the loaded
  // merge ranks, returning the resulting subword token strings in order.
  std::vector<std::string> Encode(std::string_view text) const;

  // Same as Encode, but returns vocabulary ids. Tokens absent from the vocab
  // resolve to the tokenizer's <unk> id when one is configured.
  std::vector<int> EncodeIds(std::string_view text) const;

  std::size_t VocabSize() const { return vocab_.size(); }
  bool HasToken(const std::string &token) const {
    return vocab_.count(token) > 0;
  }

private:
  std::unordered_map<std::string, int> vocab_;
  // merge_rank_[{a,b}] = priority (lower merges first), mirrors the order
  // merges appear in tokenizer.json, exactly like the reference BPE model.
  std::unordered_map<std::string, int> mergeRank_;
  std::optional<int> unkId_;

  static std::string PairKey(const std::string &a, const std::string &b) {
    return a + "\x1f" + b;
  }

  std::vector<std::string> BpeMergeWord(const std::string &word) const;
};

} // namespace us4
