#include "core/bpe_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>

#include "core/json_value.h"

namespace us4 {

namespace {

std::vector<std::string> SplitWhitespace(std::string_view text) {
  std::vector<std::string> words;
  std::string current;
  for (const char c : text) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      if (!current.empty()) {
        words.push_back(current);
        current.clear();
      }
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) {
    words.push_back(current);
  }
  return words;
}

} // namespace

std::optional<BpeTokenizer>
BpeTokenizer::LoadFromFile(const std::filesystem::path &path,
                           std::string *error) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    if (error != nullptr) {
      *error = "tokenizer.json not found at " + path.string();
    }
    return std::nullopt;
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();

  JsonValue root;
  try {
    root = JsonValue::Parse(buffer.str());
  } catch (const std::exception &ex) {
    if (error != nullptr) {
      *error = std::string("failed to parse tokenizer.json: ") + ex.what();
    }
    return std::nullopt;
  }

  if (!root.IsObject() || !root.Has("model") || !root["model"].IsObject()) {
    if (error != nullptr) {
      *error = "tokenizer.json has no model.vocab/model.merges (placeholder, "
               "not a real BPE definition)";
    }
    return std::nullopt;
  }

  const JsonValue &model = root["model"];
  if (!model.Has("type") || model["type"].AsString() != "BPE" ||
      !model.Has("vocab") || !model["vocab"].IsObject() ||
      !model.Has("merges") || !model["merges"].IsArray()) {
    if (error != nullptr) {
      *error = "tokenizer.json model is not a real BPE vocab+merges "
               "definition";
    }
    return std::nullopt;
  }

  BpeTokenizer tokenizer;

  const JsonValue &vocabObject = model["vocab"];
  for (const auto &[key, value] : vocabObject.Entries()) {
    tokenizer.vocab_[key] = static_cast<int>(value.AsNumber());
  }

  int rank = 0;
  for (const JsonValue &mergeEntry : model["merges"].AsArray()) {
    std::string mergeText;
    if (mergeEntry.IsString()) {
      mergeText = mergeEntry.AsString();
    } else if (mergeEntry.IsArray() && mergeEntry.AsArray().size() == 2) {
      mergeText = mergeEntry.AsArray()[0].AsString() + " " +
                  mergeEntry.AsArray()[1].AsString();
    }
    const std::size_t spacePos = mergeText.find(' ');
    if (spacePos == std::string::npos) {
      continue;
    }
    const std::string left = mergeText.substr(0, spacePos);
    const std::string right = mergeText.substr(spacePos + 1);
    tokenizer.mergeRank_[PairKey(left, right)] = rank++;
  }

  if (model.Has("unk_token") && model["unk_token"].IsString()) {
    const std::string unkToken = model["unk_token"].AsString();
    auto it = tokenizer.vocab_.find(unkToken);
    if (it != tokenizer.vocab_.end()) {
      tokenizer.unkId_ = it->second;
    }
  }

  if (root.Has("chat_template") && root["chat_template"].IsString()) {
    tokenizer.chatTemplate_ = root["chat_template"].AsString();
  }

  if (root.Has("added_tokens") && root["added_tokens"].IsArray()) {
    for (const JsonValue &entry : root["added_tokens"].AsArray()) {
      if (!entry.IsObject() || !entry.Has("content") ||
          !entry["content"].IsString()) {
        continue;
      }
      const std::string token = entry["content"].AsString();
      if (entry.Has("id") && entry["id"].type() == JsonValue::Type::kNumber) {
        tokenizer.vocab_[token] = static_cast<int>(entry["id"].AsNumber());
      }
      if (entry.Has("special") &&
          entry["special"].type() == JsonValue::Type::kBool &&
          entry["special"].AsBool()) {
        tokenizer.specialTokens_.push_back(token);
      }
    }
    std::sort(tokenizer.specialTokens_.begin(), tokenizer.specialTokens_.end(),
              [](const std::string &lhs, const std::string &rhs) {
                if (lhs.size() != rhs.size()) {
                  return lhs.size() > rhs.size();
                }
                return lhs < rhs;
              });
    tokenizer.specialTokens_.erase(
        std::unique(tokenizer.specialTokens_.begin(),
                    tokenizer.specialTokens_.end()),
        tokenizer.specialTokens_.end());
  }

  if (tokenizer.vocab_.empty() || tokenizer.mergeRank_.empty()) {
    if (error != nullptr) {
      *error = "tokenizer.json BPE model has empty vocab or merges";
    }
    return std::nullopt;
  }

  return tokenizer;
}

std::vector<std::string>
BpeTokenizer::BpeMergeWord(const std::string &word) const {
  std::vector<std::string> symbols;
  symbols.reserve(word.size());
  for (const char c : word) {
    symbols.emplace_back(1, c);
  }

  if (symbols.size() < 2) {
    return symbols;
  }

  while (symbols.size() > 1) {
    int bestRank = std::numeric_limits<int>::max();
    std::size_t bestIndex = symbols.size();
    for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto it = mergeRank_.find(PairKey(symbols[i], symbols[i + 1]));
      if (it != mergeRank_.end() && it->second < bestRank) {
        bestRank = it->second;
        bestIndex = i;
      }
    }
    if (bestIndex == symbols.size()) {
      break; // no mergeable pair left
    }
    symbols[bestIndex] = symbols[bestIndex] + symbols[bestIndex + 1];
    symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(bestIndex) + 1);
  }

  return symbols;
}

std::vector<std::string> BpeTokenizer::Encode(std::string_view text) const {
  std::vector<std::string> tokens;
  if (specialTokens_.empty()) {
    for (const std::string &word : SplitWhitespace(text)) {
      for (std::string &symbol : BpeMergeWord(word)) {
        tokens.push_back(std::move(symbol));
      }
    }
    return tokens;
  }

  std::string segment;
  const auto flushSegment = [&]() {
    for (const std::string &word : SplitWhitespace(segment)) {
      std::vector<std::string> merged = BpeMergeWord(word);
      tokens.insert(tokens.end(), std::make_move_iterator(merged.begin()),
                    std::make_move_iterator(merged.end()));
    }
  };
  std::size_t pos = 0;
  while (pos < text.size()) {
    bool matchedSpecialToken = false;
    for (const std::string &specialToken : specialTokens_) {
      if (text.substr(pos, specialToken.size()) == specialToken) {
        flushSegment();
        segment.clear();
        tokens.push_back(specialToken);
        pos += specialToken.size();
        matchedSpecialToken = true;
        break;
      }
    }
    if (!matchedSpecialToken) {
      segment.push_back(text[pos]);
      ++pos;
    }
  }
  flushSegment();
  return tokens;
}

std::vector<int> BpeTokenizer::EncodeIds(std::string_view text) const {
  std::vector<int> ids;
  for (const std::string &token : Encode(text)) {
    auto it = vocab_.find(token);
    if (it != vocab_.end()) {
      ids.push_back(it->second);
    } else if (unkId_.has_value()) {
      ids.push_back(*unkId_);
    }
  }
  return ids;
}

} // namespace us4
