#include "core/json_value.h"

#include <cctype>
#include <cstdlib>

namespace us4 {

class JsonParser {
public:
  explicit JsonParser(const std::string &text) : text_(text) {}

  JsonValue ParseValue() {
    SkipWhitespace();
    if (pos_ >= text_.size()) {
      throw std::runtime_error("unexpected end of JSON input");
    }
    const char c = text_[pos_];
    if (c == '{') {
      return ParseObject();
    }
    if (c == '[') {
      return ParseArray();
    }
    if (c == '"') {
      JsonValue value;
      value.type_ = JsonValue::Type::kString;
      value.string_ = ParseString();
      return value;
    }
    if (c == 't' || c == 'f') {
      return ParseBool();
    }
    if (c == 'n') {
      return ParseNull();
    }
    return ParseNumber();
  }

private:
  const std::string &text_;
  std::size_t pos_ = 0;

  void SkipWhitespace() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  char Peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }

  void Expect(char expected) {
    if (pos_ >= text_.size() || text_[pos_] != expected) {
      throw std::runtime_error("malformed JSON: expected character");
    }
    ++pos_;
  }

  std::string ParseString() {
    Expect('"');
    std::string result;
    while (pos_ < text_.size() && text_[pos_] != '"') {
      char c = text_[pos_++];
      if (c == '\\' && pos_ < text_.size()) {
        char escaped = text_[pos_++];
        switch (escaped) {
        case 'n':
          result.push_back('\n');
          break;
        case 't':
          result.push_back('\t');
          break;
        case 'r':
          result.push_back('\r');
          break;
        case '"':
          result.push_back('"');
          break;
        case '\\':
          result.push_back('\\');
          break;
        case '/':
          result.push_back('/');
          break;
        default:
          result.push_back(escaped);
          break;
        }
      } else {
        result.push_back(c);
      }
    }
    Expect('"');
    return result;
  }

  JsonValue ParseNumber() {
    const std::size_t start = pos_;
    if (Peek() == '-') {
      ++pos_;
    }
    while (pos_ < text_.size() &&
           (std::isdigit(static_cast<unsigned char>(text_[pos_])) ||
            text_[pos_] == '.' || text_[pos_] == 'e' || text_[pos_] == 'E' ||
            text_[pos_] == '+' || text_[pos_] == '-')) {
      ++pos_;
    }
    const std::string token = text_.substr(start, pos_ - start);
    if (token.empty()) {
      throw std::runtime_error("malformed JSON: expected number");
    }
    JsonValue value;
    value.type_ = JsonValue::Type::kNumber;
    value.number_ = std::strtod(token.c_str(), nullptr);
    return value;
  }

  JsonValue ParseBool() {
    JsonValue value;
    value.type_ = JsonValue::Type::kBool;
    if (text_.compare(pos_, 4, "true") == 0) {
      value.bool_ = true;
      pos_ += 4;
    } else if (text_.compare(pos_, 5, "false") == 0) {
      value.bool_ = false;
      pos_ += 5;
    } else {
      throw std::runtime_error("malformed JSON: expected boolean");
    }
    return value;
  }

  JsonValue ParseNull() {
    if (text_.compare(pos_, 4, "null") != 0) {
      throw std::runtime_error("malformed JSON: expected null");
    }
    pos_ += 4;
    return JsonValue();
  }

  JsonValue ParseArray() {
    Expect('[');
    JsonValue value;
    value.type_ = JsonValue::Type::kArray;
    SkipWhitespace();
    if (Peek() == ']') {
      ++pos_;
      return value;
    }
    while (true) {
      value.array_.push_back(ParseValue());
      SkipWhitespace();
      if (Peek() == ',') {
        ++pos_;
        continue;
      }
      break;
    }
    SkipWhitespace();
    Expect(']');
    return value;
  }

  JsonValue ParseObject() {
    Expect('{');
    JsonValue value;
    value.type_ = JsonValue::Type::kObject;
    SkipWhitespace();
    if (Peek() == '}') {
      ++pos_;
      return value;
    }
    while (true) {
      SkipWhitespace();
      std::string key = ParseString();
      SkipWhitespace();
      Expect(':');
      value.object_[key] = ParseValue();
      SkipWhitespace();
      if (Peek() == ',') {
        ++pos_;
        continue;
      }
      break;
    }
    SkipWhitespace();
    Expect('}');
    return value;
  }
};

JsonValue JsonValue::Parse(const std::string &text) {
  JsonParser parser(text);
  return parser.ParseValue();
}

} // namespace us4
