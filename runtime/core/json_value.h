#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace us4 {

// Minimal recursive-descent JSON parser covering the subset needed to read
// HuggingFace-style tokenizer.json files (objects, arrays, strings, numbers,
// booleans, null). Not a general-purpose JSON library.
class JsonValue {
public:
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  JsonValue() : type_(Type::kNull) {}

  static JsonValue Parse(const std::string &text);

  Type type() const { return type_; }
  bool IsObject() const { return type_ == Type::kObject; }
  bool IsArray() const { return type_ == Type::kArray; }
  bool IsString() const { return type_ == Type::kString; }

  const std::string &AsString() const { return string_; }
  double AsNumber() const { return number_; }
  bool AsBool() const { return bool_; }

  const std::vector<JsonValue> &AsArray() const { return array_; }

  bool Has(const std::string &key) const {
    return type_ == Type::kObject && object_.count(key) > 0;
  }

  const JsonValue &operator[](const std::string &key) const {
    static const JsonValue kNullValue;
    auto it = object_.find(key);
    if (it == object_.end()) {
      return kNullValue;
    }
    return it->second;
  }

  const std::map<std::string, JsonValue> &Entries() const { return object_; }

private:
  Type type_ = Type::kNull;
  bool bool_ = false;
  double number_ = 0.0;
  std::string string_;
  std::vector<JsonValue> array_;
  std::map<std::string, JsonValue> object_;

  friend class JsonParser;
};

} // namespace us4
