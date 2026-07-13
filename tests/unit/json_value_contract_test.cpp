#include <string>

#include <gtest/gtest.h>

#include "core/json_value.h"

namespace us4 {
namespace {

TEST(JsonValueContractTest, RejectsMalformedNumericTokens) {
  EXPECT_THROW((void)JsonValue::Parse("{\"n\":1-2}"), std::runtime_error);
  EXPECT_THROW((void)JsonValue::Parse("[1+2]"), std::runtime_error);
  EXPECT_THROW((void)JsonValue::Parse("{\"n\":--1}"), std::runtime_error);
}

TEST(JsonValueContractTest, RejectsExcessiveNestingDepth) {
  std::string nested(205, '[');
  nested.push_back('0');
  nested.append(205, ']');
  EXPECT_THROW((void)JsonValue::Parse(nested), std::runtime_error);
}

} // namespace
} // namespace us4
