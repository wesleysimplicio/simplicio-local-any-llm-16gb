#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "net/openai_chat_handler.h"

namespace us4 {
namespace {

std::filesystem::path RepoRoot() {
#ifdef US4_SOURCE_DIR
  return std::filesystem::path(US4_SOURCE_DIR);
#else
  return std::filesystem::path(__FILE__)
      .parent_path()
      .parent_path()
      .parent_path();
#endif
}

TEST(OpenAiChatHandlerContractTest, RejectsMalformedJsonBody) {
  std::string error;
  const auto request = ParseChatCompletionRequestBody("not json", &error);
  EXPECT_FALSE(request.has_value());
  EXPECT_FALSE(error.empty());
}

TEST(OpenAiChatHandlerContractTest, RejectsBodyWithoutUserMessage) {
  std::string error;
  const auto request = ParseChatCompletionRequestBody(
      R"({"model":"qwen-0.5b","messages":[]})", &error);
  EXPECT_FALSE(request.has_value());
  EXPECT_FALSE(error.empty());
}

TEST(OpenAiChatHandlerContractTest, ParsesModelPromptAndMaxTokens) {
  std::string error;
  const auto request = ParseChatCompletionRequestBody(
      R"({"model":"qwen-0.5b","max_tokens":5,)"
      R"("messages":[{"role":"system","content":"ignored"},)"
      R"({"role":"user","content":"hello there"}]})",
      &error);
  ASSERT_TRUE(request.has_value()) << error;
  EXPECT_EQ(request->model, "qwen-0.5b");
  EXPECT_EQ(request->prompt, "hello there");
  EXPECT_EQ(request->maxTokens, 5U);
  EXPECT_FALSE(request->stream);
}

TEST(OpenAiChatHandlerContractTest,
     ParsesStreamFlagAndMaxCompletionTokensAlias) {
  std::string error;
  const auto request = ParseChatCompletionRequestBody(
      R"({"model":"qwen-0.5b","max_completion_tokens":7,"stream":true,)"
      R"("messages":[{"role":"user","content":"stream this"}]})",
      &error);
  ASSERT_TRUE(request.has_value()) << error;
  EXPECT_EQ(request->maxTokens, 7U);
  EXPECT_TRUE(request->stream);
}

TEST(OpenAiChatHandlerContractTest, UnknownModelReturnsExplicitError) {
  ChatCompletionRequest request;
  request.model = "does-not-exist";
  request.prompt = "hi";
  const ChatCompletionResponse response = HandleChatCompletion(request);
  EXPECT_FALSE(response.ok);
  EXPECT_FALSE(response.errorMessage.empty());
}

// Issue #81.10: the native handler must answer from this runtime's own
// Generate() pipeline with real weights when a real model_path is given --
// not a canned/mocked response.
TEST(OpenAiChatHandlerContractTest,
     RealModelPathDrivesRealWeightsThroughNativeHandler) {
  ChatCompletionRequest request;
  request.model = "qwen-0.5b";
  request.prompt = "alpha";
  request.maxTokens = 1;
  request.modelPath = (RepoRoot() / "tests" / "fixtures" / "models" /
                       "toy-dense-real" / "toy-dense-real.safetensors")
                          .string();

  const ChatCompletionResponse response = HandleChatCompletion(request);
  ASSERT_TRUE(response.ok) << response.errorMessage;
  EXPECT_TRUE(response.usedRealWeights);
  // Same external-oracle prediction as the #85 CLI/Playwright evidence:
  // embedding("alpha") one-hot over these real weights argmaxes to "delta".
  EXPECT_EQ(response.content, "delta");

  const std::string responseJson =
      BuildChatCompletionResponseJson(response, "req-1");
  EXPECT_NE(responseJson.find("\"used_real_weights\":true"), std::string::npos);
  EXPECT_NE(responseJson.find("\"content\":\"delta\""), std::string::npos);

  const std::string chunkJson =
      BuildChatCompletionChunkJson("req-1", response.modelName, "delta", false);
  EXPECT_NE(chunkJson.find("\"object\":\"chat.completion.chunk\""),
            std::string::npos);
  EXPECT_NE(chunkJson.find("\"content\":\"delta\""), std::string::npos);
  EXPECT_NE(chunkJson.find("\"finish_reason\":null"), std::string::npos);
}

} // namespace
} // namespace us4
