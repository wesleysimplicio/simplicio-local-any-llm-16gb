#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace us4 {

// Native (in-process, no external server) handling of an OpenAI-compatible
// /v1/chat/completions request, so `us4-cli serve --native` can answer
// directly from this runtime's own adapters/Generate() pipeline instead of
// proxying to mlx_lm.server/Ollama (see scripts/openai_serve.py, which
// remains the default proxy mode).
//
// Split from the socket I/O (native_http_server.h) so the request/response
// logic is unit-testable without binding a real port.

struct ChatCompletionRequest {
  std::string model;
  std::string prompt; // content of the last "user" message
  std::size_t maxTokens = 64;
  bool stream = false;
  // us4-cli extension (not part of the OpenAI schema): optional path to a
  // .safetensors/.gguf/.us4manifest asset to load real weights from, the
  // same way `us4-cli run --model-path` does.
  std::string modelPath;
};

struct ChatCompletionResponse {
  bool ok = false;
  std::string errorMessage;
  std::string modelName;
  std::string content;
  std::vector<std::string> generatedTokens;
  bool usedRealWeights = false;
};

// Parses an OpenAI-shaped request body:
//   {"model": "...", "messages": [{"role": "user", "content": "..."}],
//    "max_tokens": N}
// Returns std::nullopt (with `error` set) for malformed JSON, a missing
// "model", or no "user" message to answer.
std::optional<ChatCompletionRequest>
ParseChatCompletionRequestBody(const std::string &jsonBody, std::string *error);

// Runs the request through the adapter registry's real Generate()
// pipeline (real weights when the resolved model asset has them, per
// #81.4/#81.5; the runtime's own explicit-fallback rules apply otherwise --
// this handler never fabricates a response when the model is unknown).
ChatCompletionResponse
HandleChatCompletion(const ChatCompletionRequest &request);

// Serializes a response into the OpenAI chat-completions JSON shape.
std::string
BuildChatCompletionResponseJson(const ChatCompletionResponse &response,
                                const std::string &requestId);

std::string BuildChatCompletionChunkJson(const std::string &requestId,
                                         const std::string &modelName,
                                         const std::string &delta,
                                         bool finish);

} // namespace us4
