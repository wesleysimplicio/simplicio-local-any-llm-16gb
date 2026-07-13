#include "net/openai_chat_handler.h"

#include "adapters/adapter_registry.h"
#include "core/hardware_probe.h"
#include "core/json_value.h"
#include "core/model_asset.h"
#include "core/runtime_context.h"

namespace us4 {

namespace {

std::string EscapeJsonString(const std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (const char ch : value) {
    switch (ch) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped.push_back(ch);
      break;
    }
  }
  return escaped;
}

} // namespace

std::optional<ChatCompletionRequest>
ParseChatCompletionRequestBody(const std::string &jsonBody,
                               std::string *error) {
  JsonValue root;
  try {
    root = JsonValue::Parse(jsonBody);
  } catch (const std::exception &ex) {
    if (error != nullptr) {
      *error = std::string("malformed JSON request body: ") + ex.what();
    }
    return std::nullopt;
  }

  if (!root.IsObject() || !root.Has("model") || !root["model"].IsString()) {
    if (error != nullptr) {
      *error = "request body must be a JSON object with a string \"model\"";
    }
    return std::nullopt;
  }

  ChatCompletionRequest request;
  request.model = root["model"].AsString();

  if (root.Has("max_tokens") &&
      root["max_tokens"].type() == JsonValue::Type::kNumber) {
    request.maxTokens = static_cast<std::size_t>(root["max_tokens"].AsNumber());
  }
  if (root.Has("max_completion_tokens") &&
      root["max_completion_tokens"].type() == JsonValue::Type::kNumber) {
    request.maxTokens =
        static_cast<std::size_t>(root["max_completion_tokens"].AsNumber());
  }
  if (root.Has("stream") && root["stream"].type() == JsonValue::Type::kBool) {
    request.stream = root["stream"].AsBool();
  }

  if (root.Has("model_path") && root["model_path"].IsString()) {
    request.modelPath = root["model_path"].AsString();
  }

  if (!root.Has("messages") || !root["messages"].IsArray()) {
    if (error != nullptr) {
      *error = "request body must have a \"messages\" array";
    }
    return std::nullopt;
  }

  bool foundUserMessage = false;
  for (const JsonValue &message : root["messages"].AsArray()) {
    if (!message.IsObject() || !message.Has("role") ||
        !message["role"].IsString() || message["role"].AsString() != "user" ||
        !message.Has("content") || !message["content"].IsString()) {
      continue;
    }
    request.prompt = message["content"].AsString();
    foundUserMessage = true;
  }

  if (!foundUserMessage) {
    if (error != nullptr) {
      *error = "request body has no \"user\" message to answer";
    }
    return std::nullopt;
  }

  return request;
}

ChatCompletionResponse
HandleChatCompletion(const ChatCompletionRequest &request) {
  ChatCompletionResponse response;
  response.modelName = request.model;

  const IUS4V6Adapter *adapter = FindAdapterByModel(request.model);
  if (adapter == nullptr) {
    response.errorMessage = "unknown model: " + request.model;
    return response;
  }

  RuntimeContext context(HardwareProbe::Detect());
  adapter->ConfigureRuntime(context);

  ModelAsset asset;
  const ModelAsset *assetPtr = nullptr;
  if (!request.modelPath.empty()) {
    std::string loadError;
    if (LoadModelAsset(request.modelPath, asset, &loadError)) {
      assetPtr = &asset;
    } else {
      response.errorMessage = "failed to load model_path: " + loadError;
      return response;
    }
  }

  const GenerationResult result =
      adapter->Generate({.prompt = request.prompt,
                         .maxTokens = request.maxTokens,
                         .asset = assetPtr},
                        context);

  response.ok = true;
  response.content = result.text;
  response.generatedTokens = result.generatedTokens;
  response.usedRealWeights = result.usedRealWeights;
  return response;
}

std::string
BuildChatCompletionResponseJson(const ChatCompletionResponse &response,
                                const std::string &requestId) {
  if (!response.ok) {
    return "{\"error\":{\"message\":\"" +
           EscapeJsonString(response.errorMessage) +
           "\",\"type\":\"invalid_request_error\"}}";
  }

  return "{"
         "\"id\":\"" +
         EscapeJsonString(requestId) +
         "\","
         "\"object\":\"chat.completion\","
         "\"model\":\"" +
         EscapeJsonString(response.modelName) +
         "\","
         "\"used_real_weights\":" +
         (response.usedRealWeights ? "true" : "false") +
         ","
         "\"choices\":[{"
         "\"index\":0,"
         "\"message\":{\"role\":\"assistant\",\"content\":\"" +
         EscapeJsonString(response.content) +
         "\"},"
         "\"finish_reason\":\"stop\""
         "}]"
         "}";
}

std::string BuildChatCompletionChunkJson(const std::string &requestId,
                                         const std::string &modelName,
                                         const std::string &delta,
                                         const bool finish) {
  return "{"
         "\"id\":\"" +
         EscapeJsonString(requestId) +
         "\","
         "\"object\":\"chat.completion.chunk\","
         "\"model\":\"" +
         EscapeJsonString(modelName) +
         "\","
         "\"choices\":[{"
         "\"index\":0,"
         "\"delta\":{\"content\":\"" +
         EscapeJsonString(delta) +
         "\"},"
         "\"finish_reason\":" +
         (finish ? "\"stop\"" : "null") +
         "}]"
         "}";
}

} // namespace us4
