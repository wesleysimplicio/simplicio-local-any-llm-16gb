#include "net/native_http_server.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "adapters/adapter_registry.h"
#include "net/openai_chat_handler.h"
#include "net/static_asset.h"

namespace us4 {

namespace {

constexpr std::size_t kMaxRequestBytes = 1U << 20; // 1 MiB, plenty for chat.

#ifdef _WIN32
using SocketHandle = SOCKET;
using SocketByteCount = int;
constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;

struct SocketRuntime {
  WSADATA data{};
  bool ready = false;

  SocketRuntime() : ready(WSAStartup(MAKEWORD(2, 2), &data) == 0) {}
  ~SocketRuntime() {
    if (ready) {
      WSACleanup();
    }
  }
  SocketRuntime(const SocketRuntime &) = delete;
  SocketRuntime &operator=(const SocketRuntime &) = delete;
};

int CloseSocket(const SocketHandle socket) { return closesocket(socket); }
SocketByteCount ReadSocket(const SocketHandle socket, char *buffer,
                           const std::size_t size) {
  return recv(socket, buffer, static_cast<int>(size), 0);
}
SocketByteCount WriteSocket(const SocketHandle socket, const char *buffer,
                            const std::size_t size) {
  return send(socket, buffer, static_cast<int>(size), 0);
}
#else
using SocketHandle = int;
using SocketByteCount = ssize_t;
constexpr SocketHandle kInvalidSocket = -1;

struct SocketRuntime {
  bool ready = true;
  SocketRuntime() = default;
};

int CloseSocket(const SocketHandle socket) { return close(socket); }
SocketByteCount ReadSocket(const SocketHandle socket, char *buffer,
                           const std::size_t size) {
  return read(socket, buffer, size);
}
SocketByteCount WriteSocket(const SocketHandle socket, const char *buffer,
                            const std::size_t size) {
  return write(socket, buffer, size);
}
#endif

bool ReadLine(const SocketHandle clientFd, std::string *line) {
  line->clear();
  char ch = '\0';
  while (true) {
    const SocketByteCount bytesRead = ReadSocket(clientFd, &ch, 1);
    if (bytesRead <= 0) {
      return false;
    }
    if (ch == '\n') {
      if (!line->empty() && line->back() == '\r') {
        line->pop_back();
      }
      return true;
    }
    line->push_back(ch);
    if (line->size() > kMaxRequestBytes) {
      return false;
    }
  }
}

struct ParsedRequest {
  std::string method;
  std::string path;
  std::size_t contentLength = 0;
  std::string body;
};

bool ReadHttpRequest(const SocketHandle clientFd, ParsedRequest *request) {
  std::string requestLine;
  if (!ReadLine(clientFd, &requestLine) || requestLine.empty()) {
    return false;
  }
  std::istringstream requestLineStream(requestLine);
  requestLineStream >> request->method >> request->path;

  std::string headerLine;
  while (ReadLine(clientFd, &headerLine) && !headerLine.empty()) {
    const std::string lowerPrefix = "content-length:";
    std::string lowered = headerLine;
    for (char &ch : lowered) {
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    if (lowered.rfind(lowerPrefix, 0) == 0) {
      const std::string valueText = headerLine.substr(lowerPrefix.size());
      request->contentLength = static_cast<std::size_t>(
          std::strtoul(valueText.c_str(), nullptr, 10));
    }
  }

  if (request->contentLength > kMaxRequestBytes) {
    return false;
  }
  request->body.resize(request->contentLength);
  std::size_t totalRead = 0;
  while (totalRead < request->contentLength) {
    const SocketByteCount bytesRead =
        ReadSocket(clientFd, request->body.data() + totalRead,
                   request->contentLength - totalRead);
    if (bytesRead <= 0) {
      return false;
    }
    totalRead += static_cast<std::size_t>(bytesRead);
  }
  return true;
}

void WriteHttpResponse(const SocketHandle clientFd, const int statusCode,
                       const std::string &statusText, const std::string &body,
                       const std::string &contentType = "application/json") {
  std::ostringstream response;
  response << "HTTP/1.1 " << statusCode << " " << statusText << "\r\n"
           << "Content-Type: " << contentType << "\r\n"
           << "Access-Control-Allow-Origin: *\r\n"
           << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
           << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
           << "Content-Length: " << body.size() << "\r\n"
           << "Connection: close\r\n"
           << "\r\n"
           << body;
  const std::string responseText = response.str();
  std::size_t totalWritten = 0;
  while (totalWritten < responseText.size()) {
    const SocketByteCount bytesWritten =
        WriteSocket(clientFd, responseText.data() + totalWritten,
                    responseText.size() - totalWritten);
    if (bytesWritten <= 0) {
      break; // best-effort: client disconnected mid-response.
    }
    totalWritten += static_cast<std::size_t>(bytesWritten);
  }
}

bool WriteAll(const SocketHandle clientFd, const std::string &payload) {
  std::size_t totalWritten = 0;
  while (totalWritten < payload.size()) {
    const SocketByteCount bytesWritten =
        WriteSocket(clientFd, payload.data() + totalWritten,
                    payload.size() - totalWritten);
    if (bytesWritten <= 0) {
      return false;
    }
    totalWritten += static_cast<std::size_t>(bytesWritten);
  }
  return true;
}

std::vector<std::string> ChunkContent(const ChatCompletionResponse &response) {
  constexpr std::size_t kChunkBytes = 24U;

  std::vector<std::string> chunks;
  if (!response.content.empty()) {
    for (std::size_t offset = 0; offset < response.content.size();
         offset += kChunkBytes) {
      chunks.push_back(response.content.substr(offset, kChunkBytes));
    }
  }
  if (chunks.empty()) {
    chunks.push_back("");
  }
  return chunks;
}

void WriteSseChatResponse(const SocketHandle clientFd,
                          const ChatCompletionResponse &response,
                          const std::string &requestId) {
  std::ostringstream headers;
  headers << "HTTP/1.1 200 OK\r\n"
          << "Content-Type: text/event-stream\r\n"
          << "Access-Control-Allow-Origin: *\r\n"
          << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
          << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
          << "Cache-Control: no-cache\r\n"
          << "Connection: close\r\n"
          << "X-Accel-Buffering: no\r\n"
          << "\r\n";
  if (!WriteAll(clientFd, headers.str())) {
    return;
  }

  const std::vector<std::string> chunks = ChunkContent(response);
  for (std::size_t index = 0; index < chunks.size(); ++index) {
    const bool finish = index + 1U == chunks.size();
    const std::string payload = "data: " +
                                BuildChatCompletionChunkJson(
                                    requestId, response.modelName, chunks[index],
                                    finish) +
                                "\n\n";
    if (!WriteAll(clientFd, payload)) {
      return;
    }
  }

  static_cast<void>(WriteAll(clientFd, "data: [DONE]\n\n"));
}

std::string BuildModelsListJson() {
  std::ostringstream json;
  json << "{\"object\":\"list\",\"data\":[";
  bool first = true;
  for (const IUS4V6Adapter *adapter : ListAdapters()) {
    if (!first) {
      json << ",";
    }
    first = false;
    json << "{\"id\":\"" << adapter->ModelName() << "\",\"object\":\"model\"}";
  }
  json << "]}";
  return json.str();
}

void HandleClient(const SocketHandle clientFd, std::size_t requestCounter,
                  const NativeServeOptions &options) {
  ParsedRequest request;
  if (!ReadHttpRequest(clientFd, &request)) {
    WriteHttpResponse(clientFd, 400, "Bad Request",
                      "{\"error\":{\"message\":\"malformed HTTP request\"}}");
    return;
  }

  if (request.method == "OPTIONS") {
    WriteHttpResponse(clientFd, 204, "No Content", "");
    return;
  }

  if (request.method == "GET" && request.path == "/v1/models") {
    WriteHttpResponse(clientFd, 200, "OK", BuildModelsListJson());
    return;
  }

  if (request.method == "GET" && request.path.rfind("/v1/", 0) != 0) {
    const auto asset = LoadStaticAsset(options.webRoot, request.path);
    if (asset.has_value()) {
      WriteHttpResponse(clientFd, 200, "OK", asset->body,
                        asset->contentType);
      return;
    }
  }

  if (request.method == "POST" && request.path == "/v1/chat/completions") {
    std::string parseError;
    const auto chatRequest =
        ParseChatCompletionRequestBody(request.body, &parseError);
    if (!chatRequest.has_value()) {
      WriteHttpResponse(clientFd, 400, "Bad Request",
                        "{\"error\":{\"message\":\"" + parseError + "\"}}");
      return;
    }
    const ChatCompletionResponse chatResponse =
        HandleChatCompletion(*chatRequest);
    const std::string requestId =
        "us4-native-" + std::to_string(requestCounter);
    if (chatRequest->stream && chatResponse.ok) {
      WriteSseChatResponse(clientFd, chatResponse, requestId);
      return;
    }
    const int statusCode = chatResponse.ok ? 200 : 404;
    const std::string statusText = chatResponse.ok ? "OK" : "Not Found";
    WriteHttpResponse(clientFd, statusCode, statusText,
                      BuildChatCompletionResponseJson(chatResponse, requestId));
    return;
  }

  WriteHttpResponse(clientFd, 404, "Not Found",
                    "{\"error\":{\"message\":\"unknown route\"}}");
}

} // namespace

int RunNativeHttpServer(const NativeServeOptions &options) {
  const SocketRuntime socketRuntime;
  if (!socketRuntime.ready) {
    std::cerr << "us4-cli serve --native: socket runtime initialization failed\n";
    return 1;
  }

  const SocketHandle listenFd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listenFd == kInvalidSocket) {
    std::cerr << "us4-cli serve --native: socket() failed\n";
    return 1;
  }

  const int reuseAddr = 1;
#ifdef _WIN32
  ::setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char *>(&reuseAddr),
               sizeof(reuseAddr));
#else
  ::setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &reuseAddr,
               sizeof(reuseAddr));
#endif

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(options.port));
  if (::inet_pton(AF_INET, options.host.c_str(), &address.sin_addr) != 1) {
    std::cerr << "us4-cli serve --native: invalid host '" << options.host
              << "'\n";
    CloseSocket(listenFd);
    return 1;
  }

  if (::bind(listenFd, reinterpret_cast<sockaddr *>(&address),
             sizeof(address)) != 0) {
    std::cerr << "us4-cli serve --native: bind() failed on " << options.host
              << ":" << options.port << "\n";
    CloseSocket(listenFd);
    return 1;
  }

  if (::listen(listenFd, /*backlog=*/16) != 0) {
    std::cerr << "us4-cli serve --native: listen() failed\n";
    CloseSocket(listenFd);
    return 1;
  }

  std::cerr << "us4-cli serve --native: listening on " << options.host << ":"
            << options.port << " (native runtime, no external process)\n";

  std::size_t requestCounter = 0;
  while (true) {
    const SocketHandle clientFd = ::accept(listenFd, nullptr, nullptr);
    if (clientFd == kInvalidSocket) {
      continue;
    }
    ++requestCounter;
    HandleClient(clientFd, requestCounter, options);
    CloseSocket(clientFd);
  }
}

} // namespace us4
