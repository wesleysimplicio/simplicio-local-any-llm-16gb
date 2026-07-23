#pragma once

#include <string>

namespace us4 {

struct NativeServeOptions {
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string webRoot = "apps/web-chat/dist";
};

// Minimal single-threaded, blocking HTTP/1.1 server answering
// GET /v1/models and POST /v1/chat/completions directly from this
// runtime's own adapter registry and Generate() pipeline -- no external
// process, unlike the scripts/openai_serve.py proxy mode. Intentionally
// simple (one request at a time, no keep-alive, no TLS): this is the
// native-serving contract required by issue #81.10, not a production
// concurrent server. Blocks until the listening socket fails or the
// process is killed; returns a non-zero exit code only on a bind/listen
// failure.
int RunNativeHttpServer(const NativeServeOptions &options);

} // namespace us4
