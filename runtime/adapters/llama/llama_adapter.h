#pragma once

#include "adapters/dense_adapter_base.h"
#include "adapters/llama/llama_config.h"

namespace us4 {

class LlamaAdapter final : public DenseAdapterBase {
public:
  LlamaAdapter();

  bool SupportsMlxBackend() const override;
  bool SupportsMetalBackend() const override;
  bool SupportsAneBackend() const override;
  GenerationResult Generate(const GenerationRequest &request,
                            const RuntimeContext &context) const override;

protected:
  std::uint32_t Seed() const override;
  std::vector<std::string> Vocabulary() const override;
  std::string DefaultPromptToken() const override;

private:
  std::vector<float> BuildQueryRow(std::size_t tokenId, std::uint32_t seed,
                                   std::size_t position,
                                   const LlamaConfig &config) const;
  std::vector<float> BuildKeyRow(std::size_t tokenId, std::uint32_t seed,
                                 std::size_t position,
                                 const LlamaConfig &config) const;
  std::vector<float> BuildValueRow(std::size_t tokenId, std::uint32_t seed,
                                   std::size_t position,
                                   const LlamaConfig &config) const;
};

} // namespace us4
