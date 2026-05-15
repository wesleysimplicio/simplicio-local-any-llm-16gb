#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/ius4v6_adapter.h"

namespace us4 {

class DenseAdapterBase : public IUS4V6Adapter {
 public:
  DenseAdapterBase(std::string family, std::string modelName);

  std::string_view Family() const override;
  std::string_view ModelName() const override;
  ArchitectureType Architecture() const override;

  bool SupportsMoe() const override;
  bool SupportsMlxBackend() const override;
  bool SupportsSpeculativeDecoding() const override;
  bool SupportsPromptRun() const override;

  RuntimeMode MinimumMode() const override;
  RuntimeMode RecommendedMode(const HardwareProbeResult& hardware) const override;
  void ConfigureRuntime(RuntimeContext& context) const override;

  std::vector<std::string> Tokenize(std::string_view text) const override;
  GenerationResult Generate(const GenerationRequest& request, const RuntimeContext& context) const override;

 protected:
  virtual std::uint32_t Seed() const = 0;
  virtual std::vector<std::string> Vocabulary() const = 0;
  virtual std::string DefaultPromptToken() const = 0;

 private:
  std::size_t TokenIdFor(std::string_view token, const std::vector<std::string>& vocabulary) const;
  std::vector<float> BuildTokenEmbedding(std::size_t tokenId, std::size_t hiddenSize, std::uint32_t seed) const;
  std::vector<float> BuildOutputProjection(const std::vector<std::string>& vocabulary,
                                           std::size_t hiddenSize,
                                           std::uint32_t seed) const;
  std::string JoinTokens(const std::vector<std::string>& tokens) const;

  std::string family_;
  std::string model_name_;
};

}  // namespace us4
