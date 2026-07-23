add_library(us4_runtime_core
  ane/ane_backend.cpp
  ane/layer_offloader.cpp
  ane/mixed_dispatch.cpp
  core/backend_selector.cpp
  core/runtime_mode.cpp
  core/hardware_probe.cpp
  core/gqa_attention.cpp
  core/bpe_tokenizer.cpp
  core/json_value.cpp
  core/local_inference_policy.cpp
  core/model_asset.cpp
  core/gguf_reader.cpp
  core/safetensors_reader.cpp
  core/rope.cpp
  core/runtime_context.cpp
  core/tensor.cpp
  cpu/int8_matmul.cpp
  cpu/scalar_matmul.cpp
  cpu/scalar_attention.cpp
  cpu/ternary_lut.cpp
  cpu/quantize_projection.cpp
  adapters/dense_adapter_base.cpp
  adapters/adapter_registry.cpp
  adapters/bitnet/bitnet_adapter.cpp
  adapters/deepseek/deepseek_moe_adapter.cpp
  adapters/qwen/qwen_adapter.cpp
  adapters/gemma/gemma_adapter.cpp
  adapters/glm/glm_moe_adapter.cpp
  adapters/kimi/kimi_moe_adapter.cpp
  adapters/llama/llama_config.cpp
  adapters/llama/llama_adapter.cpp
  adapters/minimax/minimax_moe_adapter.cpp
  adapters/ternary/ternary_adapter.cpp
  cache/multimodal_cache.cpp
  cache/sparsity_aware_cache.cpp
  kv/eviction_policy.cpp
  kv/kv_pager.cpp
  kv/prefix_cache.cpp
  kv/ssd_cold_store.cpp
  kv/summarizer.cpp
  memory/unified_allocator.cpp
  metal/command_queue.cpp
  metal/device_info.cpp
  metal/dense_dispatch.cpp
  metal/kernel_library.cpp
  mlx/dense_plan.cpp
  mlx/mlx_bridge.cpp
  net/native_http_server.cpp
  net/openai_chat_handler.cpp
  net/static_asset.cpp
  moe/expert_ffn.cpp
  moe/expert_pager.cpp
  moe/moe_telemetry.cpp
  moe/router.cpp
  moe/router_metrics.cpp
  moe/shard_loader.cpp
  moe/speculative_prefetch.cpp
  neon/bitnet_matmul.cpp
  neon/block_gemm.cpp
  neon/dequant_int4.cpp
  neon/dequant_int8.cpp
  neon/neon_attention.cpp
  neon/kernel_profile.cpp
  neon/neon_matmul.cpp
  scheduler/continuous_batcher.cpp
  scheduler/session_pool.cpp
  speculative/draft_model_loader.cpp
  speculative/eagle3_decoder.cpp
  speculative/peagle_decoder.cpp
  speculative/speculative_telemetry.cpp
  telemetry/telemetry_types.cpp
  telemetry/telemetry_sink.cpp
  tuning/auto_tuner.cpp
  tuning/profile_cache.cpp
  tuning/thermal_monitor.cpp
)

if(APPLE)
  enable_language(OBJCXX)
  target_sources(us4_runtime_core PRIVATE
    metal/autorelease_scope.mm
    metal/native_metal_backend.mm
  )
  set_source_files_properties(
    metal/native_metal_backend.mm
    PROPERTIES COMPILE_OPTIONS "-fobjc-arc"
  )
  target_link_libraries(us4_runtime_core PRIVATE
    "-framework Foundation"
    "-framework Metal"
  )

  find_package(MLX CONFIG QUIET)
  if(TARGET mlx OR TARGET MLX::mlx)
    target_sources(us4_runtime_core PRIVATE mlx/native_mlx_backend.cpp)
    if(TARGET MLX::mlx)
      target_link_libraries(us4_runtime_core PRIVATE MLX::mlx)
    else()
      target_link_libraries(us4_runtime_core PRIVATE mlx)
    endif()
    target_compile_definitions(us4_runtime_core PUBLIC US4_HAS_NATIVE_MLX=1)
    message(STATUS "US4 native MLX backend enabled")
  else()
    find_path(US4_MLX_INCLUDE_DIR mlx/mlx.h)
    find_library(US4_MLX_LIBRARY NAMES mlx)
    if(US4_MLX_INCLUDE_DIR AND US4_MLX_LIBRARY)
      target_sources(us4_runtime_core PRIVATE mlx/native_mlx_backend.cpp)
      target_include_directories(us4_runtime_core PRIVATE ${US4_MLX_INCLUDE_DIR})
      target_link_libraries(us4_runtime_core PRIVATE ${US4_MLX_LIBRARY})
      target_compile_definitions(us4_runtime_core PUBLIC US4_HAS_NATIVE_MLX=1)
      message(STATUS "US4 native MLX backend enabled")
    else()
      target_sources(us4_runtime_core PRIVATE mlx/native_mlx_backend_stub.cpp)
      message(STATUS "US4 native MLX backend disabled: MLX package not found")
    endif()
  endif()
else()
  target_sources(us4_runtime_core PRIVATE
    metal/autorelease_scope.cpp
    metal/native_metal_backend_stub.cpp
    mlx/native_mlx_backend_stub.cpp
  )
endif()

target_include_directories(us4_runtime_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/../generated
)

target_compile_features(us4_runtime_core PUBLIC cxx_std_20)

if(WIN32)
  target_link_libraries(us4_runtime_core PRIVATE ws2_32)
endif()

if(US4_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(US4_BUILD_FUZZERS)
  add_subdirectory(fuzz)
endif()
