UNAME_S := $(shell uname -s)
MINGW    := $(findstring MINGW,$(UNAME_S))
MSYS     := $(findstring MSYS,$(UNAME_S))
IS_WIN   := $(MINGW)$(MSYS)

ifeq ($(UNAME_S),Darwin)
# --- macOS / Apple Silicon ---
# Apple clang non include il runtime OpenMP: se c'e' libomp di Homebrew lo usa
# (brew install libomp), altrimenti compila single-thread (i pragma omp sono ignorati).
# Niente -march: su arm64 NEON e' baseline (i kernel __ARM_NEON si attivano da soli).
CC      = clang
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(OMPDIR),)
OMPC    = -Xclang -fopenmp -I$(OMPDIR)/include
OMPL    = -L$(OMPDIR)/lib -lomp
else
$(warning libomp non trovato: build SINGLE-THREAD. Per il multithread: brew install libomp)
OMPC    =
OMPL    =
endif
CFLAGS  = -O3 $(OMPC) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm $(OMPL)
EXE     =
else ifneq ($(IS_WIN),)
# --- Windows 11 x86-64 (MinGW-w64 / MSYS2) ---
# GCC + libgomp + winpthreads: pthread, OpenMP, clock_gettime, opendir/readdir,
# AVX2 intrinsics — tutto gratis, nessun porting.
# ARCH default = x86-64-v3 (binario portabile con AVX2). Per max velocita'
# su QUESTA macchina: make ARCH=native
CC      = gcc
ARCH   ?= x86-64-v3
CFLAGS  = -D_FILE_OFFSET_BITS=64 -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp -static
EXE     = .exe
else
# --- Linux x86-64 (percorso originale, invariato) ---
CC      = gcc
# ARCH=native -> ottimizzato per QUESTA macchina (default, piu' veloce).
# ARCH=x86-64-v3 -> binario PORTABILE su qualsiasi x86-64 moderno con AVX2 (per distribuire).
# ARCH=x86-64 -> massima compatibilita' (niente AVX2: usa il path scalare di fallback).
ARCH   ?= native
CFLAGS  = -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp
EXE     =
endif

# CUDA=1 adds an opt-in backend for resident tensors. The default build remains
# pure C and keeps the original zero-dependency runtime.
CUDA      ?= 0
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler=-Wall,-Wextra
PYTHON    ?= python3
CUDA_OBJ  =
TEST_BINS = tests/test_json$(EXE) tests/test_st$(EXE) tests/test_tier$(EXE) tests/test_quant$(EXE) tests/test_serve_protocol$(EXE)
TOKENIZER_TEST_BIN = tests/test_tok$(EXE)
CHAT_TEMPLATE_TEST_BIN = tests/test_chat_template$(EXE)
FUZZ_NAMES = json safetensors tokenizer serve_protocol
FUZZ_REPLAY_BINS = $(addprefix fuzz/bin/fuzz_,$(addsuffix _replay,$(FUZZ_NAMES)))
FUZZ_SANFLAGS = -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all
FUZZ_CLANG ?= clang
ifeq ($(CUDA),1)
ifeq ($(UNAME_S),Darwin)
$(error CUDA=1 is supported only on Linux)
endif
ifneq ($(IS_WIN),)
# GPU: stub only in Phase 1 (G0). G1 builds coli_cuda.dll with MSVC+nvcc.
$(error CUDA=1 on Windows requires G1: build coli_cuda.dll with MSVC+nvcc (see PORT_WINDOWS_PLAN.md §8))
endif
CFLAGS   += -DCOLI_CUDA
LDFLAGS  += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -lstdc++
CUDA_OBJ  = backend_cuda.o
endif

ifeq ($(EXE),)
# On POSIX, glm is the real binary target.  Do not add a glm -> glm alias:
# GNU make treats that as a circular dependency and silently drops it.
all: glm
else
# On Windows, keep the friendly `make glm` alias for the .exe target.
all: glm$(EXE)
.PHONY: glm
glm: glm$(EXE)
endif

glm$(EXE): glm.c moe_route.h st.h json.h tok.h tok_unicode.h chat_template.h chat_templates.json serve_protocol.h compat.h $(CUDA_OBJ)
	$(CC) $(CFLAGS) glm.c $(CUDA_OBJ) -o glm$(EXE) $(LDFLAGS)

backend_cuda.o: backend_cuda.cu backend_cuda.h
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) -c backend_cuda.cu -o $@

cuda-test: backend_cuda.cu backend_cuda.h tests/test_backend_cuda.cu
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) backend_cuda.cu tests/test_backend_cuda.cu -o backend_cuda_test$(EXE)
	./backend_cuda_test$(EXE)

olmoe$(EXE): olmoe.c st.h json.h compat.h
	$(CC) $(CFLAGS) olmoe.c -o olmoe$(EXE) $(LDFLAGS)

# binario portabile da distribuire su altre macchine x86-64
portable:
	$(MAKE) glm$(EXE) ARCH=x86-64-v3

iobench$(EXE): iobench.c compat.h
	$(CC) $(CFLAGS) iobench.c -o iobench$(EXE) $(LDFLAGS)

tests/test_json$(EXE): tests/test_json.c json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st$(EXE): tests/test_st.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_tier$(EXE): tests/test_tier.c tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_quant$(EXE): tests/test_quant.c glm.c moe_route.h st.h json.h tok.h tok_unicode.h serve_protocol.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_serve_protocol$(EXE): tests/test_serve_protocol.c serve_protocol.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(TOKENIZER_TEST_BIN): tests/test_tok.c tok.h tok_unicode.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(CHAT_TEMPLATE_TEST_BIN): tests/test_chat_template.c chat_template.h chat_templates.json json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-c: $(TEST_BINS)
	@for test in $(TEST_BINS); do ./$$test || exit 1; done

test-python:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

test-tokenizer: $(TOKENIZER_TEST_BIN)
	printf 'hi\t2\nhi!\t2,3\n<|special|>\t4\nhi<|special|>!\t2,4,3\n' \
	  | ./$(TOKENIZER_TEST_BIN) ../../tests/fixtures/engine/tokenizer/minimal_tokenizer.json
	@for family in glm deepseek kimi; do \
	  ./$(TOKENIZER_TEST_BIN) tests/fixtures/tokenizer/contract_tokenizer_$$family.json \
	    tests/fixtures/tokenizer/tok_vectors_$$family.json || exit 1; \
	done

test-chat-template: $(CHAT_TEMPLATE_TEST_BIN)
	./$(CHAT_TEMPLATE_TEST_BIN) chat_templates.json tests/fixtures/chat_vectors.json

test: test-c test-tokenizer test-chat-template test-python

# Local validation: one portable CPU build and dependency-free tests.
check:
	$(MAKE) clean
	$(MAKE) portable
	$(MAKE) test

# US4_ENABLE_ASAN/UBSAN (issue #122): build glm + the C test binaries with
# AddressSanitizer and UndefinedBehaviorSanitizer, single-threaded (sanitizer
# runtimes are not fork/OpenMP-safe by default here), then run the C test
# suite plus the GLM and DeepSeek teacher-forcing oracles under it when the
# optional generated fixtures are available. Set SANITIZE_REQUIRE_ORACLES=1
# to make their absence a hard failure (the default is an explicit, honest
# skip so a dependency-light checkout can still run the sanitizer suite).
# detect_leaks=0: json.h's parsed jval tree is process-lifetime by design
# across this whole codebase (glm.c's own load_cfg/cfg_root only ever frees
# the raw-text arena, never the node tree - see json_parse call sites) - a
# one-shot config/tokenizer parse in a short-lived CLI/server process is not
# a safety bug, and leaving leak detection on would flag this known,
# intentional pattern instead of the overflow/UB corruption this target
# actually exists to catch.
SANFLAGS = -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all
SANITIZE_REQUIRE_ORACLES ?= 0
sanitize: fuzz-smoke
	rm -f glm_asan tests/test_json_asan tests/test_st_asan tests/test_tier_asan tests/test_quant_asan tests/test_serve_protocol_asan tests/test_tok_asan
	$(CC) $(SANFLAGS) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function glm.c -o glm_asan -lm
	$(CC) $(SANFLAGS) tests/test_json.c -o tests/test_json_asan -lm
	$(CC) $(SANFLAGS) tests/test_st.c -o tests/test_st_asan -lm
	$(CC) $(SANFLAGS) tests/test_tier.c -o tests/test_tier_asan -lm
	$(CC) $(SANFLAGS) tests/test_quant.c -o tests/test_quant_asan -lm
	$(CC) $(SANFLAGS) tests/test_serve_protocol.c -o tests/test_serve_protocol_asan -lm
	$(CC) $(SANFLAGS) tests/test_tok.c -o tests/test_tok_asan -lm
	ASAN_OPTIONS=detect_leaks=0 ./tests/test_json_asan
	ASAN_OPTIONS=detect_leaks=0 ./tests/test_st_asan
	ASAN_OPTIONS=detect_leaks=0 ./tests/test_tier_asan
	ASAN_OPTIONS=detect_leaks=0 ./tests/test_quant_asan
	ASAN_OPTIONS=detect_leaks=0 ./tests/test_serve_protocol_asan
	printf 'hi\t2\nhi!\t2,3\n<|special|>\t4\nhi<|special|>!\t2,4,3\n' \
	  | ASAN_OPTIONS=detect_leaks=0 ./tests/test_tok_asan ../../tests/fixtures/engine/tokenizer/minimal_tokenizer.json
	if test -f ./glm_tiny/config.json && test -f ./glm_tiny/model.safetensors && \
	   test -f ./deepseek_tiny/config.json && test -f ./deepseek_tiny/model.safetensors; then \
	  SNAP=./glm_tiny TF=1 REF=ref_glm.json ASAN_OPTIONS=detect_leaks=0 ./glm_asan 64 16 16; \
	  SNAP=./deepseek_tiny TF=1 REF=ref_deepseek.json ASAN_OPTIONS=detect_leaks=0 ./glm_asan 64 16 16; \
	else \
	  if test "$(SANITIZE_REQUIRE_ORACLES)" = "1"; then \
	    echo "sanitize: oracle fixtures missing; generate glm_tiny and deepseek_tiny first" >&2; exit 2; \
	  fi; \
	  echo "sanitize: C sanitizer suite passed; oracle fixtures unavailable (explicitly skipped)"; \
	fi
	SNAP=./fixtures/kimi_k2_tiny TF=1 REF=ref_kimi.json \
	  ASAN_OPTIONS=detect_leaks=0 ./glm_asan 64 16 16
	rm -f glm_asan tests/test_json_asan tests/test_st_asan tests/test_tier_asan tests/test_quant_asan tests/test_serve_protocol_asan tests/test_tok_asan

fuzz/bin:
	mkdir -p $@

fuzz/bin/fuzz_%_replay: fuzz/fuzz_%.c fuzz/fuzz_replay.c | fuzz/bin
	$(CC) $(FUZZ_SANFLAGS) -Wall -Wextra -Werror -Wno-unused-function $^ -o $@ -lm

# Dependency-light, deterministic sanitizer replay. This is a bounded smoke
# gate over the committed corpus, not a claim of a coverage-guided campaign.
fuzz-smoke: $(FUZZ_REPLAY_BINS)
	ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 sh ./fuzz/run_smoke.sh

# Optional coverage-guided targets. A real campaign controls duration and
# corpus directories externally; CI activation remains out of scope for #122.
fuzz-libfuzzer: | fuzz/bin
	@command -v $(FUZZ_CLANG) >/dev/null 2>&1 || { echo "clang not found: set FUZZ_CLANG" >&2; exit 1; }
	@for name in $(FUZZ_NAMES); do \
	  $(FUZZ_CLANG) $(FUZZ_SANFLAGS) -fsanitize=fuzzer -Wall -Wextra -Werror -Wno-unused-function \
	    fuzz/fuzz_$$name.c -o fuzz/bin/fuzz_$$name -lm || exit 1; \
	done

clean:
	rm -f olmoe$(EXE) glm$(EXE) iobench$(EXE) backend_cuda.o backend_cuda_test$(EXE) $(TEST_BINS)
	rm -f $(TOKENIZER_TEST_BIN)
	rm -f $(CHAT_TEMPLATE_TEST_BIN)
	rm -f glm_asan tests/test_json_asan tests/test_st_asan tests/test_tier_asan tests/test_quant_asan tests/test_serve_protocol_asan tests/test_tok_asan
	rm -f $(FUZZ_REPLAY_BINS) $(addprefix fuzz/bin/fuzz_,$(FUZZ_NAMES))
	rm -rf tests/__pycache__

.PHONY: all cuda-test portable test-c test-tokenizer test-chat-template test-python test check sanitize fuzz-smoke fuzz-libfuzzer clean
