#include <cstdint>
#include <fstream>
#include <iterator>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *data,
                                      std::size_t size);

// AFL++ entry point for the existing libFuzzer-style harnesses. Keeping the
// harness contract shared means the same corpus and reference checks are used
// by both engines.
int main(int argc, char **argv) {
  if (argc != 2) {
    return 2;
  }
  std::ifstream input(argv[1], std::ios::binary);
  if (!input) {
    return 3;
  }
  const std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                                std::istreambuf_iterator<char>());
  const auto *data = reinterpret_cast<const std::uint8_t *>(bytes.data());
  return LLVMFuzzerTestOneInput(data, bytes.size());
}
