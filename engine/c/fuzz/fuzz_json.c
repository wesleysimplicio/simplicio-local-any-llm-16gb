#include <stddef.h>
#include <stdint.h>

#include "../json.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > (1u << 20)) return 0;
    jval *root = json_parse_n((const char *)data, size, NULL);
    json_free(root);
    return 0;
}
