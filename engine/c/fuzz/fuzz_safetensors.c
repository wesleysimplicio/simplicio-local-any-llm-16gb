#include <stddef.h>
#include <stdint.h>

#include "../st.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    int tensors = 0;
    (void)st_validate_header_json(data, size, &tensors);
    return 0;
}
