#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "../serve_protocol.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size >= 511) return 0;
    char line[512];
    size_t length = size;
    if (size >= 7 && !memcmp(data, "PROMPT ", 7)) {
        line[0] = '\x02';
        memcpy(line + 1, data, size);
        length++;
    } else {
        memcpy(line, data, size);
    }
    serve_line parsed;
    (void)serve_parse_line(line, length, 16, &parsed);
    return 0;
}
