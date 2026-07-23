#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../serve_protocol.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(void) {
    serve_line line;
    const char prompt[] = "\x02PROMPT 12 32 0.7 0.95 1 42";
    CHECK(serve_parse_line(prompt, sizeof(prompt) - 1, 2, &line));
    CHECK(line.kind == SERVE_LINE_PROMPT);
    CHECK(line.prompt_bytes == 12);
    CHECK(line.max_tokens == 32);
    CHECK(fabs(line.temperature - 0.7) < 1e-9);
    CHECK(fabs(line.top_p - 0.95) < 1e-9);
    CHECK(line.kv_slot == 1);
    CHECK(line.seed == 42);

    const char reset[] = "\x02RESET";
    CHECK(serve_parse_line(reset, sizeof(reset) - 1, 1, &line));
    CHECK(line.kind == SERVE_LINE_RESET);

    const char too_large[] = "\x02PROMPT 16777217 1 0 1";
    CHECK(!serve_parse_line(too_large, sizeof(too_large) - 1, 1, &line));
    const char nan_temp[] = "\x02PROMPT 1 1 nan 1";
    CHECK(!serve_parse_line(nan_temp, sizeof(nan_temp) - 1, 1, &line));
    const char trailing[] = "\x02PROMPT 1 1 0.7 1 trailing";
    CHECK(!serve_parse_line(trailing, sizeof(trailing) - 1, 1, &line));

    puts("serve protocol tests: ok");
    return 0;
}
