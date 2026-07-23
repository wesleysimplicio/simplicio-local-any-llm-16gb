#ifndef SERVE_PROTOCOL_H
#define SERVE_PROTOCOL_H

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    SERVE_LINE_INVALID = -1,
    SERVE_LINE_TEXT = 0,
    SERVE_LINE_EMPTY,
    SERVE_LINE_RESET,
    SERVE_LINE_MORE,
    SERVE_LINE_PROMPT
} serve_line_kind;

typedef struct {
    serve_line_kind kind;
    size_t prompt_bytes;
    int max_tokens;
    double temperature;
    double top_p;
    int kv_slot;
    long long seed;
} serve_line;

static void serve_skip_spaces(char **cursor) {
    while (**cursor == ' ' || **cursor == '\t') (*cursor)++;
}

static int serve_parse_ull(char **cursor, unsigned long long *value) {
    serve_skip_spaces(cursor);
    if (**cursor < '0' || **cursor > '9') return 0;
    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(*cursor, &end, 10);
    if (errno == ERANGE || end == *cursor) return 0;
    *cursor = end;
    *value = parsed;
    return 1;
}

static int serve_parse_ll(char **cursor, long long *value) {
    serve_skip_spaces(cursor);
    if (**cursor != '-' && (**cursor < '0' || **cursor > '9')) return 0;
    errno = 0;
    char *end = NULL;
    long long parsed = strtoll(*cursor, &end, 10);
    if (errno == ERANGE || end == *cursor) return 0;
    *cursor = end;
    *value = parsed;
    return 1;
}

static int serve_parse_double(char **cursor, double *value) {
    serve_skip_spaces(cursor);
    errno = 0;
    char *end = NULL;
    double parsed = strtod(*cursor, &end);
    if (errno == ERANGE || end == *cursor) return 0;
    *cursor = end;
    *value = parsed;
    return 1;
}

/* Parse only the line header. Payload bytes are consumed separately by
 * run_serve after a successful length-prefixed PROMPT result. */
static int serve_parse_line(const char *input, size_t len, int kv_slots,
                            serve_line *out) {
    if (!input || !out || kv_slots < 1 || kv_slots > 16 || len >= 512 ||
        memchr(input, '\0', len)) return 0;
    char line[512];
    memcpy(line, input, len);
    line[len] = 0;
    memset(out, 0, sizeof(*out));
    out->kind = SERVE_LINE_TEXT;
    out->seed = -1;

    if (len == 0) {
        out->kind = SERVE_LINE_EMPTY;
        return 1;
    }
    if (!strcmp(line, "\x02RESET")) {
        out->kind = SERVE_LINE_RESET;
        return 1;
    }
    if (!strcmp(line, "\x02MORE")) {
        out->kind = SERVE_LINE_MORE;
        return 1;
    }
    if (strncmp(line, "\x02PROMPT ", 8) != 0) return 1;

    char *cursor = line + 8;
    unsigned long long prompt_bytes = 0, max_tokens = 0;
    unsigned long long slot = 0;
    if (!serve_parse_ull(&cursor, &prompt_bytes) ||
        !serve_parse_ull(&cursor, &max_tokens) ||
        !serve_parse_double(&cursor, &out->temperature) ||
        !serve_parse_double(&cursor, &out->top_p)) {
        out->kind = SERVE_LINE_INVALID;
        return 0;
    }
    serve_skip_spaces(&cursor);
    if (*cursor) {
        if (!serve_parse_ull(&cursor, &slot)) {
            out->kind = SERVE_LINE_INVALID;
            return 0;
        }
        serve_skip_spaces(&cursor);
        if (*cursor && !serve_parse_ll(&cursor, &out->seed)) {
            out->kind = SERVE_LINE_INVALID;
            return 0;
        }
    }
    serve_skip_spaces(&cursor);
    if (*cursor || prompt_bytes > (16u << 20) || max_tokens == 0 ||
        max_tokens > INT_MAX || out->temperature < 0 ||
        out->temperature > 2 || !isfinite(out->temperature) ||
        out->top_p <= 0 || out->top_p > 1 || !isfinite(out->top_p) ||
        slot >= (unsigned)kv_slots) {
        out->kind = SERVE_LINE_INVALID;
        return 0;
    }
    out->kind = SERVE_LINE_PROMPT;
    out->prompt_bytes = (size_t)prompt_bytes;
    out->max_tokens = (int)max_tokens;
    out->kv_slot = (int)slot;
    return 1;
}

#endif
