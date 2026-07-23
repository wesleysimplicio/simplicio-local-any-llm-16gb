#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../tok.h"

static Tok tokenizer;
static int tokenizer_ready;

static int ensure_tokenizer(void) {
    if (tokenizer_ready) return 1;
    const char *path = getenv("COLI_FUZZ_TOKENIZER");
    if (!path) path = "../../tests/fixtures/engine/tokenizer/minimal_tokenizer.json";
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    fclose(file);
    tok_load(&tokenizer, path);
    tokenizer_ready = 1;
    return 1;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > (64u << 10) || !ensure_tokenizer()) return 0;
    size_t capacity = size + 16;
    int *ids = (int *)malloc(capacity * sizeof(int));
    char *decoded = (char *)malloc(size * 4 + 64);
    if (!ids || !decoded) {
        free(ids);
        free(decoded);
        return 0;
    }
    int count = tok_encode(&tokenizer, (const char *)data, (int)size, ids,
                           (int)capacity);
    (void)tok_decode(&tokenizer, ids, count, decoded, (int)(size * 4 + 63));
    free(ids);
    free(decoded);
    return 0;
}
