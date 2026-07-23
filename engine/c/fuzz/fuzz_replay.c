#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <seed>\n", argv[0]);
        return 2;
    }
    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        perror(argv[1]);
        return 2;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return 2;
    }
    long length = ftell(file);
    if (length < 0 || length > (1 << 20) || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        fprintf(stderr, "seed exceeds the 1 MiB replay budget\n");
        return 2;
    }
    uint8_t *data = (uint8_t *)malloc((size_t)length + 1);
    if (!data) {
        fclose(file);
        return 2;
    }
    size_t size = fread(data, 1, (size_t)length, file);
    fclose(file);
    if (size != (size_t)length) {
        free(data);
        return 2;
    }
    int result = LLVMFuzzerTestOneInput(data, size);
    /* Small deterministic mutation budget for hosts without libFuzzer. This
     * is a smoke check, not coverage-guided campaign evidence. */
    for (size_t round = 0; result == 0 && round < 64; round++) {
        if (size > 0) {
            size_t index = (round * 131u) % size;
            uint8_t saved = data[index];
            data[index] ^= (uint8_t)(1u << (round % 8));
            result = LLVMFuzzerTestOneInput(data, size);
            data[index] = saved;
            if (result != 0) break;
            result = LLVMFuzzerTestOneInput(data, round % (size + 1));
        }
        if (result == 0) {
            data[size] = (uint8_t)round;
            result = LLVMFuzzerTestOneInput(data, size + 1);
        }
    }
    free(data);
    return result;
}
