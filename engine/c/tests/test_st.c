#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "../st.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(void) {
    CHECK(bf16_to_f32(0x3f80) == 1.0f);
    CHECK(bf16_to_f32(0xc020) == -2.5f);
    CHECK(f16_to_f32(0x3c00) == 1.0f);
    CHECK(f16_to_f32(0xc100) == -2.5f);
    CHECK(f16_to_f32(0x0001) > 0.0f);
    CHECK(isinf(f16_to_f32(0x7c00)));
    CHECK(st_hash("tensor.weight") == st_hash("tensor.weight"));
    CHECK(st_hash("tensor.weight") != st_hash("tensor.bias"));
    const char valid[] =
        "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2,2],"
        "\"data_offsets\":[0,16]}}";
    int tensors = 0;
    CHECK(st_validate_header_json(valid, sizeof(valid) - 1, &tensors));
    CHECK(tensors == 1);
    const char reversed[] =
        "{\"weight\":{\"dtype\":\"F16\",\"shape\":[4],"
        "\"data_offsets\":[8,2]}}";
    CHECK(!st_validate_header_json(reversed, sizeof(reversed) - 1, NULL));
    const char overflow[] =
        "{\"weight\":{\"dtype\":\"BF16\","
        "\"shape\":[9223372036854775807,2],\"data_offsets\":[0,1]}}";
    CHECK(!st_validate_header_json(overflow, sizeof(overflow) - 1, NULL));

    puts("safetensors primitive tests: ok");
    return 0;
}
