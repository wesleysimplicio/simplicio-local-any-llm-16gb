#define _GNU_SOURCE
#include "../chat_template.h"

static int boolean_value(jval *object, const char *key) {
    jval *value = json_get(object, key);
    return value && value->t == J_BOOL ? value->boolean : 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s chat_templates.json chat_vectors.json\n", argv[0]);
        return 1;
    }
    char *vectors_text = chat_read_file(argv[2]);
    if (!vectors_text) {
        fprintf(stderr, "cannot read vectors: %s\n", argv[2]);
        return 1;
    }
    char *arena = NULL;
    jval *document = json_parse(vectors_text, &arena);
    jval *cases = json_get(document, "cases");
    if (!cases || cases->t != J_ARR || cases->len < 9) {
        fprintf(stderr, "chat vectors must contain at least 9 cases\n");
        return 1;
    }

    int passed = 0;
    for (int i = 0; i < cases->len; i++) {
        jval *test_case = cases->kids[i];
        const char *name = chat_string(test_case, "name");
        const char *family = chat_string(test_case, "family");
        const char *expected = chat_string(test_case, "expected");
        jval *messages = json_get(test_case, "messages");
        jval *effort = json_get(test_case, "reasoning_effort");
        if (!name || !family || !expected || !messages || messages->t != J_ARR) {
            fprintf(stderr, "invalid chat vector at index %d\n", i);
            return 1;
        }
        ChatTemplate template;
        char error[256];
        if (!chat_template_load(&template, argv[1], family, error, sizeof(error))) {
            fprintf(stderr, "%s: %s\n", name, error);
            return 1;
        }
        ChatMessage *input = calloc((size_t)messages->len, sizeof(*input));
        for (int message = 0; message < messages->len; message++) {
            input[message].role = chat_string(messages->kids[message], "role");
            input[message].content = chat_string(messages->kids[message], "content");
            if (!input[message].role || !input[message].content) {
                fprintf(stderr, "%s: invalid message\n", name);
                return 1;
            }
        }
        char actual[16384];
        int ok = chat_render(&template, input, messages->len,
            boolean_value(test_case, "enable_thinking"),
            effort && effort->t == J_STR ? effort->str : NULL,
            boolean_value(test_case, "add_generation_prompt"),
            actual, sizeof(actual), error, sizeof(error));
        free(input);
        if (!ok || strcmp(actual, expected)) {
            fprintf(stderr, "%s: %s\nexpected: %s\nactual:   %s\n", name,
                ok ? "prompt mismatch" : error, expected, actual);
            return 2;
        }
        passed++;
    }

    ChatTemplate kimi;
    char error[256], output[128];
    ChatMessage message = {"user", "Hi"};
    if (!chat_template_load(&kimi, argv[1], "kimi", error, sizeof(error)) ||
        chat_render(&kimi, &message, 1, 1, NULL, 1, output, sizeof(output),
                    error, sizeof(error))) {
        fprintf(stderr, "Kimi thinking must fail closed\n");
        return 2;
    }
    if (chat_template_load(&kimi, argv[1], "unknown", error, sizeof(error))) {
        fprintf(stderr, "unknown family must fail closed\n");
        return 2;
    }
    const char *families[] = {"glm", "deepseek", "kimi"};
    for (int i = 0; i < 3; i++) {
        char model_path[256], detected[32];
        snprintf(model_path, sizeof(model_path), "tests/fixtures/chat_models/%s",
                 families[i]);
        if (!chat_detect_family(model_path, argv[1], detected, sizeof(detected),
                                error, sizeof(error)) ||
            strcmp(detected, families[i])) {
            fprintf(stderr, "family detection failed for %s: %s\n", families[i], error);
            return 2;
        }
    }
    char detected[32];
    if (chat_detect_family("tests/fixtures/chat_models/unknown", argv[1], detected,
                           sizeof(detected), error, sizeof(error)) ||
        chat_detect_family("tests/fixtures/chat_models/ambiguous", argv[1], detected,
                           sizeof(detected), error, sizeof(error))) {
        fprintf(stderr, "unknown and ambiguous metadata must fail closed\n");
        return 2;
    }
    printf("CHAT TEMPLATE: %d/%d\n", passed, cases->len);
    free(vectors_text);
    free(arena);
    return 0;
}
