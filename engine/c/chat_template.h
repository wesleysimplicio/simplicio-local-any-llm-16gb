/* Family-specific chat rendering backed by chat_templates.json.
 * Unknown or ambiguous model metadata is rejected instead of falling back. */
#ifndef CHAT_TEMPLATE_H
#define CHAT_TEMPLATE_H

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

typedef struct {
    const char *prefix;
    const char *suffix;
    const char *strip_after;
    int trim;
} ChatRole;

typedef struct {
    const char *family;
    const char *bos;
    const char *default_system;
    const char *system_mode;
    const char *system_separator;
    const char *reasoning_effort_format;
    const char *generation_prompt;
    const char *thinking_generation_prompt;
    int supports_thinking;
    ChatRole system;
    ChatRole developer;
    ChatRole user;
    ChatRole assistant;
} ChatTemplate;

typedef struct {
    const char *role;
    const char *content;
} ChatMessage;

static int chat_error(char *error, size_t cap, const char *fmt, ...) {
    if (error && cap) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(error, cap, fmt, args);
        va_end(args);
    }
    return 0;
}

static char *chat_read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    char *text = (char *)malloc((size_t)length + 1);
    if (!text || fread(text, 1, (size_t)length, file) != (size_t)length) {
        free(text);
        fclose(file);
        return NULL;
    }
    text[length] = 0;
    fclose(file);
    return text;
}

static const char *chat_string(jval *object, const char *key) {
    jval *value = json_get(object, key);
    return value && value->t == J_STR ? value->str : NULL;
}

static int chat_bool(jval *object, const char *key, int *out) {
    jval *value = json_get(object, key);
    if (!value || value->t != J_BOOL) return 0;
    *out = value->boolean;
    return 1;
}

static int chat_load_role(ChatRole *out, jval *roles, const char *name) {
    jval *role = json_get(roles, name);
    int trim = 0;
    if (!role || role->t != J_OBJ || !chat_bool(role, "trim", &trim)) return 0;
    out->prefix = chat_string(role, "prefix");
    out->suffix = chat_string(role, "suffix");
    out->strip_after = chat_string(role, "strip_after");
    out->trim = trim;
    return out->prefix && out->suffix && out->strip_after;
}

static int chat_template_load(ChatTemplate *out, const char *path, const char *family,
                              char *error, size_t error_cap) {
    memset(out, 0, sizeof(*out));
    char *text = chat_read_file(path);
    if (!text) return chat_error(error, error_cap, "cannot read chat templates: %s", path);
    char *arena = NULL;
    jval *root = json_parse(text, &arena);
    jval *schema = json_get(root, "schema_version");
    jval *families = json_get(root, "families");
    jval *entry = families ? json_get(families, family) : NULL;
    if (!schema || schema->t != J_NUM || (int)schema->num != 1 || !entry ||
        entry->t != J_OBJ) {
        free(text);
        free(arena);
        return chat_error(error, error_cap, "unsupported chat family: %s", family);
    }
    out->family = family;
    out->bos = chat_string(entry, "bos");
    out->default_system = chat_string(entry, "default_system");
    out->system_mode = chat_string(entry, "system_mode");
    out->system_separator = chat_string(entry, "system_separator");
    out->reasoning_effort_format = chat_string(entry, "reasoning_effort_format");
    out->generation_prompt = chat_string(entry, "generation_prompt");
    out->thinking_generation_prompt = chat_string(entry, "thinking_generation_prompt");
    jval *roles = json_get(entry, "roles");
    int valid = chat_bool(entry, "supports_thinking", &out->supports_thinking) &&
        out->bos && out->default_system && out->system_mode && out->system_separator &&
        out->reasoning_effort_format && out->generation_prompt &&
        out->thinking_generation_prompt && roles &&
        (!strcmp(out->system_mode, "role") || !strcmp(out->system_mode, "prefix")) &&
        chat_load_role(&out->system, roles, "system") &&
        chat_load_role(&out->developer, roles, "developer") &&
        chat_load_role(&out->user, roles, "user") &&
        chat_load_role(&out->assistant, roles, "assistant");
    free(text);
    free(arena);
    if (!valid) {
        memset(out, 0, sizeof(*out));
        return chat_error(error, error_cap, "invalid chat template contract for %s", family);
    }
    return 1;
}

static int chat_contains_ci(const char *value, const char *alias) {
    size_t value_len = strlen(value), alias_len = strlen(alias);
    if (!alias_len || alias_len > value_len) return 0;
    for (size_t i = 0; i + alias_len <= value_len; i++) {
        size_t j = 0;
        while (j < alias_len &&
               tolower((unsigned char)value[i + j]) == tolower((unsigned char)alias[j])) j++;
        if (j == alias_len) return 1;
    }
    return 0;
}

static int chat_document_matches(jval *document, jval *aliases) {
    const char *keys[] = {"model_type", "tokenizer_class", "name_or_path"};
    for (size_t key = 0; key < sizeof(keys) / sizeof(keys[0]); key++) {
        const char *value = chat_string(document, keys[key]);
        if (!value) continue;
        for (int alias = 0; alias < aliases->len; alias++)
            if (aliases->kids[alias]->t == J_STR &&
                chat_contains_ci(value, aliases->kids[alias]->str)) return 1;
    }
    jval *architectures = json_get(document, "architectures");
    if (architectures && architectures->t == J_ARR)
        for (int item = 0; item < architectures->len; item++)
            if (architectures->kids[item]->t == J_STR)
                for (int alias = 0; alias < aliases->len; alias++)
                    if (aliases->kids[alias]->t == J_STR &&
                        chat_contains_ci(architectures->kids[item]->str,
                                         aliases->kids[alias]->str)) return 1;
    return 0;
}

static int chat_detect_family(const char *model_dir, const char *templates_path,
                              char *family, size_t family_cap, char *error,
                              size_t error_cap) {
    char *templates_text = chat_read_file(templates_path);
    if (!templates_text)
        return chat_error(error, error_cap, "cannot read chat templates: %s", templates_path);
    char *templates_arena = NULL;
    jval *templates = json_parse(templates_text, &templates_arena);
    jval *families = json_get(templates, "families");
    if (!families || families->t != J_OBJ) {
        free(templates_text);
        free(templates_arena);
        return chat_error(error, error_cap, "invalid chat template contract");
    }

    jval *documents[2] = {NULL, NULL};
    char *texts[2] = {NULL, NULL};
    char *arenas[2] = {NULL, NULL};
    const char *names[2] = {"config.json", "tokenizer_config.json"};
    for (int i = 0; i < 2; i++) {
        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", model_dir, names[i]);
        texts[i] = chat_read_file(path);
        if (texts[i]) documents[i] = json_parse(texts[i], &arenas[i]);
    }

    int matches = 0;
    const char *matched = NULL;
    for (int i = 0; i < families->len; i++) {
        jval *aliases = json_get(families->kids[i], "aliases");
        if (!aliases || aliases->t != J_ARR) continue;
        int hit = 0;
        for (int document = 0; document < 2 && !hit; document++)
            if (documents[document]) hit = chat_document_matches(documents[document], aliases);
        if (hit) {
            matches++;
            matched = families->keys[i];
        }
    }
    for (int i = 0; i < 2; i++) {
        free(texts[i]);
        free(arenas[i]);
    }
    size_t matched_length = matches == 1 ? strlen(matched) : 0;
    if (matches == 1 && matched_length + 1 <= family_cap) strcpy(family, matched);
    free(templates_text);
    free(templates_arena);
    if (matches != 1)
        return chat_error(error, error_cap, matches ? "ambiguous model family" :
                          "unknown model family");
    if (!family_cap || matched_length + 1 > family_cap)
        return chat_error(error, error_cap, "model family buffer is too small");
    return 1;
}

static const ChatRole *chat_role(const ChatTemplate *template, const char *role) {
    if (!strcmp(role, "system")) return &template->system;
    if (!strcmp(role, "developer")) return &template->developer;
    if (!strcmp(role, "user")) return &template->user;
    if (!strcmp(role, "assistant")) return &template->assistant;
    return NULL;
}

static int chat_append(char *out, size_t cap, size_t *length, const char *text,
                       size_t text_len) {
    if (text_len >= cap || *length >= cap - text_len) return 0;
    memcpy(out + *length, text, text_len);
    *length += text_len;
    out[*length] = 0;
    return 1;
}

static int chat_append_string(char *out, size_t cap, size_t *length, const char *text) {
    return chat_append(out, cap, length, text, strlen(text));
}

static int chat_append_message(const ChatRole *rule, const char *content, char *out,
                               size_t cap, size_t *length) {
    const char *start = content;
    if (rule->strip_after[0]) {
        const char *cursor = content;
        const char *last = NULL;
        while ((cursor = strstr(cursor, rule->strip_after)) != NULL) {
            last = cursor;
            cursor += strlen(rule->strip_after);
        }
        if (last) start = last + strlen(rule->strip_after);
    }
    size_t content_len = strlen(start);
    if (rule->trim) {
        while (content_len && isspace((unsigned char)*start)) {
            start++;
            content_len--;
        }
        while (content_len && isspace((unsigned char)start[content_len - 1])) content_len--;
    }
    return chat_append_string(out, cap, length, rule->prefix) &&
        chat_append(out, cap, length, start, content_len) &&
        chat_append_string(out, cap, length, rule->suffix);
}

static int chat_render(const ChatTemplate *template, const ChatMessage *messages,
                       int message_count, int enable_thinking, const char *reasoning_effort,
                       int add_generation_prompt, char *out, size_t cap, char *error,
                       size_t error_cap) {
    if (!template || !messages || message_count < 1 || !out || cap < 1)
        return chat_error(error, error_cap, "invalid chat render arguments");
    if (enable_thinking && !template->supports_thinking)
        return chat_error(error, error_cap, "thinking is not supported for family %s",
                          template->family);
    for (int i = 0; i < message_count; i++)
        if (!messages[i].role || !messages[i].content)
            return chat_error(error, error_cap, "message %d is missing role or content", i);
    size_t length = 0;
    out[0] = 0;
    if (!chat_append_string(out, cap, &length, template->bos))
        return chat_error(error, error_cap, "rendered prompt exceeds buffer");

    if (!strcmp(template->system_mode, "prefix")) {
        int emitted = 0;
        for (int i = 0; i < message_count; i++) {
            if (strcmp(messages[i].role, "system") &&
                strcmp(messages[i].role, "developer")) continue;
            if (emitted++ && !chat_append_string(out, cap, &length,
                                                  template->system_separator))
                return chat_error(error, error_cap, "rendered prompt exceeds buffer");
            if (!chat_append_string(out, cap, &length, messages[i].content))
                return chat_error(error, error_cap, "rendered prompt exceeds buffer");
        }
    } else if (strcmp(messages[0].role, "system") &&
               strcmp(messages[0].role, "developer") &&
               template->default_system[0]) {
        if (!chat_append_message(&template->system, template->default_system,
                                 out, cap, &length))
            return chat_error(error, error_cap, "rendered prompt exceeds buffer");
    }

    if (enable_thinking && template->reasoning_effort_format[0]) {
        const char *effort = reasoning_effort && !strcmp(reasoning_effort, "high") ?
            "High" : "Max";
        char reasoning[256];
        const char *placeholder = strstr(template->reasoning_effort_format, "{effort}");
        if (!placeholder)
            return chat_error(error, error_cap, "invalid reasoning effort format");
        int prefix = (int)(placeholder - template->reasoning_effort_format);
        int written = snprintf(reasoning, sizeof(reasoning), "%.*s%s%s", prefix,
            template->reasoning_effort_format, effort, placeholder + 8);
        if (written < 0 || (size_t)written >= sizeof(reasoning) ||
            !chat_append_message(&template->system, reasoning, out, cap, &length))
            return chat_error(error, error_cap, "rendered prompt exceeds buffer");
    }

    for (int i = 0; i < message_count; i++) {
        const ChatRole *role = chat_role(template, messages[i].role);
        if (!role)
            return chat_error(error, error_cap, "unsupported message role: %s",
                              messages[i].role);
        if (!strcmp(template->system_mode, "prefix") &&
            (!strcmp(messages[i].role, "system") ||
             !strcmp(messages[i].role, "developer"))) continue;
        if (!chat_append_message(role, messages[i].content, out, cap, &length))
            return chat_error(error, error_cap, "rendered prompt exceeds buffer");
    }
    if (add_generation_prompt &&
        !chat_append_string(out, cap, &length, enable_thinking ?
                           template->thinking_generation_prompt :
                           template->generation_prompt))
        return chat_error(error, error_cap, "rendered prompt exceeds buffer");
    return 1;
}

static int chat_render_user_turn(const ChatTemplate *template, const char *content,
                                 int first, int enable_thinking, char *out, size_t cap,
                                 char *error, size_t error_cap) {
    if (!template || !content || !out || cap < 1)
        return chat_error(error, error_cap, "invalid chat turn arguments");
    if (enable_thinking && !template->supports_thinking)
        return chat_error(error, error_cap, "thinking is not supported for family %s",
                          template->family);
    size_t length = 0;
    out[0] = 0;
    if (first && !chat_append_string(out, cap, &length, template->bos))
        return chat_error(error, error_cap, "rendered prompt exceeds buffer");
    if (first && !strcmp(template->system_mode, "role") &&
        template->default_system[0] &&
        !chat_append_message(&template->system, template->default_system,
                             out, cap, &length))
        return chat_error(error, error_cap, "rendered prompt exceeds buffer");
    if (!chat_append_message(&template->user, content, out, cap, &length) ||
        !chat_append_string(out, cap, &length, enable_thinking ?
                           template->thinking_generation_prompt :
                           template->generation_prompt))
        return chat_error(error, error_cap, "rendered prompt exceeds buffer");
    return 1;
}

#endif
