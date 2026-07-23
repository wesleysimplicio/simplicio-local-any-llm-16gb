/* Parser JSON minimale, header-only. Serve per:
 *  - l'header dei file safetensors (un grande oggetto nome->{dtype,shape,data_offsets})
 *  - ref.json (per leggere prompt_ids / full_ids)
 * Non e' completo (niente unicode \uXXXX, niente notazione esotica) ma copre cio' che serve. */
#ifndef JSON_H
#define JSON_H
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype;

typedef struct jval {
    jtype t;
    double num;            /* J_NUM */
    int    boolean;        /* J_BOOL */
    char  *str;            /* J_STR (NUL-terminata, dentro l'arena) */
    /* array: figli in [0..len); oggetto: chiavi[] e figli[] in parallelo */
    struct jval **kids;
    char        **keys;    /* solo per J_OBJ */
    int           len;
} jval;

typedef struct {
    const char *s;
    const char *end;
    char       *arena;     /* buffer per le stringhe smontate */
    size_t      acap, aoff;
    unsigned    depth;
    int         error;
} jparser;

enum { JSON_MAX_DEPTH = 128 };

static void j_ws(jparser *p) {
    while (p->s < p->end && isspace((unsigned char)*p->s)) p->s++;
}

static jval *j_new(jtype t) {
    jval *v = (jval *)calloc(1, sizeof(jval));
    if (v) v->t = t;
    return v;
}

static jval *j_parse_val(jparser *p);
static void json_free(jval *v);

static char *j_parse_str_raw(jparser *p) {
    if (p->s >= p->end || *p->s != '"') {
        p->error = 1;
        return NULL;
    }
    p->s++;
    size_t cap = (size_t)(p->end - p->s) + 1;
    char *tmp = (char *)malloc(cap);
    size_t n = 0;
    if (!tmp) {
        p->error = 1;
        return NULL;
    }
    #define J_PUT(ch) do { tmp[n++] = (char)(ch); } while (0)
    while (p->s < p->end && *p->s != '"') {
        char c = *p->s++;
        if ((unsigned char)c < 0x20) {
            p->error = 1;
            break;
        }
        if (c == '\\') {
            if (p->s >= p->end) {
                p->error = 1;
                break;
            }
            char e = *p->s++;
            switch (e) {
                case 'n': c = '\n'; break; case 't': c = '\t'; break;
                case 'r': c = '\r'; break; case 'b': c = '\b'; break;
                case 'f': c = '\f'; break; case '/': c = '/'; break;
                case '\\': c = '\\'; break; case '"': c = '"'; break;
                case 'u': {  /* \uXXXX -> codepoint UTF-8 (con coppie surrogate) */
                    if ((size_t)(p->end - p->s) < 4) {
                        p->error = 1;
                        break;
                    }
                    for (int i = 0; i < 4; i++) {
                        if (!isxdigit((unsigned char)p->s[i])) {
                            p->error = 1;
                            break;
                        }
                    }
                    if (p->error) break;
                    unsigned cp = (unsigned)strtoul((char[]){p->s[0],p->s[1],p->s[2],p->s[3],0}, NULL, 16);
                    p->s += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF &&
                        (size_t)(p->end - p->s) >= 6 &&
                        p->s[0]=='\\' && p->s[1]=='u') {
                        for (int i = 2; i < 6; i++) {
                            if (!isxdigit((unsigned char)p->s[i])) {
                                p->error = 1;
                                break;
                            }
                        }
                        if (p->error) break;
                        unsigned lo = (unsigned)strtoul((char[]){p->s[2],p->s[3],p->s[4],p->s[5],0}, NULL, 16);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) { cp = 0x10000 + ((cp-0xD800)<<10) + (lo-0xDC00); p->s += 6; }
                    }
                    if ((cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF) {
                        p->error = 1;
                        break;
                    }
                    if (cp < 0x80) { J_PUT(cp); }
                    else if (cp < 0x800) { J_PUT(0xC0|(cp>>6)); J_PUT(0x80|(cp&0x3F)); }
                    else if (cp < 0x10000) { J_PUT(0xE0|(cp>>12)); J_PUT(0x80|((cp>>6)&0x3F)); J_PUT(0x80|(cp&0x3F)); }
                    else { J_PUT(0xF0|(cp>>18)); J_PUT(0x80|((cp>>12)&0x3F)); J_PUT(0x80|((cp>>6)&0x3F)); J_PUT(0x80|(cp&0x3F)); }
                    continue;
                }
                default: p->error = 1; break;
            }
            if (p->error) break;
        }
        J_PUT(c);
    }
    #undef J_PUT
    if (p->s >= p->end || *p->s != '"') p->error = 1;
    else p->s++;
    if (p->error) {
        free(tmp);
        return NULL;
    }
    tmp[n] = 0;
    return tmp;
}

static jval *j_parse_val(jparser *p) {
    j_ws(p);
    if (p->error || p->s >= p->end || p->depth >= JSON_MAX_DEPTH) {
        p->error = 1;
        return NULL;
    }
    p->depth++;
    char c = *p->s;
    jval *result = NULL;
    if (c == '"') {
        result = j_new(J_STR);
        if (result) result->str = j_parse_str_raw(p);
        if (!result || !result->str) p->error = 1;
        goto done;
    }
    if (c == '{') {
        p->s++; jval *v = j_new(J_OBJ);
        int cap = 8;
        if (!v) { p->error = 1; goto done; }
        v->keys = malloc((size_t)cap * sizeof(char*));
        v->kids = malloc((size_t)cap * sizeof(jval*));
        if (!v->keys || !v->kids) { p->error = 1; result = v; goto done; }
        j_ws(p);
        if (p->s < p->end && *p->s == '}') { p->s++; result = v; goto done; }
        for (;;) {
            j_ws(p);
            char *key = j_parse_str_raw(p);
            j_ws(p);
            if (!key || p->s >= p->end || *p->s != ':') {
                free(key); p->error = 1; break;
            }
            p->s++;
            jval *val = j_parse_val(p);
            if (!val || p->error) { free(key); json_free(val); break; }
            if (v->len == cap) {
                if (cap > INT_MAX / 2) { free(key); p->error = 1; break; }
                cap *= 2;
                char **keys = realloc(v->keys, (size_t)cap*sizeof(char*));
                if (!keys) { free(key); p->error = 1; break; }
                v->keys = keys;
                jval **kids = realloc(v->kids, (size_t)cap*sizeof(jval*));
                if (!kids) { free(key); p->error = 1; break; }
                v->kids = kids;
            }
            v->keys[v->len] = key; v->kids[v->len] = val; v->len++;
            j_ws(p);
            if (p->s < p->end && *p->s == ',') { p->s++; continue; }
            if (p->s < p->end && *p->s == '}') { p->s++; break; }
            p->error = 1;
            break;
        }
        result = v;
        goto done;
    }
    if (c == '[') {
        p->s++; jval *v = j_new(J_ARR);
        int cap = 8;
        if (!v) { p->error = 1; goto done; }
        v->kids = malloc((size_t)cap * sizeof(jval*));
        if (!v->kids) { p->error = 1; result = v; goto done; }
        j_ws(p);
        if (p->s < p->end && *p->s == ']') { p->s++; result = v; goto done; }
        for (;;) {
            jval *val = j_parse_val(p);
            if (!val || p->error) { json_free(val); break; }
            if (v->len == cap) {
                if (cap > INT_MAX / 2) { p->error = 1; break; }
                cap *= 2;
                jval **kids = realloc(v->kids, (size_t)cap*sizeof(jval*));
                if (!kids) { p->error = 1; break; }
                v->kids = kids;
            }
            v->kids[v->len++] = val;
            j_ws(p);
            if (p->s < p->end && *p->s == ',') { p->s++; continue; }
            if (p->s < p->end && *p->s == ']') { p->s++; break; }
            p->error = 1;
            break;
        }
        result = v;
        goto done;
    }
    if ((size_t)(p->end-p->s) >= 4 && !memcmp(p->s,"true",4)) {
        p->s += 4; result = j_new(J_BOOL);
        if (result) result->boolean = 1; else p->error = 1;
        goto done;
    }
    if ((size_t)(p->end-p->s) >= 5 && !memcmp(p->s,"false",5)) {
        p->s += 5; result = j_new(J_BOOL);
        if (result) result->boolean = 0; else p->error = 1;
        goto done;
    }
    if ((size_t)(p->end-p->s) >= 4 && !memcmp(p->s,"null",4)) {
        p->s += 4; result = j_new(J_NULL);
        if (!result) p->error = 1;
        goto done;
    }
    /* numero */
    {
        if (c != '-' && (c < '0' || c > '9')) {
            p->error = 1;
            goto done;
        }
        char *end = NULL;
        errno = 0;
        double d = strtod(p->s, &end);
        if (end == p->s || end > p->end || errno == ERANGE || !isfinite(d)) {
            p->error = 1;
            goto done;
        }
        p->s = end;
        result = j_new(J_NUM);
        if (result) result->num = d; else p->error = 1;
    }
done:
    p->depth--;
    return result;
}

/* API */
static void json_free(jval *v) {
    if (!v) return;
    if (v->t == J_OBJ) {
        for (int i = 0; i < v->len; i++) free(v->keys[i]);
    }
    if (v->t == J_STR) free(v->str);
    for (int i = 0; i < v->len; i++) json_free(v->kids[i]);
    free(v->keys);
    free(v->kids);
    free(v);
}

static jval *json_parse_n(const char *text, size_t text_len, char **arena_out) {
    if (arena_out) *arena_out = NULL;
    if (!text || text_len == SIZE_MAX) return NULL;
    char *copy = (char *)malloc(text_len + 1);
    if (!copy) return NULL;
    memcpy(copy, text, text_len);
    copy[text_len] = 0;
    jparser p = { copy, copy + text_len, NULL, 0, 0, 0, 0 };
    jval *v = j_parse_val(&p);
    j_ws(&p);
    if (p.error || p.s != p.end) {
        json_free(v);
        v = NULL;
    }
    free(copy);
    return v;
}

static jval *json_parse(const char *text, char **arena_out) {
    return text ? json_parse_n(text, strlen(text), arena_out) : NULL;
}

static jval *json_get(jval *o, const char *key) {
    if (!o || o->t != J_OBJ) return NULL;
    for (int i = 0; i < o->len; i++) if (strcmp(o->keys[i], key) == 0) return o->kids[i];
    return NULL;
}

#endif
