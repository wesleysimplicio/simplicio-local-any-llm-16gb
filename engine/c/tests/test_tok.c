/* Validate the C tokenizer against committed vectors or legacy TSV input.
 * usage: test_tok tokenizer.json [tok_vectors.json] */
#define _GNU_SOURCE
#include "../tok.h"

static int check_case(Tok *tokenizer, const char *name, const char *text, int text_len,
                      const int *expected, int expected_count) {
    int got[4096];
    if (text_len < 0 || text_len > 8191 || expected_count > 4096) {
        fprintf(stderr, "%s: vector exceeds test limits\n", name);
        return 0;
    }
    int got_count = tok_encode(tokenizer, text, text_len, got, 4096);
    int encode_ok = got_count == expected_count;
    for (int i = 0; i < got_count && encode_ok; i++) encode_ok = got[i] == expected[i];
    char decoded[8192];
    int decoded_len = tok_decode(tokenizer, got, got_count, decoded, 8191);
    int decode_ok = decoded_len == text_len && !memcmp(decoded, text, (size_t)text_len);
    if (!encode_ok || !decode_ok) {
        fprintf(stderr, "MISMATCH %s\n  exp(%d):", name, expected_count);
        for (int i = 0; i < expected_count; i++) fprintf(stderr, " %d", expected[i]);
        fprintf(stderr, "\n  got(%d):", got_count);
        for (int i = 0; i < got_count; i++) fprintf(stderr, " %d", got[i]);
        fprintf(stderr, "\n  decode_ok=%d\n", decode_ok);
    }
    return encode_ok && decode_ok;
}

static int run_json_vectors(Tok *tokenizer, const char *path) {
    long length = 0;
    char *text = tk_read_file(path, &length);
    char *arena = NULL;
    jval *document = json_parse(text, &arena);
    jval *cases = json_get(document, "cases");
    jval *family_value = json_get(document, "family");
    const char *family = family_value && family_value->t == J_STR ?
        family_value->str : "unknown";
    if (!cases || cases->t != J_ARR || cases->len < 40) {
        fprintf(stderr, "%s: tokenizer vectors require at least 40 cases\n", path);
        return 1;
    }
    int passed = 0;
    for (int i = 0; i < cases->len; i++) {
        jval *test_case = cases->kids[i];
        jval *name = json_get(test_case, "name");
        jval *input = json_get(test_case, "text");
        jval *ids = json_get(test_case, "ids");
        if (!name || name->t != J_STR || !input || input->t != J_STR ||
            !ids || ids->t != J_ARR || ids->len > 4096) {
            fprintf(stderr, "%s: invalid vector at index %d\n", path, i);
            return 1;
        }
        int expected[4096];
        for (int id = 0; id < ids->len; id++) {
            if (ids->kids[id]->t != J_NUM) {
                fprintf(stderr, "%s: invalid token id at index %d\n", name->str, id);
                return 1;
            }
            expected[id] = (int)ids->kids[id]->num;
        }
        passed += check_case(tokenizer, name->str, input->str, (int)strlen(input->str),
                             expected, ids->len);
    }
    printf("TOKENIZER %s: %d/%d encode+roundtrip\n", family, passed, cases->len);
    free(text);
    free(arena);
    return passed == cases->len ? 0 : 2;
}

static int run_legacy_tsv(Tok *tokenizer) {
    char *line = NULL;
    size_t cap = 0;
    ssize_t nr;
    int passed = 0, total = 0;
    while((nr=getline(&line,&cap,stdin))>=0){
        if(nr>0 && line[nr-1]=='\n'){ line[--nr]=0; }
        if(nr==0) continue;
        char *tab=strchr(line,'\t'); if(!tab) continue;
        *tab=0; const char *text=line; const char *idstr=tab+1;
        /* il testo puo' contenere \n e \t codificati come \\n \\t */
        char tbuf[4096]; int tn=0;
        for(const char *q=text; *q && tn<4095; q++){
            if(q[0]=='\\' && q[1]=='n'){ tbuf[tn++]='\n'; q++; }
            else if(q[0]=='\\' && q[1]=='t'){ tbuf[tn++]='\t'; q++; }
            else if(q[0]=='\\' && q[1]=='r'){ tbuf[tn++]='\r'; q++; }
            else if(q[0]=='\\' && q[1]=='\\'){ tbuf[tn++]='\\'; q++; }
            else tbuf[tn++]=*q;
        }
        tbuf[tn]=0;
        int exp[4096], ne=0;
        for(const char *q=idstr; *q; ){ while(*q==','||*q==' ')q++; if(!*q)break; exp[ne++]=atoi(q); while(*q&&*q!=',')q++; }
        total++;
        passed += check_case(tokenizer, text, tbuf, tn, exp, ne);
    }
    free(line);
    printf("TOKENIZER legacy: %d/%d encode+roundtrip\n", passed, total);
    return passed == total ? 0 : 2;
}

int main(int argc, char **argv){
    if(argc<2 || argc>3){
        fprintf(stderr,"usage: %s tokenizer.json [tok_vectors.json]\n",argv[0]);
        return 1;
    }
    Tok tokenizer;
    tok_load(&tokenizer, argv[1]);
    fprintf(stderr,"loaded: vocab_ids=%d specials=%d\n", tokenizer.n_ids, tokenizer.nsp);
    return argc == 3 ? run_json_vectors(&tokenizer, argv[2]) :
        run_legacy_tsv(&tokenizer);
}
