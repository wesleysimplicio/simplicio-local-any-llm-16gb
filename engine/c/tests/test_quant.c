/*
 * Regression vectors for colibri's production quantizers and CPU kernels.
 *
 * The expected packed bytes were generated independently with the documented
 * Python converter rules: symmetric per-row scale (absmax/qmax), round-to-even,
 * and little-endian packing of two int4 or four int2 values per byte.
 */
#define main colibri_glm_program_main
#include "../glm.c"
#undef main

#include <assert.h>

static void assert_close(float actual, float expected, float tolerance){
    if(fabsf(actual-expected)>tolerance){
        fprintf(stderr,"float mismatch: actual=%g expected=%g tolerance=%g\n",
                actual,expected,tolerance);
        assert(0);
    }
}

static void assert_bytes(const void *actual, const void *expected, size_t size){
    if(memcmp(actual,expected,size)){
        const uint8_t *bytes=actual;
        fprintf(stderr,"packed-byte mismatch:");
        for(size_t i=0;i<size;i++) fprintf(stderr," %02x",bytes[i]);
        fputc('\n',stderr);
        assert(0);
    }
}

static void test_packed_vectors(void){
    const float weights[8]={-8.f,-4.f,0.f,4.f,7.f,3.f,-2.f,1.f};
    const int8_t expected_i8[8]={-127,-64,0,64,111,48,-32,16};
    const uint8_t expected_i4[4]={0x51,0xb8,0xbe,0x96};
    const uint8_t expected_i2[2]={0xa9,0xab};
    int8_t q8[8]={0};
    uint8_t q4[4]={0}, q2[2]={0};
    float s8=0.f, s4=0.f, s2=0.f;

    quantize_rows(weights,q8,&s8,1,8,8);
    pack_int4(weights,q4,&s4,1,8,4);
    pack_int2(weights,q2,&s2,1,8,2);

    assert_bytes(q8,expected_i8,sizeof(q8));
    assert_bytes(q4,expected_i4,sizeof(q4));
    assert_bytes(q2,expected_i2,sizeof(q2));
    assert_close(s8,8.f/127.f,1e-7f);
    assert_close(s4,8.f/7.f,1e-7f);
    assert_close(s2,8.f,1e-7f);
}

static void test_quantized_matmul_against_scalar(void){
    const float weights[8]={-8.f,-4.f,0.f,4.f,7.f,3.f,-2.f,1.f};
    const float input[8]={.25f,-.5f,.75f,1.f,-.25f,.5f,-.75f,1.25f};
    int8_t q8[8], xq[8];
    uint8_t q4[4], q2[2];
    float s8, s4, s2, sx;
    float got_i8, got_i4, got_i2, got_i8_idot, got_i4_idot;
    float ref_i8=0.f, ref_i4=0.f, ref_i2=0.f;

    quantize_rows(weights,q8,&s8,1,8,8);
    pack_int4(weights,q4,&s4,1,8,4);
    pack_int2(weights,q2,&s2,1,8,2);
    for(int i=0;i<8;i++){
        ref_i8+=input[i]*(float)q8[i]*s8;
        ref_i4+=input[i]*(float)((int)((q4[i>>1]>>((i&1)*4))&15)-8)*s4;
        ref_i2+=input[i]*(float)((int)((q2[i>>2]>>((i&3)*2))&3)-2)*s2;
    }

    matmul_q(&got_i8,input,q8,&s8,1,8,1);
    matmul_i4(&got_i4,input,q4,&s4,1,8,1);
    matmul_i2(&got_i2,input,q2,&s2,1,8,1);
    assert_close(got_i8,ref_i8,1e-5f);
    assert_close(got_i4,ref_i4,1e-5f);
    assert_close(got_i2,ref_i2,1e-5f);

    sx=qrow_i8(input,xq,8);
    matmul_q_idot(&got_i8_idot,xq,&sx,q8,&s8,1,8,1);
    matmul_i4_idot(&got_i4_idot,xq,&sx,q4,&s4,1,8,1);
    /* Activation quantization adds at most one Q8 step per product here. */
    assert_close(got_i8_idot,ref_i8,.1f);
    assert_close(got_i4_idot,ref_i4,.1f);
}

int main(void){
    test_packed_vectors();
    test_quantized_matmul_against_scalar();
    puts("quant: packed vectors and scalar/IDOT parity OK");
    return 0;
}
