#ifndef MNIST_IP_H
#define MNIST_IP_H

#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>

// Image / pooling dimensions
#define IMG_DIM         28
#define POOL1_DIM       14
#define POOL2_DIM        7
#define FC_INPUT_SIZE   98
#define NUM_CLASSES     10

// Weight BRAM (32‑bit wide)
#define WBRAM_DEPTH     266

// Offsets inside the shared weight BRAM
#define W_OFF_CONV1     0
#define B_OFF_CONV1     5
#define W_OFF_CONV2     6
#define B_OFF_CONV2    15
#define W_OFF_FC       16
#define B_OFF_FC      261

// Scratch BRAM – 32‑bit wide, padded (one 32‑bit word per spatial pixel)
#define SCRATCH_DEPTH   196   // 14*14 = 196, enough for Conv1 output; Conv2 uses 49

// Packed BRAM helpers (read int8 / int16 from 32‑bit words)
static inline int8_t read_int8(const ap_uint<32> *bram, int word_off, int flat_idx) {
    #pragma HLS INLINE
    ap_uint<32> data = bram[word_off + flat_idx/4];
    int byte = flat_idx % 4;
    return (int8_t)(data.range(byte*8+7, byte*8));
}

static inline int16_t read_int16(const ap_uint<32> *bram, int word_off, int flat_idx) {
    #pragma HLS INLINE
    ap_uint<32> data = bram[word_off + flat_idx/2];
    int half = flat_idx % 2;
    return (int16_t)(data.range(half*16+15, half*16));
}

// No pack_pair / pack_single needed – we just store a padded 32‑bit word

class CNN_Engine {
public:
    static void Conv1Fused(hls::stream<ap_axiu<8,0,0,0>> &input_stream,
                           ap_uint<32> weight_bram[WBRAM_DEPTH],
                           ap_uint<32> scratch[SCRATCH_DEPTH]);

    static void Conv2Fused(ap_uint<32> weight_bram[WBRAM_DEPTH],
                           ap_uint<32> scratch[SCRATCH_DEPTH]);

    static void FullyConnected(ap_uint<32> weight_bram[WBRAM_DEPTH],
                               ap_uint<32> scratch[SCRATCH_DEPTH],
                               ap_int<16> scores[NUM_CLASSES]);

    static ap_uint<4> Argmax(const ap_int<16> scores[NUM_CLASSES]);
};

void mnist_ip(hls::stream<ap_axiu<8,0,0,0>> &input_stream,
              ap_uint<4>                    &prediction,
              ap_uint<32>                   &ctrl_reg,
              ap_uint<32>                   &status_reg,
              bool                          &intr,
              ap_uint<32>                   &weight_addr,
              ap_uint<32>                   *ddr,
              ap_uint<32>                   scratch[SCRATCH_DEPTH]);

#endif