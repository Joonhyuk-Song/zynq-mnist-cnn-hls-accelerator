#include "mnist_ip.h"

static inline ap_uint<8> saturate_relu(int32_t acc) {
    #pragma HLS INLINE
    if (acc < 0)       return (ap_uint<8>)0;
    else if (acc > 255) return (ap_uint<8>)255;
    else               return (ap_uint<8>)acc;
}

// ====================================================================
// Conv1Fused – 28×28 → 14×14×2, fused conv + 2×2 max‑pool
// Writes padded 32‑bit scratch words (one per pooled pixel)
// ====================================================================
void CNN_Engine::Conv1Fused(
    hls::stream<ap_axiu<8,0,0,0>> &input_stream,
    ap_uint<32> weight_bram[WBRAM_DEPTH],
    ap_uint<32> scratch[SCRATCH_DEPTH])
{
    ap_uint<8> linebuf[3][IMG_DIM];
    #pragma HLS ARRAY_PARTITION variable=linebuf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=linebuf complete dim=2

    ap_uint<8> win[3][3];
    #pragma HLS ARRAY_PARTITION variable=win complete dim=0

    ap_uint<8> pend0[POOL1_DIM];
    ap_uint<8> pend1[POOL1_DIM];
    #pragma HLS ARRAY_PARTITION variable=pend0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=pend1 complete dim=1

    ap_uint<8> pool_row0[POOL1_DIM];
    ap_uint<8> pool_row1[POOL1_DIM];
    #pragma HLS ARRAY_PARTITION variable=pool_row0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=pool_row1 complete dim=1

    // Preload rows 0 and 1
    for (int y = 0; y < 2; y++)
        for (int x = 0; x < IMG_DIM; x++) {
            #pragma HLS PIPELINE II=1
            ap_axiu<8,0,0,0> tmp = input_stream.read();
            linebuf[y % 3][x] = tmp.data;
        }

    for (int out_y = 0; out_y < IMG_DIM; out_y++) {
        int load_y = out_y + 1;
        if (load_y >= 2 && load_y < IMG_DIM) {
            for (int x = 0; x < IMG_DIM; x++) {
                #pragma HLS PIPELINE II=1
                ap_axiu<8,0,0,0> tmp = input_stream.read();
                linebuf[load_y % 3][x] = tmp.data;
            }
        }

        for (int px = 0; px < POOL1_DIM; px++) {
            #pragma HLS PIPELINE II=1
            ap_uint<8> hmax0 = 0, hmax1 = 0;

            // Compute two adjacent columns and horizontally max‑pool
            for (int dx = 0; dx < 2; dx++) {
                #pragma HLS UNROLL
                int out_x = px * 2 + dx;

                // Fill 3×3 window
                for (int ky = -1; ky <= 1; ky++) {
                    int src_y = out_y + ky;
                    for (int kx = -1; kx <= 1; kx++) {
                        int src_x = out_x + kx;
                        ap_uint<8> pixel = 0;
                        if (src_y >= 0 && src_y < IMG_DIM &&
                            src_x >= 0 && src_x < IMG_DIM)
                            pixel = linebuf[src_y % 3][src_x];
                        win[ky+1][kx+1] = pixel;
                    }
                }

                // Conv for both filters
                for (int f = 0; f < 2; f++) {
                    #pragma HLS UNROLL
                    int32_t acc = (int32_t)read_int16(weight_bram, B_OFF_CONV1, f);
                    for (int ky = 0; ky < 3; ky++)
                        for (int kx = 0; kx < 3; kx++) {
                            #pragma HLS UNROLL
                            int8_t w = read_int8(weight_bram, W_OFF_CONV1,
                                                f*9 + ky*3 + kx);
                            acc += (int32_t)win[ky][kx] * (int32_t)w;
                        }
                    ap_uint<8> val = saturate_relu(acc);
                    if (f == 0 && val > hmax0) hmax0 = val;
                    if (f == 1 && val > hmax1) hmax1 = val;
                }
            }

            // Incremental vertical pooling (no cross‑px feedback)
            if (out_y % 2 == 0) {
                pend0[px] = hmax0;
                pend1[px] = hmax1;
            } else {
                ap_uint<8> m0 = (hmax0 > pend0[px]) ? hmax0 : pend0[px];
                ap_uint<8> m1 = (hmax1 > pend1[px]) ? hmax1 : pend1[px];
                pool_row0[px] = m0;
                pool_row1[px] = m1;
            }
        }

        // When a pooled row is complete, write to scratch (one 32‑bit word per pixel)
        if (out_y % 2 == 1) {
            int py = out_y / 2;
            for (int px = 0; px < POOL1_DIM; px++) {
                #pragma HLS PIPELINE II=1
                ap_uint<32> padded = 0;
                padded.range(7,0)  = pool_row0[px];
                padded.range(15,8) = pool_row1[px];
                scratch[py * POOL1_DIM + px] = padded;
            }
        }
    }
}

// ====================================================================
// Conv2Fused – 14×14 → 7×7×2
// Reads padded 32‑bit scratch, outputs padded 32‑bit scratch
// ====================================================================
void CNN_Engine::Conv2Fused(
    ap_uint<32> weight_bram[WBRAM_DEPTH],
    ap_uint<32> scratch[SCRATCH_DEPTH])
{
    ap_uint<8> feat_in[2][POOL1_DIM][POOL1_DIM];
    #pragma HLS ARRAY_PARTITION variable=feat_in complete dim=1

    // Unpack scratch into feat_in (lower 16 bits used)
    for (int py = 0; py < POOL1_DIM; py++) {
        for (int px = 0; px < POOL1_DIM; px++) {
            #pragma HLS PIPELINE II=1
            ap_uint<32> w = scratch[py * POOL1_DIM + px];
            feat_in[0][py][px] = w.range(7,0);
            feat_in[1][py][px] = w.range(15,8);
        }
    }

    ap_uint<8> conv_out[2][POOL1_DIM][POOL1_DIM];
    #pragma HLS ARRAY_PARTITION variable=conv_out complete dim=1

    // Convolution
    for (int out_y = 0; out_y < POOL1_DIM; out_y++) {
        for (int out_x = 0; out_x < POOL1_DIM; out_x++) {
            #pragma HLS PIPELINE II=1
            for (int f_out = 0; f_out < 2; f_out++) {
                #pragma HLS UNROLL
                int32_t acc = (int32_t)read_int16(weight_bram, B_OFF_CONV2, f_out);
                for (int f_in = 0; f_in < 2; f_in++) {
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            int src_y = out_y + ky;
                            int src_x = out_x + kx;
                            ap_uint<8> pixel = 0;
                            if (src_y >= 0 && src_y < POOL1_DIM &&
                                src_x >= 0 && src_x < POOL1_DIM)
                                pixel = feat_in[f_in][src_y][src_x];
                            int8_t w8 = read_int8(weight_bram, W_OFF_CONV2,
                                                 f_out*18 + f_in*9 +
                                                 (ky+1)*3 + (kx+1));
                            acc += (int32_t)pixel * (int32_t)w8;
                        }
                    }
                }
                conv_out[f_out][out_y][out_x] = saturate_relu(acc);
            }
        }
    }

    // Max‑pool 2×2 → 7×7, write padded 32‑bit scratch
    for (int py = 0; py < POOL2_DIM; py++) {
        for (int px = 0; px < POOL2_DIM; px++) {
            #pragma HLS PIPELINE II=1
            ap_uint<8> m0 = 0, m1 = 0;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    #pragma HLS UNROLL
                    ap_uint<8> v0 = conv_out[0][py*2+dy][px*2+dx];
                    ap_uint<8> v1 = conv_out[1][py*2+dy][px*2+dx];
                    if (v0 > m0) m0 = v0;
                    if (v1 > m1) m1 = v1;
                }
            }
            ap_uint<32> padded = 0;
            padded.range(7,0)  = m0;
            padded.range(15,8) = m1;
            scratch[py * POOL2_DIM + px] = padded;
        }
    }
}

// ====================================================================
// FullyConnected – reads padded 32‑bit scratch, writes 10 scores
// FC inner loop uses II=2 to avoid scheduling hang with cyclic weight BRAM
// ====================================================================
void CNN_Engine::FullyConnected(
    ap_uint<32> weight_bram[WBRAM_DEPTH],
    ap_uint<32> scratch[SCRATCH_DEPTH],
    ap_int<16> scores[NUM_CLASSES])
{
    ap_uint<8> flat[FC_INPUT_SIZE];
    #pragma HLS ARRAY_PARTITION variable=flat complete dim=1

    // Unpack scratch (7×7) into flat array
    int idx = 0;
    for (int y = 0; y < POOL2_DIM; y++) {
        for (int x = 0; x < POOL2_DIM; x++) {
            #pragma HLS PIPELINE II=1
            ap_uint<32> w = scratch[y * POOL2_DIM + x];
            flat[idx++] = w.range(7,0);
            flat[idx++] = w.range(15,8);
        }
    }

    for (int n = 0; n < NUM_CLASSES; n++) {
        #pragma HLS UNROLL
        int32_t acc = (int32_t)read_int16(weight_bram, B_OFF_FC, n);
        for (int i = 0; i < FC_INPUT_SIZE; i++) {
            #pragma HLS PIPELINE II=2   // II=2 avoids scheduling explosion
            int8_t w8 = read_int8(weight_bram, W_OFF_FC, n * FC_INPUT_SIZE + i);
            acc += (int32_t)flat[i] * (int32_t)w8;
        }
        scores[n] = (ap_int<16>)acc;
    }
}

ap_uint<4> CNN_Engine::Argmax(const ap_int<16> scores[NUM_CLASSES]) {
    ap_uint<4> max_idx = 0;
    ap_int<16> max_val = scores[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        #pragma HLS UNROLL
        if (scores[i] > max_val) { max_val = scores[i]; max_idx = i; }
    }
    return max_idx;
}

// ====================================================================
// Top‑level IP
// ====================================================================
void mnist_ip(
    hls::stream<ap_axiu<8,0,0,0>> &input_stream,
    ap_uint<4>                    &prediction,
    ap_uint<32>                   &ctrl_reg,
    ap_uint<32>                   &status_reg,
    bool                          &intr,
    ap_uint<32>                   &weight_addr,
    ap_uint<32>                   *ddr,
    ap_uint<32>                   scratch[SCRATCH_DEPTH])
{
    #pragma HLS INTERFACE axis          port=input_stream depth=784
    #pragma HLS INTERFACE s_axilite     port=prediction  bundle=ctrl
    #pragma HLS INTERFACE s_axilite     port=ctrl_reg    bundle=ctrl
    #pragma HLS INTERFACE s_axilite     port=status_reg  bundle=ctrl
    #pragma HLS INTERFACE s_axilite     port=intr        bundle=ctrl
    #pragma HLS INTERFACE s_axilite     port=weight_addr bundle=ctrl
    #pragma HLS INTERFACE s_axilite     port=return      bundle=ctrl
    #pragma HLS INTERFACE m_axi         port=ddr         offset=slave bundle=gmem depth=266
    #pragma HLS INTERFACE bram          port=scratch

    static ap_uint<32> weight_bram[WBRAM_DEPTH];
    #pragma HLS ARRAY_PARTITION variable=weight_bram cyclic factor=4 dim=1

    static enum { IDLE, LOAD_W, CONV1, WAIT_CONV2, CONV2, WAIT_FC, FC, DONE } state = IDLE;
    static bool conv1_done = false, conv2_done = false, fc_done = false;
    static ap_int<16> scores[NUM_CLASSES];
    static ap_uint<4>  final_pred = 0;

    bool start   = ctrl_reg[0];
    bool clr_irq = ctrl_reg[1];
    ap_uint<2> layer_sel = ctrl_reg.range(3,2);

    status_reg = 0;

    switch (state) {
        case IDLE:
            if (start && layer_sel == 0) {
                conv1_done = conv2_done = fc_done = false;
                state = LOAD_W;
            }
            break;

        case LOAD_W:
            for (int i = 0; i < WBRAM_DEPTH; i++)
                #pragma HLS PIPELINE II=1
                weight_bram[i] = ddr[weight_addr + i];
            state = CONV1;
            break;

        case CONV1:
            CNN_Engine::Conv1Fused(input_stream, weight_bram, scratch);
            conv1_done = true;
            state = WAIT_CONV2;
            break;

        case WAIT_CONV2:
            if (start && layer_sel == 1) {
                conv1_done = false;
                state = CONV2;
            }
            break;

        case CONV2:
            CNN_Engine::Conv2Fused(weight_bram, scratch);
            conv2_done = true;
            state = WAIT_FC;
            break;

        case WAIT_FC:
            if (start && layer_sel == 2) {
                conv2_done = false;
                state = FC;
            }
            break;

        case FC:
            CNN_Engine::FullyConnected(weight_bram, scratch, scores);
            final_pred = CNN_Engine::Argmax(scores);
            fc_done = true;
            state = DONE;
            break;

        case DONE:
            if (start && layer_sel == 0) {
                conv1_done = conv2_done = fc_done = false;
                state = LOAD_W;
            }
            break;
    }

    prediction = final_pred;

    status_reg[0] = (state == IDLE);
    status_reg[1] = conv1_done;
    status_reg[2] = conv2_done;
    status_reg[3] = fc_done;
    bool irq = conv1_done || conv2_done || fc_done;
    status_reg[4] = irq;
    intr = irq;

    if (clr_irq) {
        conv1_done = conv2_done = fc_done = false;
    }
}