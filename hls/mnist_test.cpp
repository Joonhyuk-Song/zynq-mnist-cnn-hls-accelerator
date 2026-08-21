#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "mnist_ip.h"

/* ============================================================
 * Hex parsers – unchanged from your original testbench
 * ============================================================ */
static inline bool is_hex_digit(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
static inline int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}
static bool looks_like_hex(const char *p) {
    const char *s = p;
    while (*s && is_hex_digit(*s)) {
        if ((*s >= 'a' && *s <= 'f') || (*s >= 'A' && *s <= 'F')) return true;
        s++;
    }
    return false;
}

int read_int8_mem(const char *filename, int8_t *arr, int max_count) {
    FILE *f = fopen(filename, "r");
    if (!f) { printf("ERROR: Cannot open %s\n", filename); return -1; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    char *buffer = (char*)malloc(fsize + 1);
    if (!buffer) { fclose(f); return -1; }
    fread(buffer, 1, fsize, f); buffer[fsize] = '\0'; fclose(f);
    int idx = 0; char *p = buffer;
    while (*p && idx < max_count) {
        while (*p && (isspace(*p) || *p == ',' || *p == ';' || *p == '[' || *p == ']' ||
                      *p == '{' || *p == '}' || *p == ':' || *p == '(' || *p == ')' ||
                      *p == '/' || *p == '"' || *p == '\'')) p++;
        if (!*p) break;
        if (*p == '/' && *(p+1) == '/') { while (*p && *p != '\n') p++; continue; }
        int val = 0; bool is_negative = false, parsed = false;
        if (*p == '-') { is_negative = true; p++; }
        if (*p == '0' && (*(p+1) == 'x' || *(p+1) == 'X')) {
            p += 2;
            if (is_hex_digit(*p)) {
                while (*p && is_hex_digit(*p)) { val = val * 16 + hex_val(*p); p++; }
                if (is_negative) val = -val; parsed = true;
            }
        } else if (isdigit(*p)) {
            if (looks_like_hex(p)) {
                while (*p && is_hex_digit(*p)) { val = val * 16 + hex_val(*p); p++; }
            } else {
                while (*p && isdigit(*p)) { val = val * 10 + (*p - '0'); p++; }
            }
            if (is_negative) val = -val; parsed = true;
        } else if ((*p >= 'a' && *p <= 'f') || (*p >= 'A' && *p <= 'F')) {
            while (*p && is_hex_digit(*p)) { val = val * 16 + hex_val(*p); p++; }
            if (is_negative) val = -val; parsed = true;
        } else p++;
        if (parsed) arr[idx++] = (int8_t)val;
    }
    free(buffer);
    return idx;
}

int read_int16_mem(const char *filename, int16_t *arr, int max_count) {
    FILE *f = fopen(filename, "r");
    if (!f) { printf("ERROR: Cannot open %s\n", filename); return -1; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    char *buffer = (char*)malloc(fsize + 1);
    if (!buffer) { fclose(f); return -1; }
    fread(buffer, 1, fsize, f); buffer[fsize] = '\0'; fclose(f);
    int idx = 0; char *p = buffer;
    while (*p && idx < max_count) {
        while (*p && (isspace(*p) || *p == ',' || *p == ';' || *p == '[' || *p == ']' ||
                      *p == '{' || *p == '}' || *p == ':' || *p == '(' || *p == ')' ||
                      *p == '/' || *p == '"' || *p == '\'')) p++;
        if (!*p) break;
        if (*p == '/' && *(p+1) == '/') { while (*p && *p != '\n') p++; continue; }
        int val = 0; bool is_negative = false, parsed = false;
        if (*p == '-') { is_negative = true; p++; }
        if (*p == '0' && (*(p+1) == 'x' || *(p+1) == 'X')) {
            p += 2;
            if (is_hex_digit(*p)) {
                while (*p && is_hex_digit(*p)) { val = val * 16 + hex_val(*p); p++; }
                if (is_negative) val = -val; parsed = true;
            }
        } else if (isdigit(*p)) {
            if (looks_like_hex(p)) {
                while (*p && is_hex_digit(*p)) { val = val * 16 + hex_val(*p); p++; }
            } else {
                while (*p && isdigit(*p)) { val = val * 10 + (*p - '0'); p++; }
            }
            if (is_negative) val = -val; parsed = true;
        } else if ((*p >= 'a' && *p <= 'f') || (*p >= 'A' && *p <= 'F')) {
            while (*p && is_hex_digit(*p)) { val = val * 16 + hex_val(*p); p++; }
            if (is_negative) val = -val; parsed = true;
        } else p++;
        if (parsed) arr[idx++] = (int16_t)val;
    }
    free(buffer);
    return idx;
}

/* ============================================================
 * ORIGINAL load_weights – exactly as in your passing testbench
 * (packs .mem files into weight_bram)
 * ============================================================ */
bool load_weights(const char *folder, ap_uint<32> weight_bram[WBRAM_DEPTH]) {
    char path[512];
    int8_t  conv1_w[18];   int16_t conv1_b[2];
    int8_t  conv2_w[36];   int16_t conv2_b[2];
    int8_t  fc_w[980];     int16_t fc_b[10];

    sprintf(path, "%s/conv1_weights.mem", folder);
    if (read_int8_mem(path, conv1_w, 18) != 18) return false;
    sprintf(path, "%s/conv1_bias.mem", folder);
    if (read_int16_mem(path, conv1_b, 2) != 2) return false;
    sprintf(path, "%s/conv2_weights.mem", folder);
    if (read_int8_mem(path, conv2_w, 36) != 36) return false;
    sprintf(path, "%s/conv2_bias.mem", folder);
    if (read_int16_mem(path, conv2_b, 2) != 2) return false;
    sprintf(path, "%s/fc_weights.mem", folder);
    if (read_int8_mem(path, fc_w, 980) != 980) return false;
    sprintf(path, "%s/fc_bias.mem", folder);
    if (read_int16_mem(path, fc_b, 10) != 10) return false;

    memset(weight_bram, 0, WBRAM_DEPTH * sizeof(ap_uint<32>));
    for (int i = 0; i < 18; i++) {
        int word = W_OFF_CONV1 + i/4;
        int byte = i % 4;
        weight_bram[word] |= ((ap_uint<32>)(uint8_t)conv1_w[i]) << (byte*8);
    }
    for (int i = 0; i < 2; i++) {
        weight_bram[B_OFF_CONV1] |= ((ap_uint<32>)(uint16_t)conv1_b[i]) << (i*16);
    }
    for (int i = 0; i < 36; i++) {
        int word = W_OFF_CONV2 + i/4;
        int byte = i % 4;
        weight_bram[word] |= ((ap_uint<32>)(uint8_t)conv2_w[i]) << (byte*8);
    }
    for (int i = 0; i < 2; i++) {
        weight_bram[B_OFF_CONV2] |= ((ap_uint<32>)(uint16_t)conv2_b[i]) << (i*16);
    }
    for (int i = 0; i < 980; i++) {
        int word = W_OFF_FC + i/4;
        int byte = i % 4;
        weight_bram[word] |= ((ap_uint<32>)(uint8_t)fc_w[i]) << (byte*8);
    }
    for (int i = 0; i < 10; i++) {
        int word = B_OFF_FC + i/2;
        int half = i % 2;
        weight_bram[word] |= ((ap_uint<32>)(uint16_t)fc_b[i]) << (half*16);
    }
    printf("*** weight_bram built successfully ***\n\n");
    return true;
}

/* ============================================================
 * BMP loader – 8‑bit grayscale
 * ============================================================ */
bool load_bmp_grayscale(const char *filename, uint8_t img[IMG_DIM][IMG_DIM]) {
    FILE *f = fopen(filename, "rb");
    if (!f) { printf("ERROR: Cannot open %s\n", filename); return false; }
    uint16_t bfType; fread(&bfType, sizeof(uint16_t), 1, f);
    if (bfType != 0x4D42) { printf("ERROR: Not a BMP file.\n"); fclose(f); return false; }
    fseek(f, 10, SEEK_SET); uint32_t bfOffBits; fread(&bfOffBits, sizeof(uint32_t), 1, f);
    fseek(f, 18, SEEK_SET); int32_t width, height;
    fread(&width, sizeof(int32_t), 1, f); fread(&height, sizeof(int32_t), 1, f);
    if (width != IMG_DIM || height != IMG_DIM) {
        printf("ERROR: BMP must be %dx%d (got %dx%d)\n", IMG_DIM, IMG_DIM, width, height);
        fclose(f); return false;
    }
    fseek(f, 28, SEEK_SET); uint16_t bitCount; fread(&bitCount, sizeof(uint16_t), 1, f);
    if (bitCount != 8) { printf("ERROR: Only 8‑bit grayscale BMP supported.\n"); fclose(f); return false; }
    fseek(f, bfOffBits, SEEK_SET);
    uint8_t row[IMG_DIM];
    for (int y = IMG_DIM-1; y >= 0; y--) {
        fread(row, 1, IMG_DIM, f);
        memcpy(img[y], row, IMG_DIM);
    }
    fclose(f);
    return true;
}


void print_feature_map(const ap_uint<32> *scratch, int dim, const char *layer) {
    printf("\n===== Feature Map after %s (%dx%dx2) =====\n", layer, dim, dim);
    for (int c = 0; c < 2; c++) {
        printf("  Channel %d:\n", c);
        for (int y = 0; y < dim; y++) {
            printf("    ");
            for (int x = 0; x < dim; x++) {
                ap_uint<32> w = scratch[y * dim + x];   // one word per pixel now
                ap_uint<8> val = (c == 0) ? w.range(7,0) : w.range(15,8);
                printf("%3d ", (int)val);
            }
            printf("\n");
        }
    }
    printf("==========================================\n\n");
}
/* ============================================================
 * Main
 * ============================================================ */
int main() {
    printf("========================================\n");
    printf("  MNIST CNN Engine - C Simulation\n");
    printf("========================================\n\n");

    const char *image_path   = "C:\\Users\\syg\\Desktop\\mnist\\mnist_sample_5.bmp";
    const char *weight_folder = "C:\\Users\\syg\\Desktop\\mnist";

    // 1. Load image
    uint8_t img[IMG_DIM][IMG_DIM];
    if (!load_bmp_grayscale(image_path, img)) {
        printf("WARNING: Could not load image. Using dummy gray image.\n");
        memset(img, 128, sizeof(img));
    } else {
        printf("[IMG] Image loaded: %s\n", image_path);
    }

    // 2. Build the reference weight_bram using the original function
    ap_uint<32> ref_weight_bram[WBRAM_DEPTH];
    if (!load_weights(weight_folder, ref_weight_bram)) {
        printf("FAILED: Weight loading failed.\n");
        return 1;
    }
    printf("[WEIGHTS] Reference weight_bram built.\n");

    // 3. Allocate simulated DDR and copy the reference BRAM into it
    const int DDR_SIZE = 65536;
    ap_uint<32> *ddr = (ap_uint<32>*)calloc(DDR_SIZE, sizeof(ap_uint<32>));
    if (!ddr) {
        printf("ERROR: Cannot allocate DDR\n");
        return 1;
    }
    ap_uint<32> weight_ddr_addr = 0x0000;
    for (int i = 0; i < WBRAM_DEPTH; i++) {
        ddr[weight_ddr_addr + i] = ref_weight_bram[i];
    }
    printf("[DDR] Copied weight_bram to DDR at 0x%08X\n", (unsigned)weight_ddr_addr);

    // 4. Stream image into the IP
    hls::stream<ap_axiu<8,0,0,0>> input_stream("input_stream");
    for (int y = 0; y < IMG_DIM; y++) {
        for (int x = 0; x < IMG_DIM; x++) {
            ap_axiu<8,0,0,0> tmp;
            tmp.data = img[y][x];
            tmp.keep = 1;
            tmp.last = (y == IMG_DIM-1 && x == IMG_DIM-1) ? 1 : 0;
            input_stream.write(tmp);
        }
    }
    printf("[STREAM] Image streamed (%d pixels).\n\n", IMG_DIM*IMG_DIM);

    // 5. IP signals (scratch is now 32‑bit)
    ap_uint<32> scratch_bram[SCRATCH_DEPTH] = {0};
    ap_uint<4>  prediction = 0;
    ap_uint<32> ctrl_reg   = 0;
    ap_uint<32> status_reg = 0;
    bool        intr       = 0;

    // 6. Stage 1: Conv1
    printf("=== STAGE 1: Conv1 ===\n");
    ctrl_reg = 0x01;  // start=1, layer_sel=0
    int cycles = 0;
    do {
        mnist_ip(input_stream, prediction, ctrl_reg, status_reg, intr,
                 weight_ddr_addr, ddr, scratch_bram);
        ctrl_reg = 0;  // clear start after first call
        cycles++;
        if (cycles > 100000) {
            printf("ERROR: Conv1 timeout!\n");
            free(ddr);
            return 1;
        }
    } while (!intr);
    printf("Conv1 done. Status=0x%08X (cycles=%d)\n", (unsigned)status_reg, cycles);
    print_feature_map(scratch_bram, POOL1_DIM, "Conv1");

    // 7. Stage 2: Conv2
    printf("=== STAGE 2: Conv2 ===\n");
    ctrl_reg = 0x02;  // clr_irq
    mnist_ip(input_stream, prediction, ctrl_reg, status_reg, intr,
             weight_ddr_addr, ddr, scratch_bram);
    printf("IRQ cleared. Status=0x%08X\n", (unsigned)status_reg);

    ctrl_reg = 0x05;  // start=1, layer_sel=1
    cycles = 0;
    do {
        mnist_ip(input_stream, prediction, ctrl_reg, status_reg, intr,
                 weight_ddr_addr, ddr, scratch_bram);
        ctrl_reg = 0;
        cycles++;
        if (cycles > 100000) {
            printf("ERROR: Conv2 timeout!\n");
            free(ddr);
            return 1;
        }
    } while (!intr);
    printf("Conv2 done. Status=0x%08X (cycles=%d)\n", (unsigned)status_reg, cycles);
    print_feature_map(scratch_bram, POOL2_DIM, "Conv2");

    // 8. Stage 3: Fully Connected
    printf("=== STAGE 3: Fully Connected ===\n");
    ctrl_reg = 0x02;  // clr_irq
    mnist_ip(input_stream, prediction, ctrl_reg, status_reg, intr,
             weight_ddr_addr, ddr, scratch_bram);
    printf("IRQ cleared. Status=0x%08X\n", (unsigned)status_reg);

    ctrl_reg = 0x09;  // start=1, layer_sel=2
    cycles = 0;
    do {
        mnist_ip(input_stream, prediction, ctrl_reg, status_reg, intr,
                 weight_ddr_addr, ddr, scratch_bram);
        ctrl_reg = 0;
        cycles++;
        if (cycles > 100000) {
            printf("ERROR: FC timeout!\n");
            free(ddr);
            return 1;
        }
    } while (!intr);
    printf("FC done. Status=0x%08X (cycles=%d)\n", (unsigned)status_reg, cycles);

    // 9. Final result
    printf("\n========================================\n");
    printf("  PREDICTION: %d\n", (int)prediction);
    printf("========================================\n");

    free(ddr);
    if (prediction == 5) {
        printf("*** CO-SIMULATION PASSED ***\n");
        return 0;
    } else {
        printf("*** CO-SIMULATION FAILED (expected 5, got %d) ***\n", (int)prediction);
        return 1;
    }
}