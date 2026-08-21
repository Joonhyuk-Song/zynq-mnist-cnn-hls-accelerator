/* ===================================================================
 * mnist_ip_baremetal.c
 * Combined single-file bare-metal driver for the mnist_ip HLS core
 * on Zynq-7000. Register map ported from Table A.2 of the HLS report;
 * weight-region offsets copied from your mnist_ip.h (WBRAM_DEPTH=266).
 * =================================================================== */

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "xil_io.h"
#include "xil_cache.h"
#include "xparameters.h"
#include "xuartps.h"
#include "xaxidma.h"
#include "xstatus.h"

/* ===================================================================
 * TODO -- fill these in from YOUR Vivado Address Editor / xparameters.h
 * (Address Editor -> right-click each AXI-Lite interface -> "Copy Address")
 * =================================================================== */
#ifndef MNIST_CTRL_BASEADDR
#define MNIST_CTRL_BASEADDR     0x40010000U   /* s_axi_ctrl slave (control/status/prediction) -- confirmed via Address Editor */
#endif
#ifndef MNIST_CONTROL_BASEADDR
#define MNIST_CONTROL_BASEADDR  0x40000000U   /* s_axi_control slave (m_axi_gmem pointer regs) -- confirmed via Address Editor */
#endif
#ifndef DMA_DEV_ID
#define DMA_DEV_ID               XPAR_AXIDMA_0_DEVICE_ID  /* your AXI DMA feeding input_stream */
#endif
#ifndef UART_DEV_ID
#define UART_DEV_ID               XPAR_XUARTPS_1_DEVICE_ID
#endif

/* Reserved physical DDR regions -- CARVE THESE OUT OF YOUR LINKER SCRIPT
 * (lscript.ld) so .bss/heap/stack never overlap them. */
#ifndef WEIGHT_DDR_BASEADDR
#define WEIGHT_DDR_BASEADDR      0x0F000000U
#endif
#ifndef IMAGE_DDR_BASEADDR
#define IMAGE_DDR_BASEADDR       0x0F100000U
#endif

/* ===================================================================
 * Real values from your mnist_ip.h
 * =================================================================== */
#define WBRAM_DEPTH     266   /* 32-bit words */
#define W_OFF_CONV1       0
#define B_OFF_CONV1       5
#define W_OFF_CONV2       6
#define B_OFF_CONV2      15
#define W_OFF_FC         16
#define B_OFF_FC        261

#define IMG_DIM         28
#define IMG_PIXELS      (IMG_DIM * IMG_DIM)

#define CONV1_W_LEN     18
#define CONV1_B_LEN      2
#define CONV2_W_LEN     36
#define CONV2_B_LEN      2
#define FC_W_LEN       980
#define FC_B_LEN        10

/* ===================================================================
 * s_axi_ctrl register offsets -- Table A.2 of the report
 * =================================================================== */
#define REG_CTRL              0x00  /* bit0 ap_start, bit1 ap_done, bit2 ap_idle,
                                        bit3 ap_ready, bit7 auto_restart, bit9 interrupt */
#define REG_GIER               0x04
#define REG_IP_IER               0x08
#define REG_IP_ISR                 0x0C
#define REG_PREDICTION               0x10  /* R, low 4 bits = predicted digit */
#define REG_PREDICTION_CTRL             0x14  /* R, bit0 = data valid */
#define REG_CTRL_REG                       0x20  /* W, FSM control word */
#define REG_STATUS_REG                        0x28  /* R */
#define REG_STATUS_REG_CTRL                      0x2C  /* R, bit0 = data valid */
#define REG_INTR                                    0x38  /* R, dedicated intr boolean */
#define REG_INTR_CTRL                                  0x3C  /* R, bit0 = data valid */
#define REG_WEIGHT_ADDR                                  0x48  /* W, DDR word-offset for weights */

#define CTRL_BIT_AP_START     (1U << 0)
#define CTRL_BIT_AP_DONE      (1U << 1)
#define CTRL_BIT_AP_IDLE      (1U << 2)
#define CTRL_BIT_AP_READY     (1U << 3)
#define CTRL_BIT_AUTO_RESTART (1U << 7)

/* ---- s_axi_control offsets: m_axi_gmem base pointer ---- */
#define REG_DDR_PTR_LO   0x10  /* W, ddr_1 -- DDR base phys addr bits[31:0] */
#define REG_DDR_PTR_HI   0x14  /* W, ddr_2 -- bits[63:32]; 0 on Zynq-7000 (32-bit AXI) */

/* ---- ctrl_reg (0x20) encodings, ported 1:1 from your HLS testbench ---- */
#define CTRLREG_START_CONV1   0x01U  /* start=1, layer_sel=0 -> LOAD_W then CONV1 */
#define CTRLREG_CLR_IRQ       0x02U
#define CTRLREG_START_CONV2   0x05U  /* start=1, layer_sel=1 */
#define CTRLREG_START_FC      0x09U  /* start=1, layer_sel=2 */

typedef struct {
    XUartPs   uart;
    XAxiDma   dma;
} MnistBoard;

/* Low-level register access */
static inline void mnist_ctrl_write(uint32_t offset, uint32_t val) {
    Xil_Out32(MNIST_CTRL_BASEADDR + offset, val);
}
static inline uint32_t mnist_ctrl_read(uint32_t offset) {
    return Xil_In32(MNIST_CTRL_BASEADDR + offset);
}
static inline void mnist_control_write(uint32_t offset, uint32_t val) {
    Xil_Out32(MNIST_CONTROL_BASEADDR + offset, val);
}

/* ===================================================================
 * UART hex-value receiver
 * Ported from your read_int8_mem()/read_int16_mem() character-class
 * logic, pulled byte-by-byte off UART. Each call parses `count`
 * whitespace/comma/newline separated hex tokens (e.g. "f86f\r\n"),
 * sign-extending each as a 16-bit two's-complement value before the
 * caller narrows it to int8_t or int16_t.
 * =================================================================== */
static inline bool is_hex_digit_c(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
static inline int hex_val_c(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

static int uart_getc_blocking(XUartPs *uart) {
    uint8_t ch;
    while (XUartPs_Recv(uart, &ch, 1) == 0) {
        /* spin -- add a watchdog/timeout here if you want a hard bail-out */
    }
    return (int)ch;
}

static int uart_read_hex_values(XUartPs *uart, int32_t *out, int count) {
    int idx = 0;
    while (idx < count) {
        int c = uart_getc_blocking(uart);

        if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == ',' ||
            c == ';' || c == '[' || c == ']' || c == '{' || c == '}') {
            continue;
        }
        if (!is_hex_digit_c((char)c)) {
            continue; /* stray character -- ignore and keep scanning */
        }

        uint16_t raw = 0;
        int nibbles = 0;
        int ch = c;
        while (is_hex_digit_c((char)ch) && nibbles < 4) {
            raw = (uint16_t)((raw << 4) | hex_val_c((char)ch));
            nibbles++;
            ch = uart_getc_blocking(uart);
            if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' || ch == ',') break;
        }
        out[idx++] = (int16_t)raw; /* sign-extend, matches "f86f" -> negative */
    }
    return idx;
}

/* ===================================================================
 * Board init
 * =================================================================== */
int mnist_board_init(MnistBoard *b) {
    int status;

    XUartPs_Config *uart_cfg = XUartPs_LookupConfig(UART_DEV_ID);
    if (!uart_cfg) return XST_FAILURE;
    status = XUartPs_CfgInitialize(&b->uart, uart_cfg, uart_cfg->BaseAddress);
    if (status != XST_SUCCESS) return status;
    XUartPs_SetBaudRate(&b->uart, 115200);

    XAxiDma_Config *dma_cfg = XAxiDma_LookupConfig(DMA_DEV_ID);
    if (!dma_cfg) return XST_FAILURE;
    status = XAxiDma_CfgInitialize(&b->dma, dma_cfg);
    if (status != XST_SUCCESS) return status;

    /* Poll mode: leave the AXI-Lite interrupt path disabled and rely on
     * REG_INTR / REG_STATUS_REG like the original testbench. */
    mnist_ctrl_write(REG_GIER, 0);
    mnist_ctrl_write(REG_IP_IER, 0);

    /* Program the m_axi_gmem base pointer once -- weight_addr (0x48) is
     * then just a word offset within this region. */
    mnist_control_write(REG_DDR_PTR_LO, (uint32_t)(WEIGHT_DDR_BASEADDR & 0xFFFFFFFFU));
    mnist_control_write(REG_DDR_PTR_HI, 0);
    mnist_ctrl_write(REG_WEIGHT_ADDR, 0);

    return XST_SUCCESS;
}

/* ===================================================================
 * Weight loading over UART, packed exactly like your original
 * load_weights() -- same bit layout, written straight into physical
 * DDR via a pointer instead of a local ap_uint<32> array.
 * =================================================================== */
int mnist_load_weights_uart(MnistBoard *b) {
    int32_t tmp[FC_W_LEN]; /* largest of the six blocks; reused for each */

    int8_t  conv1_w[CONV1_W_LEN]; int16_t conv1_b[CONV1_B_LEN];
    int8_t  conv2_w[CONV2_W_LEN]; int16_t conv2_b[CONV2_B_LEN];
    int8_t  fc_w[FC_W_LEN];       int16_t fc_b[FC_B_LEN];

    xil_printf("Send conv1_weights (%d hex values)...\r\n", CONV1_W_LEN);
    if (uart_read_hex_values(&b->uart, tmp, CONV1_W_LEN) != CONV1_W_LEN) return XST_FAILURE;
    for (int i = 0; i < CONV1_W_LEN; i++) conv1_w[i] = (int8_t)tmp[i];

    xil_printf("Send conv1_bias (%d hex values)...\r\n", CONV1_B_LEN);
    if (uart_read_hex_values(&b->uart, tmp, CONV1_B_LEN) != CONV1_B_LEN) return XST_FAILURE;
    for (int i = 0; i < CONV1_B_LEN; i++) conv1_b[i] = (int16_t)tmp[i];

    xil_printf("Send conv2_weights (%d hex values)...\r\n", CONV2_W_LEN);
    if (uart_read_hex_values(&b->uart, tmp, CONV2_W_LEN) != CONV2_W_LEN) return XST_FAILURE;
    for (int i = 0; i < CONV2_W_LEN; i++) conv2_w[i] = (int8_t)tmp[i];

    xil_printf("Send conv2_bias (%d hex values)...\r\n", CONV2_B_LEN);
    if (uart_read_hex_values(&b->uart, tmp, CONV2_B_LEN) != CONV2_B_LEN) return XST_FAILURE;
    for (int i = 0; i < CONV2_B_LEN; i++) conv2_b[i] = (int16_t)tmp[i];

    xil_printf("Send fc_weights (%d hex values)...\r\n", FC_W_LEN);
    if (uart_read_hex_values(&b->uart, tmp, FC_W_LEN) != FC_W_LEN) return XST_FAILURE;
    for (int i = 0; i < FC_W_LEN; i++) fc_w[i] = (int8_t)tmp[i];

    xil_printf("Send fc_bias (%d hex values)...\r\n", FC_B_LEN);
    if (uart_read_hex_values(&b->uart, tmp, FC_B_LEN) != FC_B_LEN) return XST_FAILURE;
    for (int i = 0; i < FC_B_LEN; i++) fc_b[i] = (int16_t)tmp[i];

    /* Pack into the same word layout load_weights() built, writing
     * straight into physical DDR at WEIGHT_DDR_BASEADDR. */
    volatile uint32_t *ddr = (volatile uint32_t *)WEIGHT_DDR_BASEADDR;
    memset((void *)ddr, 0, WBRAM_DEPTH * sizeof(uint32_t));

    for (int i = 0; i < CONV1_W_LEN; i++) {
        int word = W_OFF_CONV1 + i / 4, byte = i % 4;
        ddr[word] |= ((uint32_t)(uint8_t)conv1_w[i]) << (byte * 8);
    }
    for (int i = 0; i < CONV1_B_LEN; i++)
        ddr[B_OFF_CONV1] |= ((uint32_t)(uint16_t)conv1_b[i]) << (i * 16);

    for (int i = 0; i < CONV2_W_LEN; i++) {
        int word = W_OFF_CONV2 + i / 4, byte = i % 4;
        ddr[word] |= ((uint32_t)(uint8_t)conv2_w[i]) << (byte * 8);
    }
    for (int i = 0; i < CONV2_B_LEN; i++)
        ddr[B_OFF_CONV2] |= ((uint32_t)(uint16_t)conv2_b[i]) << (i * 16);

    for (int i = 0; i < FC_W_LEN; i++) {
        int word = W_OFF_FC + i / 4, byte = i % 4;
        ddr[word] |= ((uint32_t)(uint8_t)fc_w[i]) << (byte * 8);
    }
    for (int i = 0; i < FC_B_LEN; i++) {
        int word = B_OFF_FC + i / 2, half = i % 2;
        ddr[word] |= ((uint32_t)(uint16_t)fc_b[i]) << (half * 16);
    }

    /* m_axi_gmem is a hardware master reading physical DRAM directly --
     * push the CPU's cached writes out to DRAM before the IP fetches it. */
    Xil_DCacheFlushRange((INTPTR)WEIGHT_DDR_BASEADDR, WBRAM_DEPTH * sizeof(uint32_t));

    xil_printf("Weights loaded and flushed to DDR @ 0x%08X\r\n", (unsigned)WEIGHT_DDR_BASEADDR);
    return XST_SUCCESS;
}

/* ===================================================================
 * Image loading over UART: 784 grayscale bytes, sent as hex tokens
 * same as the weight blocks, into a DDR buffer the DMA reads from.
 * =================================================================== */
int mnist_load_image_uart(MnistBoard *b, uint8_t img[IMG_DIM][IMG_DIM]) {
    int32_t tmp[IMG_PIXELS];
    xil_printf("Send image (%d hex byte values, row-major)...\r\n", IMG_PIXELS);
    if (uart_read_hex_values(&b->uart, tmp, IMG_PIXELS) != IMG_PIXELS) return XST_FAILURE;

    volatile uint8_t *ddr_img = (volatile uint8_t *)IMAGE_DDR_BASEADDR;
    for (int i = 0; i < IMG_PIXELS; i++) {
        uint8_t px = (uint8_t)tmp[i];
        ddr_img[i] = px;
        img[i / IMG_DIM][i % IMG_DIM] = px; /* local copy, e.g. for debug printing */
    }
    Xil_DCacheFlushRange((INTPTR)IMAGE_DDR_BASEADDR, IMG_PIXELS);
    return XST_SUCCESS;
}

/* Kicks a simple-mode AXI DMA MM2S transfer: DDR image buffer -> input_stream. */
int mnist_stream_image(MnistBoard *b) {
    int status = XAxiDma_SimpleTransfer(&b->dma,
                                         (UINTPTR)IMAGE_DDR_BASEADDR,
                                         IMG_PIXELS,
                                         XAXIDMA_DMA_TO_DEVICE);
    if (status != XST_SUCCESS) return status;

    while (XAxiDma_Busy(&b->dma, XAXIDMA_DMA_TO_DEVICE)) {
        /* spin until the DMA finishes pushing all 784 bytes into input_stream */
    }
    return XST_SUCCESS;
}

/* ===================================================================
 * Layer execution: writes ctrl_reg, launches the FSM with auto_restart,
 * and polls REG_INTR the way the testbench polled its local `intr` bool.
 *
 * NOTE: this assumes auto_restart keeps the block re-invoking and
 * re-sampling ctrl_reg each cycle without you re-pulsing ap_start.
 * Verify this against a Vivado-exported xmnist_ip_hw.h if you have one --
 * if the IP instead needs ap_start pulsed on every internal step, add a
 * per-iteration Xil_Out32(REG_CTRL, ...) inside the poll loop.
 * =================================================================== */
int mnist_run_layer(uint32_t ctrl_reg_value, uint32_t timeout_polls) {
    mnist_ctrl_write(REG_CTRL_REG, ctrl_reg_value);
    mnist_ctrl_write(REG_CTRL, CTRL_BIT_AP_START | CTRL_BIT_AUTO_RESTART);

    uint32_t polls = 0;
    while ((mnist_ctrl_read(REG_INTR) & 0x1U) == 0) {
        if (++polls > timeout_polls) {
            xil_printf("ERROR: layer timeout (ctrl_reg=0x%02X)\r\n", (unsigned)ctrl_reg_value);
            return XST_FAILURE;
        }
    }
    return XST_SUCCESS;
}

int mnist_clear_irq(void) {
    mnist_ctrl_write(REG_CTRL_REG, CTRLREG_CLR_IRQ);
    mnist_ctrl_write(REG_CTRL, CTRL_BIT_AP_START | CTRL_BIT_AUTO_RESTART);
    uint32_t polls = 0;
    while ((mnist_ctrl_read(REG_INTR) & 0x1U) != 0) {
        if (++polls > 100000) return XST_FAILURE;
    }
    return XST_SUCCESS;
}

/* ===================================================================
 * Top-level inference, mirrors your original main()'s three stages.
 * =================================================================== */
int mnist_run_inference(MnistBoard *b, uint8_t *prediction_out) {
    int status;

    xil_printf("=== STAGE 1: Conv1 ===\r\n");
    if (mnist_stream_image(b) != XST_SUCCESS) return XST_FAILURE;
    status = mnist_run_layer(CTRLREG_START_CONV1, 100000);
    if (status != XST_SUCCESS) return status;
    xil_printf("Conv1 done. status_reg=0x%08X\r\n", (unsigned)mnist_ctrl_read(REG_STATUS_REG));

    if (mnist_clear_irq() != XST_SUCCESS) return XST_FAILURE;

    xil_printf("=== STAGE 2: Conv2 ===\r\n");
    status = mnist_run_layer(CTRLREG_START_CONV2, 100000);
    if (status != XST_SUCCESS) return status;
    xil_printf("Conv2 done. status_reg=0x%08X\r\n", (unsigned)mnist_ctrl_read(REG_STATUS_REG));

    if (mnist_clear_irq() != XST_SUCCESS) return XST_FAILURE;

    xil_printf("=== STAGE 3: Fully Connected ===\r\n");
    status = mnist_run_layer(CTRLREG_START_FC, 100000);
    if (status != XST_SUCCESS) return status;
    xil_printf("FC done. status_reg=0x%08X\r\n", (unsigned)mnist_ctrl_read(REG_STATUS_REG));

    if ((mnist_ctrl_read(REG_PREDICTION_CTRL) & 0x1U) == 0) {
        xil_printf("WARNING: prediction_ap_vld not set\r\n");
    }
    *prediction_out = (uint8_t)(mnist_ctrl_read(REG_PREDICTION) & 0xF);
    return XST_SUCCESS;
}

/* ===================================================================
 * main() -- replaces the co-sim testbench's main()
 * =================================================================== */
int main(void) {
    MnistBoard board;
    uint8_t img[IMG_DIM][IMG_DIM];
    uint8_t prediction;

    Xil_DCacheEnable();

    xil_printf("========================================\r\n");
    xil_printf("  MNIST CNN Engine - Zynq-7000 Bare Metal\r\n");
    xil_printf("========================================\r\n\r\n");

    if (mnist_board_init(&board) != XST_SUCCESS) {
        xil_printf("FAILED: board init\r\n");
        return 1;
    }

    if (mnist_load_weights_uart(&board) != XST_SUCCESS) {
        xil_printf("FAILED: weight load\r\n");
        return 1;
    }

    if (mnist_load_image_uart(&board, img) != XST_SUCCESS) {
        xil_printf("FAILED: image load\r\n");
        return 1;
    }

    if (mnist_run_inference(&board, &prediction) != XST_SUCCESS) {
        xil_printf("FAILED: inference\r\n");
        return 1;
    }

    xil_printf("\r\n========================================\r\n");
    xil_printf("  PREDICTION: %d\r\n", prediction);
    xil_printf("========================================\r\n");

    Xil_DCacheDisable();
    return 0;
}
