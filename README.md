PROJECT OVERVIEW
----------------
An end-to-end hardware-software co-design framework implementing a quantized
Convolutional Neural Network (CNN) accelerator IP core targeting the Xilinx 
Zynq-7000 Zybo FPGA platform[cite: 1]. The core is developed using AMD Vitis 
High-Level Synthesis (HLS) and integrated via AXI bus protocols[cite: 1].

- Target Board: Xilinx Zynq-7000 Zybo FPGA[cite: 1]
- Target Frequency: 100 MHz (10.0 ns period)[cite: 1]
- Hardware Architecture: Custom 2-layer, 2-channel CNN followed by an FNN 
  classifier[cite: 1]. Core state control is managed via an 8-state Finite State 
  Machine (FSM)[cite: 1].
- Quantization: Quantization-Aware Training (QAT) in TensorFlow using 8-bit 
  signed weights and 16-bit signed biases with post-training Batch 
  Normalization folding[cite: 1].
- Interfaces: AXI4-Stream for streaming image pixels[cite: 1], AXI4-Master for DDR 
  parameter loading[cite: 1], AXI4-Lite for core configuration[cite: 1], and a point-to-point 
  dual-port BRAM interface for intermediate feature storage[cite: 1].
- Host Software: Python-based UART script driving memory transfers, FSM 
  execution, and status polling[cite: 1].


FINITE STATE MACHINE (FSM) EXECUTION FLOW:
1. IDLE       : Initial reset state[cite: 1].
2. LOAD_W     : Fetches 8-bit weights and 16-bit biases from off-chip DDR 
                memory into internal static BRAM[cite: 1].
3. CONV1      : Streams the 784-pixel input image via AXI4-Stream and computes 
                Layer 1 feature maps[cite: 1].
4. WAIT_CONV2 : Execution pause awaiting host control register update[cite: 1].
5. CONV2      : Reads intermediate features from external scratchpad BRAM and 
                computes Layer 2[cite: 1].
6. WAIT_FC    : Execution pause awaiting host trigger for the dense layer[cite: 1].
7. FC         : Computes fully connected classification layer and runs argmax 
                classification[cite: 1].
8. DONE       : Asserts completion flags (ap_done) and returns output[cite: 1].


POST-IMPLEMENTATION PERFORMANCE
-------------------------------
- Achieved Clock Period: 9.278 ns[cite: 1]
- Worst Negative Slack (WNS): +0.722 ns (Met 100 MHz target constraint)[cite: 1]
- Critical Path: Deeply nested 'if' statements in the final layer's argmax module[cite: 1].

Resource Utilization Metrics:
-----------------------------------------------------------------
Resource         | Guideline | Actual Utilization | Status
-----------------------------------------------------------------
LUT              | 70%       | 59.68%             | OK[cite: 1]
FD               | 50%       | 20.57%             | OK[cite: 1]
LUTRAM+SRL       | 25%       | 7.58%              | OK[cite: 1]
MUXF7            | 15%       | 13.76%             | OK[cite: 1]
LUT Combining    | 20%       | 22.69%             | REVIEW[cite: 1]
DSP              | 80%       | 90.00%             | REVIEW[cite: 1]
BRAM/FIFO        | 80%       | 10.83%             | OK[cite: 1]
-----------------------------------------------------------------


MEMORY MAP REGISTERS (s_axi_ctrl)
---------------------------------
- 0x00 (CTRL)           : R/W | Bit 0: ap_start, Bit 1: ap_done, 
                                Bit 2: ap_idle, Bit 3: ap_ready[cite: 1]
- 0x04 (GIER)           : R/W | Bit 0: Global Interrupt Enable[cite: 1]
- 0x10 (prediction)     : R   | Bits [3:0]: Output digit prediction (0-9)[cite: 1]
- 0x14 (prediction_ctrl): R   | Bit 0: Valid flag (prediction_ap_vld)[cite: 1]
- 0x20 (ctrl_reg)       : W   | Host FSM state trigger register[cite: 1]
- 0x28 (status_reg)     : R   | Hardware status feedback register[cite: 1]
- 0x48 (weight_addr)    : W   | DDR physical base address offset for 
                                weights/biases[cite: 1]


REPOSITORY STRUCTURE
--------------------
docs/   
driver/                 # driver 
hls/                    # Vitis HLS C/C++ source code and top headers
hardware/               #  XSA
weights/                # Quantized weights and biases files
host/                   # Python UART scripts and test BMP images
README.txt              # Project text documentation

## How to Run

### Step 1: Set Up Vitis Application
1. Open **AMD Vitis IDE**.
2. Go to **File → New → Platform Project**, select your `block_design.xsa` file[cite: 1], and click **Build Project**.
3. Go to **File → New → Application Project**, select the platform you just built, choose **ps7_cortexa9_0**[cite: 1], and select **Empty Application (C/C++)**.
4. Drag your custom driver files and main C code into the application's `src/` folder[cite: 1].
5. Right-click your application project and select **Build Project**.

### Step 2: Program & Run
1. Power on your Zynq Zybo board (connected via USB).
2. Right-click your application project in Vitis and select **Run As → Launch Hardware (Single Application Debug)**.

### Step 3: Stream Image via Python Host
Run the Python script from your PC terminal to stream your image and view the digit prediction:
