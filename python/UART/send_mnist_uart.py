#!/usr/bin/env python3
"""
send_mnist_uart.py

Feeds weights (.mem files) and a sample image (.bmp) to the Zynq board
over UART, in the exact order/format mnist_ip_baremetal.c's
mnist_load_weights_uart() / mnist_load_image_uart() are blocking on.

Order matters and must match main()'s call sequence:
    conv1_weights (18 tokens) -> conv1_bias (2) -> conv2_weights (36) ->
    conv2_bias (2) -> fc_weights (980) -> fc_bias (10) -> image (784)

Usage:
    python3 send_mnist_uart.py --port COM5 --dir /path/to/mem_files
    python3 send_mnist_uart.py --port /dev/ttyUSB1 --dir . --baud 115200
"""

import argparse
import struct
import sys
import time

import serial


def read_mem_tokens(path):
    """.mem files are already newline-separated hex tokens -- just
    read and forward them as-is (matches uart_read_hex_values())."""
    with open(path, "r") as f:
        tokens = [line.strip() for line in f if line.strip()]
    return tokens


def read_bmp_grayscale_28x28(path):
    """Mirrors load_bmp_grayscale(): 8-bit grayscale, 28x28, BMP rows
    stored bottom-up in the file -> reversed here so index 0 is the
    top row, matching img[y][x] row-major order on the board."""
    with open(path, "rb") as f:
        data = f.read()

    if data[0:2] != b"BM":
        sys.exit("ERROR: not a BMP file")

    bf_off_bits = struct.unpack("<I", data[10:14])[0]
    width = struct.unpack("<i", data[18:22])[0]
    height = struct.unpack("<i", data[22:26])[0]
    bit_count = struct.unpack("<H", data[28:30])[0]

    if width != 28 or height != 28:
        sys.exit(f"ERROR: BMP must be 28x28, got {width}x{height}")
    if bit_count != 8:
        sys.exit(f"ERROR: BMP must be 8-bit grayscale, got {bit_count}-bit")

    row_bytes = width  # 28 is already a multiple of 4, no row padding
    rows = []
    for r in range(height):
        start = bf_off_bits + r * row_bytes
        rows.append(data[start:start + row_bytes])

    # File stores bottom-up; reverse so rows[0] becomes the top row
    rows.reverse()

    pixels = []
    for row in rows:
        pixels.extend(row)  # each pixel is 0-255
    return pixels  # 784 bytes, row-major, top row first


def send_tokens(ser, tokens, label, delay_s=0.0):
    print(f"[SEND] {label}: {len(tokens)} tokens")
    for tok in tokens:
        ser.write((tok + "\n").encode("ascii"))
        if delay_s:
            time.sleep(delay_s)
    ser.flush()


def drain_and_print(ser, seconds=1.0):
    """Print whatever the board's xil_printf() debug messages send back,
    useful to confirm it's asking for the next block."""
    end = time.time() + seconds
    buf = b""
    while time.time() < end:
        n = ser.in_waiting
        if n:
            buf += ser.read(n)
        else:
            time.sleep(0.02)
    if buf:
        sys.stdout.write(buf.decode("ascii", errors="replace"))
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM5", help="e.g. COM5 or /dev/ttyUSB1")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dir", default=r"C:\Users\joons\Desktop\mnist", help="folder containing the .mem/.bmp files")
    ap.add_argument("--delay", type=float, default=0.01,
                     help="seconds between tokens if the board's UART FIFO needs pacing")
    args = ap.parse_args()

    d = args.dir.rstrip("/")

    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    time.sleep(0.2)  # let the port settle

    print("Listening for board prompt before sending conv1_weights...")
    drain_and_print(ser, 1.0)

    send_tokens(ser, read_mem_tokens(f"{d}/conv1_weights.mem"), "conv1_weights", args.delay)
    drain_and_print(ser, 0.3)
    send_tokens(ser, read_mem_tokens(f"{d}/conv1_bias.mem"), "conv1_bias", args.delay)
    drain_and_print(ser, 0.3)
    send_tokens(ser, read_mem_tokens(f"{d}/conv2_weights.mem"), "conv2_weights", args.delay)
    drain_and_print(ser, 0.3)
    send_tokens(ser, read_mem_tokens(f"{d}/conv2_bias.mem"), "conv2_bias", args.delay)
    drain_and_print(ser, 0.3)
    send_tokens(ser, read_mem_tokens(f"{d}/fc_weights.mem"), "fc_weights", args.delay)
    drain_and_print(ser, 0.3)
    send_tokens(ser, read_mem_tokens(f"{d}/fc_bias.mem"), "fc_bias", args.delay)
    drain_and_print(ser, 0.5)

    pixels = read_bmp_grayscale_28x28(f"{d}/mnist_sample_5.bmp")
    hex_pixels = [f"{p:02x}" for p in pixels]
    send_tokens(ser, hex_pixels, "image (784 px)", args.delay)

    print("All data sent. Reading board output for 5s...")
    drain_and_print(ser, 5.0)

    ser.close()


if __name__ == "__main__":
    main()
