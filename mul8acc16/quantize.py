# Code written by Jaduk Suh
# February 1st 2024, KAIST

# Note: bounding box size is set to 16
# Due to triton limitations, bounding box size must be power of 2

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def fp8e4m3_matmul_kernel(x_ptr, y_ptr, out_ptr,
                          x_stride_r, x_stride_c, 
                          y_stride_r, y_stride_c,
                          o_stride_r, o_stride_c,
                          BLOCK_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_N: tl.constexpr,
                          BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    x_row = pid_m * BLOCK_SIZE_M
    y_row = pid_k * BLOCK_SIZE_K
    x_offset = x_row * x_stride_r
    y_offset = y_row * y_stride_r


    for i in tl.static_range(BLOCK_SIZE_M):
        for j in tl.static_range(BLOCK_SIZE_N):
            block_offset_x = pid_k * BLOCK_SIZE_K * x_stride_c
            block_offset_y = pid_n * BLOCK_SIZE_N * y_stride_c
            x_local_offset = x_offset + i * x_stride_r + block_offset_x
            y_local_offset = y_offset + block_offset_y + j * y_stride_c

            x_data_indices = x_local_offset + tl.arange(0, BLOCK_SIZE_K) * x_stride_c
            y_data_indices = y_local_offset + (tl.arange(0, BLOCK_SIZE_K) * y_stride_r)

            x_raw_data = tl.load(x_ptr + x_data_indices)
            y_raw_data = tl.load(y_ptr + y_data_indices)

            x_sign = (x_raw_data >> 7) & 0x1
            x_expo = ((x_raw_data >> 3) & 0x0F).to(tl.int16)
            x_mant = (x_raw_data & 0x07).to(tl.int16)
            y_sign = (y_raw_data >> 7) & 0x1
            y_expo = ((y_raw_data >> 3) & 0x0F).to(tl.int16)
            y_mant = (y_raw_data & 0x07).to(tl.int16)

            x_leading_bit = tl.where(x_expo == 0, 0, 1)
            y_leading_bit = tl.where(y_expo == 0, 0, 1)
            x_mant |= x_leading_bit << 3
            y_mant |= y_leading_bit << 3

            product_sign = ((x_sign ^ y_sign) * -2) + 1
            product = (x_mant * y_mant * product_sign).to(tl.float16)
            scalar = x_expo + y_expo + (1 - x_leading_bit) + (1 - y_leading_bit) - 20
            scalar_half = tl.math.pow(2.0, scalar).to(tl.float16)
            partial_sum = tl.sum(product * scalar_half)
            target_row = x_row + i
            target_col = pid_n * BLOCK_SIZE_N + j
            output_offset = target_row * o_stride_r + target_col * o_stride_c
            tl.atomic_add(out_ptr + output_offset, partial_sum)

@triton.jit
def fp8e5m2_matmul_kernel(x_ptr, y_ptr, out_ptr,
                          x_stride_r, x_stride_c, 
                          y_stride_r, y_stride_c,
                          o_stride_r, o_stride_c,
                          BLOCK_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_N: tl.constexpr,
                          BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    x_row = pid_m * BLOCK_SIZE_M
    y_row = pid_k * BLOCK_SIZE_K
    x_offset = x_row * x_stride_r
    y_offset = y_row * y_stride_r


    for i in tl.static_range(BLOCK_SIZE_M):
        for j in tl.static_range(BLOCK_SIZE_N):
            block_offset_x = pid_k * BLOCK_SIZE_K * x_stride_c
            block_offset_y = pid_n * BLOCK_SIZE_N * y_stride_c
            x_local_offset = x_offset + i * x_stride_r + block_offset_x
            y_local_offset = y_offset + block_offset_y + j * y_stride_c

            x_data_indices = x_local_offset + tl.arange(0, BLOCK_SIZE_K) * x_stride_c
            y_data_indices = y_local_offset + (tl.arange(0, BLOCK_SIZE_K) * y_stride_r)

            x_raw_data = tl.load(x_ptr + x_data_indices)
            y_raw_data = tl.load(y_ptr + y_data_indices)

            x_sign = (x_raw_data >> 7) & 0x1
            x_expo = ((x_raw_data >> 2) & 0x1F).to(tl.int16)
            x_mant = (x_raw_data & 0x03).to(tl.int16)
            y_sign = (y_raw_data >> 7) & 0x1
            y_expo = ((y_raw_data >> 2) & 0x1F).to(tl.int16)
            y_mant = (y_raw_data & 0x03).to(tl.int16)

            x_leading_bit = tl.where(x_expo == 0, 0, 1)
            y_leading_bit = tl.where(y_expo == 0, 0, 1)
            x_mant |= x_leading_bit << 2
            y_mant |= y_leading_bit << 2

            product_sign = ((x_sign ^ y_sign) * -2) + 1
            product = (x_mant * y_mant * product_sign).to(tl.float16)
            scalar = x_expo + y_expo + (1 - x_leading_bit) + (1 - y_leading_bit) - 34
            scalar_half = tl.math.pow(2.0, scalar).to(tl.float16)
            partial_sum = tl.sum(product * scalar_half)
            target_row = x_row + i
            target_col = pid_n * BLOCK_SIZE_N + j
            output_offset = target_row * o_stride_r + target_col * o_stride_c
            tl.atomic_add(out_ptr + output_offset, partial_sum)

# Perform matrix multiplication xy^T
# x and y are both fp8e4m3 precision
# x and y are converted into fp8e4m3 precision before matrix multiplication
# result is in fp16 precison
def fp8e4m3_matmul(x: torch.Tensor, y: torch.Tensor):
    assert x.device == y.device
    torch.cuda.set_device(x.device.index)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 16, 16, 16
    num_warps = 4

    x_buffer = x.untyped_storage()
    y_buffer = y.untyped_storage()
    
    new_x = torch.tensor([], dtype=torch.uint8, device=x.device).set_(
        source=x_buffer, 
        storage_offset=0, 
        size=(x.shape[0], x.shape[1]), 
        stride=(x.stride(0), x.stride(1))).view(x.shape[0], x.shape[1])
    new_y = torch.tensor([], dtype=torch.uint8, device=y.device).set_(
        source=y_buffer, 
        storage_offset=0, 
        size=(y.shape[0], y.shape[1]), 
        stride=(y.stride(0), y.stride(1))).view(y.shape[0], y.shape[1])

    M, N, K = new_x.shape[0], new_y.shape[1], new_x.shape[1]
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = triton.cdiv(K, BLOCK_SIZE_K)
    output = torch.zeros([M, N], dtype=torch.float16).to(new_x.device)
    fp8e4m3_matmul_kernel[(num_pid_m, num_pid_n, num_pid_k)](new_x, new_y, output,
                                                  new_x.stride(0), new_x.stride(1),
                                                  new_y.stride(0), new_y.stride(1),
                                                  output.stride(0), output.stride(1),
                                                  BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                                  num_warps=num_warps)
    return output.to(torch.float8_e4m3)

# Perform matrix multiplication xy^T
# x and y are both fp32 precision
# x and y are converted into fp8e5m2 precision before matrix multiplication
# result is in fp16 precison
def fp8e5m2_matmul(x: torch.Tensor, y: torch.Tensor):
    assert x.device == y.device
    torch.cuda.set_device(x.device.index)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 16, 16, 16
    num_warps = 4

    x_buffer = x.untyped_storage()
    y_buffer = y.untyped_storage()
    
    new_x = torch.tensor([], dtype=torch.uint8, device=x.device).set_(
        source=x_buffer, 
        storage_offset=0, 
        size=(x.shape[0], x.shape[1]), 
        stride=(x.stride(0), x.stride(1))).view(x.shape[0], x.shape[1])
    new_y = torch.tensor([], dtype=torch.uint8, device=y.device).set_(
        source=y_buffer, 
        storage_offset=0, 
        size=(y.shape[0], y.shape[1]), 
        stride=(y.stride(0), y.stride(1))).view(y.shape[0], y.shape[1])

    M, N, K = new_x.shape[0], new_y.shape[1], new_x.shape[1]
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = triton.cdiv(K, BLOCK_SIZE_K)
    output = torch.zeros([M, N], dtype=torch.float16).to(new_x.device)
    fp8e5m2_matmul_kernel[(num_pid_m, num_pid_n, num_pid_k)](new_x, new_y, output,
                                                  new_x.stride(0), new_x.stride(1),
                                                  new_y.stride(0), new_y.stride(1),
                                                  output.stride(0), output.stride(1),
                                                  BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                                  num_warps=num_warps)
    return output.to(torch.float8_e5m2)