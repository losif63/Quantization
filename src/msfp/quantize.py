# Code written by Jaduk Suh
# February 1st 2024, KAIST

# Note: bounding box size is set to 16
# Due to triton limitations, bounding box size must be power of 2

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def fp32_to_msfp16_kernel(input_ptr, output_ptr, 
                          input_stride_r, input_stride_c,
                          output_stride_r, output_stride_c,
                          BLOCK_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_N: tl.constexpr
                          ):

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    output_BLOCK_N = BLOCK_SIZE_N + 1

    row_start = pid_m * BLOCK_SIZE_M
    row_start_id = pid_m * BLOCK_SIZE_M * input_stride_r
    output_row_start_id = pid_m * BLOCK_SIZE_M * output_stride_r
    block_start_id = row_start_id + pid_n * BLOCK_SIZE_N * input_stride_c
    output_block_start_id = output_row_start_id + pid_n * output_BLOCK_N * input_stride_c
    for i in tl.static_range(BLOCK_SIZE_M):
        local_start = block_start_id + i * input_stride_r
        offsets = local_start + tl.arange(0, BLOCK_SIZE_N) * input_stride_c
        mask = offsets < (row_start_id + input_stride_r + i * input_stride_r)
        x = tl.load(input_ptr + offsets, mask=mask)
        E = ((x.to(tl.uint32, bitcast=True) << 1) >> 24).to(tl.int32, bitcast=True) - 127
        scaler = tl.where(E > 0, -32, 32)
        scaler_float = tl.where(E > 0, 2.0 ** -32, 2.0 ** 32)
        x *= scaler_float
        
        # Now, x is all normalized values except zero
        x_I = x.to(tl.uint32, bitcast=True)
        expos = ((x_I << 1) >> 24).to(tl.int32, bitcast=True) - scaler - 127
        shared_expo = tl.max(expos) + 1
        shift1 = shared_expo - expos
        shift2 = tl.where(-126 - expos > 0, -126 - expos, 0)
        nbits2round = 16 + shift1 + shift2

        # Perform mantissa rounding
        bouncer = ((nbits2round << 23) + (x_I & 0x7F800000)).to(tl.float32, bitcast=True)
        x += bouncer
        x -= bouncer

        # Recompute everything
        x_I = x.to(tl.uint32, bitcast=True)
        signs = x_I >> 31
        expos = ((x_I << 1) >> 24).to(tl.int32, bitcast=True) - scaler
        shared_expo = tl.max(expos) + 1
        shift1 = shared_expo - expos
        shift2 = tl.where(1 - expos > 0, 1 - expos, 0)
        nbits2round = 16 + shift1 + shift2
        mantissas = (x_I << 9) >> 9
        leading_bit = tl.where(mantissas == 0 and shift2 != 0, 0, 1)
        mantissas |= leading_bit << 23
        mantissas >>= nbits2round    

        # Compute output offset
        output_local_start = output_block_start_id + i * output_stride_r
        output_elem_offsets = output_local_start + tl.arange(0, BLOCK_SIZE_N) * output_stride_c
        output_elem_mask = output_elem_offsets < (output_row_start_id + output_stride_r + i * output_stride_r)
        output_expo_offset = output_local_start + BLOCK_SIZE_N * output_stride_c
        output_expo_mask = output_expo_offset < (output_row_start_id + output_stride_r + i * output_stride_r)
        
        tl.store(output_ptr + output_elem_offsets, (signs << 7) | mantissas, mask=output_elem_mask)
        tl.store(output_ptr + output_expo_offset, shared_expo, mask=output_expo_mask)

@triton.jit
def msfp16_matmul_kernel(x_ptr, y_ptr, output_ptr,
                         x_stride_r, x_stride_c, 
                         y_stride_r, y_stride_c,
                         o_stride_r, o_stride_c,
                         BLOCK_SIZE_M: tl.constexpr,
                         BLOCK_SIZE_N: tl.constexpr,
                         BLOCK_SIZE_K: tl.constexpr
                         ):
    
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

            x_data_indices = x_local_offset + tl.arange(0, BLOCK_SIZE_K - 1) * x_stride_c
            x_shared_expo_index = x_local_offset + (BLOCK_SIZE_K - 1) * x_stride_c
            y_data_indices = y_local_offset + (tl.arange(0, BLOCK_SIZE_K - 1) * y_stride_r)
            y_shared_expo_index = y_local_offset + ((BLOCK_SIZE_K - 1) * y_stride_r)

            x_raw_data = tl.load(x_ptr + x_data_indices).to(tl.uint8)
            x_shared_expo = tl.load(x_ptr + x_shared_expo_index).to(tl.int32) - 127
            y_raw_data = tl.load(y_ptr + y_data_indices).to(tl.uint8)
            y_shared_expo = tl.load(y_ptr + y_shared_expo_index).to(tl.int32) - 127

            x_sign = (x_raw_data >> 7) & 0x1
            x_data = (x_raw_data & 0x7F).to(tl.int32)
            y_sign = (y_raw_data >> 7) & 0x1
            y_data = (y_raw_data & 0x7F).to(tl.int32)

            product_expo = (x_shared_expo + y_shared_expo - 14).to(tl.float32)
            product_sign = ((x_sign ^ y_sign) * -2) + 1
            int_sum = tl.sum(x_data * y_data * product_sign).to(tl.float32)
            partial_sum = int_sum * (tl.math.exp2(product_expo))

            target_row = x_row + i
            target_col = pid_n * BLOCK_SIZE_N + j
            output_offset = target_row * o_stride_r + target_col * o_stride_c
            tl.atomic_add(output_ptr + output_offset, partial_sum)

########################################################################

# translate a 2D tensor from fp32 precision to msfp format
def fp32_to_msfp16(x: torch.Tensor):
    assert x.is_cuda
    torch.cuda.set_device(x.device.index)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    num_warps = 4
    
    if x.shape[1] % BLOCK_SIZE_N != 0:
        x = F.pad(x, (0, BLOCK_SIZE_N - (x.shape[1] % BLOCK_SIZE_N)), value=0.0)
    if x.shape[0] % BLOCK_SIZE_M != 0:
        x = F.pad(x, (0, 0, 0, BLOCK_SIZE_M - (x.shape[0] % BLOCK_SIZE_M)), value=0.0)
    M, N = x.shape
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    output = torch.empty([x.size(0), x.size(1) * (BLOCK_SIZE_N + 1) // BLOCK_SIZE_N], dtype=torch.uint8).to(x.device)
    fp32_to_msfp16_kernel[(num_pid_m, num_pid_n)](x, output,
                                                  x.stride(0), x.stride(1),
                                                  output.stride(0), output.stride(1), 
                                                  BLOCK_SIZE_M, BLOCK_SIZE_N, num_warps=num_warps)

    return output

# Performs matrix multiplication xy^T
# Both x and y are in msfp format
# For simplicity, y doesn't need to be transposed
# Note that BLOCK_SIZE_K must be 1 + power of 2
def msfp16_matmul(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda

    BLOCK_SIZE_K = 17
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    num_warps = 4

    assert x.shape[1] == y.shape[0]
    assert x.shape[0] % BLOCK_SIZE_M == 0
    assert y.shape[1] % BLOCK_SIZE_N == 0                     # For MSFP16, the tensor must be padded by 16
    assert x.shape[1] % BLOCK_SIZE_K == 0                     # Multiply dimension must be divisible by 17 (bounding box size + 1)

    num_pid_m = triton.cdiv(x.shape[0], BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(y.shape[1], BLOCK_SIZE_N)
    num_pid_k = triton.cdiv(x.shape[1], BLOCK_SIZE_K)

    output = torch.zeros([x.shape[0], y.shape[1]], dtype=torch.float32).to(x.device)
    msfp16_matmul_kernel[(num_pid_m, num_pid_n, num_pid_k)](x, y, output, 
                                                            x.stride(0), x.stride(1), 
                                                            y.stride(0), y.stride(1),
                                                            output.stride(0), output.stride(1), 
                                                            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, 
                                                            num_warps=num_warps)

    return output



