# Quantization

A custom tensor quantization library featuring various precision formats

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)

## Introduction

Welcome to Quantization, a project dedicated to custom tensor using custom Triton GPU kernels. This project aims to contribute in optimizing and accelerating various machine learning models through providing quantization utilities, particularly targeting various precision formats like MSFP, FP8_E4M3, FP8_E5M2, and et cetera.

Quantization is a crucial step in deep learning model optimization, especially for deployment on resource-constrained devices like edge devices or mobile phones. By reducing the bit precision of model parameters and activations, quantization can significantly reduce memory footprint and computational requirements while preserving model accuracy to a certain extent.

In this project, we provide custom Triton GPU kernels tailored for efficient quantization operations. These kernels enable native matrix multiplication support for different quantization formats, allowing seamless integration into existing deep learning frameworks like PyTorch.


Whether you're interested in optimizing deep learning models for edge deployment or exploring novel quantization techniques, Quantization offers a comprehensive toolkit and framework for experimentation and research in this exciting field of deep learning optimization.

In the following sections, we'll delve into the features of this project, guide you through the necessary requirements, and provide instructions on how to use the provided tools and scripts for quantization and evaluation tasks. Let's dive in!

## Features

As of current, the following list summarizes the overall features of this library.

- Custom Triton GPU kernels dedicated for quantization & matrix multiplication
- MSFP conversion + Matrix multiplication
- FP8 precision format (E4M3, E5M2)

## Requirements

Since the custom GPU kernels are implemented by [triton](https://triton-lang.org/main/index.html), one must install triton before using this library.

Also, as usual, [PyTorch](https://pytorch.org/) and CUDAToolkit is required as well.

## Usage

The utilities included in this library can be used in the following manner.

```python
import torch
import quantize.fp8 as fp8
import quantize.msfp as msfp

x = torch.randn((16, 32), dtype=torch.float32)
y = torch.randn((16, 32), dtype=torch.float32)

x_84, x_85 = x.to(torch.float8_e4m3fn), x.to(torch.float8_e5m2)
y_84, y_85 = y.to(torch.float8_e4m3fn), y.to(torch.float8_e5m2)
fp8.fp8e4m3_matmul(x_84, y_84.transpose(0, 1))
fp8.fp8e5m2_matmul(x_85, y_85.transpose(0, 1))

x_msfp = msfp.fp32_to_msfp16(x)
y_msfp = msfp.fp32_to_msfp16(y)
msfp.msfp16_matmul(x_msfp, y_msfp.transpose(0, 1))
```
