/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <x86intrin.h>

// nnacl gemm in x86 fma asm code
void nnacl_gemm_fma_1x8_kernel_nc8hw8_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                           const size_t act_flag, const size_t row_block, const size_t col_block,
                                           const size_t deep, const size_t src_stride, const size_t dst_stride,
                                           const size_t inc_flag) {
  size_t deep_t = deep >> 3;
  size_t dst_stride_t = dst_stride << 2;
  size_t src_stride_t = src_stride << 2;
  asm volatile(
    // inc in deep
    "and $0x1, %[inc_flag]\n"
    "je 0f\n"
    "vmovups 0(%[dst]), %%ymm0\n"
    "jmp 2f\n"
    "0:\n"
    "cmpq $0, %[bias]\n"
    "je 1f\n"
    "vmovaps 0(%[bias]), %%ymm0\n"
    "jmp 2f\n"
    "1:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "2:\n"
    :
    : [ dst ] "r"(dst), [ bias ] "r"(bias), [ dst_stride ] "r"(dst_stride_t), [ inc_flag ] "r"(inc_flag)
    : "%ymm0");
  asm volatile(
    "0:\n"
    // block 0
    "vmovaps 0(%[weight]), %%ymm15\n"
    "vbroadcastss 0(%[src]), %%ymm14\n"
    "vfmadd231ps %%ymm0, %%ymm14, %%ymm15\n"
    // block 1
    "vmovaps 32(%[weight]), %%ymm15\n"
    "vbroadcastss 1(%[src]), %%ymm14\n"
    "vfmadd231ps %%ymm0, %%ymm14, %%ymm15\n"
    // block 2
    "vmovaps 64(%[weight]), %%ymm15\n"
    "vbroadcastss 2(%[src]), %%ymm14\n"
    "vfmadd231ps %%ymm0, %%ymm14, %%ymm15\n"
    // block 3
    "vmovaps 96(%[weight]), %%ymm15\n"
    "vbroadcastss 3(%[src]), %%ymm14\n"
    "vfmadd231ps %%ymm0, %%ymm14, %%ymm15\n"
    // block 4
    "vmovaps 128(%[weight]), %%ymm15\n"
    "vbroadcastss 4(%[src]), %%ymm14\n"
    "vfmadd231ps %%ymm0, %%ymm14, %%ymm15\n"
    // block 5
    "vmovaps 160(%[weight]), %%ymm15\n"
    "vbroadcastss 5(%[src]), %%ymm14\n"
    "vfmadd231ps %%ymm0, %%ymm14, %%ymm15\n"
    // block 6
    "vmovaps 192(%[weight]), %%ymm15\n"
    "vbroadcastss 6(%[src]), %%ymm14\n"
    "vfmadd231ps %%ymm0, %%ymm14, %%ymm15\n"
    // block 7
    "vmovaps 224(%[weight]), %%ymm15\n"
    "vbroadcastss 7(%[src]), %%ymm14\n"
    "vfmadd231ps %%ymm0, %%ymm14, %%ymm15\n"
    "dec %[deep]\n"
    "add 256, %[weight]\n"
    "add %[src_stride], %[src]\n"
    "jg 0b\n"

    "movq %[inc_flag], %%rax\n"
    "and $0x2, %%eax\n"
    "je 3f\n"
    "movq %[act_flag], %%rax\n"
    "and $0x3, %%eax\n"
    "je 3f\n"
    // relu
    "vxorps %%ymm15, %%ymm15, %%ymm15\n"
    "vmaxps %%ymm0, %%ymm15, %%ymm0\n"
    "and $0x1, %%eax\n"
    "je 3f\n"
    // relu6
    "mov $0x40C00000, %%eax\n"
    "vmovd %%eax, %%xmm14\n"
    "vpermps %%ymm14, %%ymm15, %%ymm14\n"
    "vminps %%ymm0, %%ymm14, %%ymm0\n"
    "3:\n"
    "vmovups %%ymm0, 0(%[dst])\n"
    :
    : [ src ] "r"(src), [ src_stride ] "r"(src_stride_t), [ weight ] "r"(weight), [ deep ] "r"(deep_t),
      [ inc_flag ] "r"(inc_flag), [ act_flag ] "r"(act_flag), [ dst ] "r"(dst), [ dst_stride ] "r"(dst_stride_t)
    : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
      "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}
