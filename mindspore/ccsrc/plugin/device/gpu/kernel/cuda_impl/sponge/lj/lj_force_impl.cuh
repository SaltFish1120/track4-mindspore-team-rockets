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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_LJ_LJ_FORCE_IMPL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_LJ_LJ_FORCE_IMPL_H_

#include <curand_kernel.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

void LJForce(const int atom_numbers, const float cutoff_square, const int *uint_crd_f, const int *LJtype,
             const float *charge, const float *scaler_f, float *uint_crd_with_LJ, int *nl_atom_numbers,
             int *nl_atom_serial, int *nl, const float *d_LJ_A, const float *d_LJ_B, float *frc_f, cudaStream_t stream);

#endif
