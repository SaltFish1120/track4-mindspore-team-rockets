/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CUDA_UNARY_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CUDA_UNARY_H_

#include <memory>
#include <vector>
#include "src/runtime/kernel/cuda/cuda_kernel.h"
#include "cuda_impl/cuda_class/unary_helper.h"

namespace mindspore::kernel {
class UnaryCudaKernel : public CudaKernel {
 public:
  UnaryCudaKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                  const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : CudaKernel(parameter, inputs, outputs, ctx) {}
  ~UnaryCudaKernel() override = default;  // cudaFree
  int Prepare() override;
  int Run() override;

 private:
  std::shared_ptr<cukernel::UnaryHelperGpuKernel<float>> unary_helper_{nullptr};
};
}  // namespace mindspore::kernel
#endif
