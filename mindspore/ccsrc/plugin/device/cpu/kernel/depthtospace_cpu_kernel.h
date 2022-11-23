/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DEPTHTOSPACE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DEPTHTOSPACE_CPU_KERNEL_H_

#include <vector>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class DepthToSpaceCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  DepthToSpaceCpuKernelMod() = default;
  ~DepthToSpaceCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using DepthToSpaceFunc = std::function<bool(DepthToSpaceCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                              const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, DepthToSpaceFunc>> func_list_;
  DepthToSpaceFunc kernel_func_;

  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  size_t block_size_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DEPTHTOSPACE_CPU_KERNEL_H_