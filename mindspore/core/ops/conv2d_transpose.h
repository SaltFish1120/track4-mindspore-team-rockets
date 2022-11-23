/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_CONV2D_TRANSPOSE_H_
#define MINDSPORE_CORE_OPS_CONV2D_TRANSPOSE_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "mindapi/base/format.h"

namespace mindspore {
namespace ops {
constexpr auto kNameConv2DTranspose = "Conv2DTranspose";
/// \brief 2D transposed convolution layer. Refer to Python API @ref mindspore.nn.Conv2dTranspose for more details.
class MIND_API Conv2DTranspose : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Conv2DTranspose);
  /// \brief Constructor.
  Conv2DTranspose() : BaseOperator(kNameConv2DTranspose) {
    InitIOName({"out_backprop", "filter", "input_sizes"}, {"output"});
  }
  explicit Conv2DTranspose(const std::string k_name) : BaseOperator(k_name) {
    InitIOName({"out_backprop", "filter", "input_sizes"}, {"output"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.nn.Conv2dTranspose for the inputs.
  void Init(int64_t in_channel, int64_t out_channel, const std::vector<int64_t> &kernel_size, int64_t mode = 1,
            const PadMode &pad_mode = VALID, const std::vector<int64_t> &pad = {0, 0, 0, 0},
            const std::vector<int64_t> &stride = {1, 1}, const std::vector<int64_t> &dilation = {1, 1},
            int64_t group = 1, const Format &format = NCHW, const std::vector<int64_t> &pad_list = {0, 0, 0, 0});
  /// \brief Set in_channel.
  void set_in_channel(int64_t in_channel);
  /// \brief Set out_channel.
  void set_out_channel(int64_t out_channel);
  /// \brief Set kernel_size.
  virtual void set_kernel_size(const std::vector<int64_t> &kernel_size);
  /// \brief Set stride.
  void set_stride(const std::vector<int64_t> &stride);
  /// \brief Set dilation.
  virtual void set_dilation(const std::vector<int64_t> &dilation);
  /// \brief Set pad_mode.
  void set_pad_mode(const PadMode &pad_mode);
  /// \brief Set pad.
  void set_pad(const std::vector<int64_t> &pad);
  /// \brief Set mode.
  void set_mode(int64_t mode);
  /// \brief Set group.
  void set_group(int64_t group);
  /// \brief Set format.
  void set_format(const Format &format);
  /// \brief Set pad_list.
  void set_pad_list(const std::vector<int64_t> &pad_list);

  /// \brief Get in_channel.
  ///
  /// \return in_channel.
  int64_t get_in_channel() const;
  /// \brief Get out_channel.
  ///
  /// \return out_channel.
  int64_t get_out_channel() const;
  /// \brief Get kernel_size.
  ///
  /// \return kernel_size.
  std::vector<int64_t> get_kernel_size() const;
  /// \brief Get stride.
  ///
  /// \return stride.
  std::vector<int64_t> get_stride() const;
  /// \brief Get dilation.
  ///
  /// \return dilation.
  std::vector<int64_t> get_dilation() const;
  /// \brief Get pad_mode.
  ///
  /// \return pad_mode.
  PadMode get_pad_mode() const;
  /// \brief Get pad.
  ///
  /// \return pad.
  std::vector<int64_t> get_pad() const;
  /// \brief Get mode.
  ///
  /// \return mode.
  int64_t get_mode() const;
  /// \brief Get group.
  ///
  /// \return group.
  int64_t get_group() const;
  /// \brief Get format.
  ///
  /// \return format.
  Format get_format() const;
  /// \brief Get pad_list.
  ///
  /// \return pad_list.
  std::vector<int64_t> get_pad_list() const;
};
abstract::AbstractBasePtr Conv2DTransposeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_CONV2D_TRANSPOSE_H_