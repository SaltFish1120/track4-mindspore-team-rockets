/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef LITE_MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_CONV_TUPLEGETITEM_FUSION_H_
#define LITE_MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_CONV_TUPLEGETITEM_FUSION_H_
#include <string>
#include "backend/common/optimizer/optimizer.h"
namespace mindspore::opt {
class ConvTupleGetItemFusion : public PatternProcessPass {
 public:
  explicit ConvTupleGetItemFusion(const std::string &name = "ConvTupleGetItemFusion", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {}
  ~ConvTupleGetItemFusion() override = default;

 private:
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};
}  // namespace mindspore::opt

#endif  // LITE_MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_CONV_TUPLEGETITEM_FUSION_H_
