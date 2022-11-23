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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_CLIP_BY_VALUE_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_CLIP_BY_VALUE_FUSION_H_

#include <memory>
#include "backend/common/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class ClipByValueFusion : public PatternProcessPass {
 public:
  explicit ClipByValueFusion(bool multigraph = true) : PatternProcessPass("clip_by_value_fusion", multigraph) {
    maximum_input0_ = std::make_shared<Var>();
    maximum_input1_ = std::make_shared<Var>();
  }
  ~ClipByValueFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  VarPtr maximum_input0_;
  VarPtr maximum_input1_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_CLIP_BY_VALUE_FUSION_H_
