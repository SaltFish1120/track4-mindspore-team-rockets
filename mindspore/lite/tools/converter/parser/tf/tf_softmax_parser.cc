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
#include "tools/converter/parser/tf/tf_softmax_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/softmax.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TFSoftmaxParser::Parse(const tensorflow::NodeDef &tf_op,
                                     const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                     std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Softmax>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  tensorflow::AttrValue attr_value;
  int axis = -1;
  if (TensorFlowUtils::FindAttrValue(tf_op, "axis", &attr_value)) {
    axis = static_cast<int32_t>(attr_value.i());
  }
  prim->set_axis({axis});

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim->GetPrim();
}
TFNodeRegistrar g_tfSoftmaxParser("Softmax", new TFSoftmaxParser());
}  // namespace lite
}  // namespace mindspore