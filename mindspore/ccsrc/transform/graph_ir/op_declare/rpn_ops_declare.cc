/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/rpn_ops_declare.h"//按照路径寻找以下文件，导入到本文件

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
// NMSWithMask
INPUT_MAP(NMSWithMask) = {{1, INPUT_DESC(box_scores)}};//将变量box_scores处理并存入对应InputDesc结构体的相应变量中，存入NMSWithMask对应空间内并用input_map指针保存
ATTR_MAP(NMSWithMask) = {{"iou_threshold", ATTR_DESC(iou_threshold, AnyTraits<float>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                          存入NMSWithMask对应空间并用attr_map_指针保存
//                                                                                          其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(NMSWithMask) = {
  {0, OUTPUT_DESC(selected_boxes)}, {1, OUTPUT_DESC(selected_idx)}, {2, OUTPUT_DESC(selected_mask)}};
//将变量selected_boxes、selected_idx、selected_mask处理并存入对应OutputDesc结构体的相应变量中，存入NMSWithMask对应空间内并用output_map_指针保存
REG_ADPT_DESC(NMSWithMask, kNameNMSWithMask, ADPT_DESC(NMSWithMask))//构造指向NMSWithMask的指针并储存，创建结构体RegAdptDescNMSWithMask
}  // namespace mindspore::transform
