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

#include "transform/graph_ir/op_declare/ctc_ops_declare.h"

namespace mindspore::transform {
// CTCLoss
INPUT_MAP(CTCLoss) = {{1, INPUT_DESC(inputs)},
                      {2, INPUT_DESC(labels_indices)},
                      {3, INPUT_DESC(labels_values)},
                      {4, INPUT_DESC(sequence_length)}};//收录四组变量并分别与标准比较其容量并返回对应的值，以input_map_为key储存相应内容
ATTR_MAP(CTCLoss) = {
  {"preprocess_collapse_repeated", ATTR_DESC(preprocess_collapse_repeated, AnyTraits<bool>())},
  {"ctc_merge_repeated", ATTR_DESC(ctc_merge_repeated, AnyTraits<bool>())},
  {"ignore_longer_outputs_than_inputs", ATTR_DESC(ignore_longer_outputs_than_inputs, AnyTraits<bool>())}};//收录四组变量并分别与标准比较其容量并返回对应的值，以attr_map_为key储存相应内容
  /*
    将ignore_longer_outputs_than_inputs处理并存入对应ATTR_DESC结构体的相应变量中
  将ignore_longer_outputs_than_inputs内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为ATTR_DESC结构体
//AnyTraits作用:对相应变量处理并存入对应AttrDesc结构体的相应变量中
存入MaxPool对应空间并用attr_map_指针保存
其中AnyTraits<>的作用为将<>内类型进行构建
*/
OUTPUT_MAP(CTCLoss) = {{0, OUTPUT_DESC(loss)}, {1, OUTPUT_DESC(gradient)}};//将收录的两组数据的类型与标准进行对比，后进行空间调整，并用指针output_map_为key存储相应内容
REG_ADPT_DESC(CTCLoss, kNameCTCLoss, ADPT_DESC(CTCLoss))//将CTCLoss处理并存入对应REG_ADPT_Desc结构体的相应变量中
  //将name变量内容转为字符串变量并存储至结构体的name变量中
  //引用Operator空间并将指针所指的类转为REG_ADPT_Desc结构体

// CTCGreedyDecoder
INPUT_MAP(CTCGreedyDecoder) = {{1, INPUT_DESC(inputs)}, {2, INPUT_DESC(sequence_length)}};
ATTR_MAP(CTCGreedyDecoder) = {{"merge_repeated", ATTR_DESC(merge_repeated, AnyTraits<bool>())}};
OUTPUT_MAP(CTCGreedyDecoder) = {{0, OUTPUT_DESC(decoded_indices)},
                                {1, OUTPUT_DESC(decoded_values)},
                                {2, OUTPUT_DESC(decoded_shape)},
                                {3, OUTPUT_DESC(log_probability)}};
REG_ADPT_DESC(CTCGreedyDecoder, kNameCTCGreedyDecoder, ADPT_DESC(CTCGreedyDecoder))
}  // namespace mindspore::transform
