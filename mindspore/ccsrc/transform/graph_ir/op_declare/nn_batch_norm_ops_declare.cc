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

#include "transform/graph_ir/op_declare/nn_batch_norm_ops_declare.h"
#include <vector>

namespace mindspore::transform {
// BatchNorm
INPUT_MAP(BatchNorm) = {{1, INPUT_DESC(x)},
                        {2, INPUT_DESC(scale)},
                        {3, INPUT_DESC(offset)},
                        {4, INPUT_DESC(mean)},
                        {5, INPUT_DESC(variance)}};
ATTR_MAP(BatchNorm) = {{"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
                       {"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())},
                       {"is_training", ATTR_DESC(is_training, AnyTraits<bool>())}};
OUTPUT_MAP(BatchNorm) = {{0, OUTPUT_DESC(y)},
                         {1, OUTPUT_DESC(batch_mean)},
                         {2, OUTPUT_DESC(batch_variance)},
                         {3, OUTPUT_DESC(reserve_space_1)},
                         {4, OUTPUT_DESC(reserve_space_2)}};
// BNInference is BatchNorm for caffe
//用这一部分做注释，宏定义用的比较全
INPUT_MAP(BNInference) = {{1, INPUT_DESC(x)},        {2, INPUT_DESC(mean)},  {3, INPUT_DESC(variance)},
                          {4, INPUT_DESC(momentum)}, {5, INPUT_DESC(scale)}, {6, INPUT_DESC(offset)}};
//以其中一句为例
/*
    将x处理并存入对应INPUT_DESC结构体的相应变量中
  将x内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为INPUT_DESC结构体
将BNInference的类型与标准进行对比，后进行空间调整，并用指针input_map_为key存储相应内容
*/
ATTR_MAP(BNInference) = {{"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())},
                         {"use_global_stats", ATTR_DESC(use_global_stats, AnyTraits<bool>())},
                         {"mode", ATTR_DESC(mode, AnyTraits<int64_t>())}};
/*
  将epsilon处理并存入对应ATTR_DESC结构体的相应变量中
  将epsilon内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为ATTR_DESC结构体
//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入MaxPool对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建
//std:vector<>的作用为构建<>内类型的容量可变的数组
将BNInference的类型与标准进行对比，后进行空间调整，并用指针attr_map_为key存储相应内容
*/
OUTPUT_MAP(BNInference) = {{0, OUTPUT_DESC(y)}};/*
  将y处理并存入对应OUTPUT_DESC结构体的相应变量中
  将y内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为INPUT_DESC结构体
将BNInference的类型与标准进行对比，后进行空间调整，并用指针output_map_为key存储相应内容
*/

REG_ADPT_DESC(BNInference, kNameBNInference, ADPT_DESC(BNInference))/*
  将NInference处理并存入对应ADPT_DESC结构体的相应变量中
  将NInference内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为ADPT_DESC结构体
  
  再将NInference处理并存入对应REG_ADPT_DESC结构体的相应变量中
  将NInference内容转为字符串变量并存储至REG_ADPT_DESC的结构体的name变量中
  引用Operator空间并将指针所指的类转为REG_ADPT_DESC结构体
*/          
REG_ADPT_DESC(BatchNorm, kNameBatchNorm, ADPT_DESC(BatchNorm))
REG_ADPT_DESC(FusedBatchNorm, kNameFusedBatchNorm, ADPT_DESC(BatchNorm))

// BatchNormGrad
INPUT_MAP(BatchNormGrad) = {{1, INPUT_DESC(y_backprop)},
                            {2, INPUT_DESC(x)},
                            {3, INPUT_DESC(scale)},
                            {4, INPUT_DESC(reserve_space_1)},
                            {5, INPUT_DESC(reserve_space_2)}};
ATTR_MAP(BatchNormGrad) = {{"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
                           {"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())},
                           {"is_training", ATTR_DESC(is_training, AnyTraits<bool>())}};
OUTPUT_MAP(BatchNormGrad) = {{0, OUTPUT_DESC(x_backprop)},
                             {1, OUTPUT_DESC(scale_backprop)},
                             {2, OUTPUT_DESC(offset_backprop)},
                             {3, OUTPUT_DESC(reserve_space_4)},
                             {4, OUTPUT_DESC(reserve_space_5)}};
REG_ADPT_DESC(BatchNormGrad, kNameBatchNormGrad, ADPT_DESC(BatchNormGrad))

// L2NormalizeGrad
INPUT_MAP(L2NormalizeGrad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(y)}, {3, INPUT_DESC(dy)}};
ATTR_MAP(L2NormalizeGrad) = {
  {"axis", ATTR_DESC(dim, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"epsilon", ATTR_DESC(eps, AnyTraits<float>())}};
OUTPUT_MAP(L2NormalizeGrad) = {{0, OUTPUT_DESC(dx)}};
REG_ADPT_DESC(L2NormalizeGrad, kNameL2NormalizeGrad, ADPT_DESC(L2NormalizeGrad))

// L2Normalize
INPUT_MAP(L2Normalize) = {{1, INPUT_DESC(x)}};
ATTR_MAP(L2Normalize) = {
  {"axis", ATTR_DESC(axis, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"epsilon", ATTR_DESC(eps, AnyTraits<float>())}};
OUTPUT_MAP(L2Normalize) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(L2Normalize, kNameL2Normalize, ADPT_DESC(L2Normalize))
}  // namespace mindspore::transform
