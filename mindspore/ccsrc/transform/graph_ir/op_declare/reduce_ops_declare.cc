/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/reduce_ops_declare.h"//按照路径寻找以下文件，导入到本文件
#include <vector>//提供vector数组构建函数模版等

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
// BNTrainingReduce
INPUT_MAP(BNTrainingReduce) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入BNTrainingReduce对应空间内并用input_map指针保存
ATTR_MAP(BNTrainingReduce) = EMPTY_ATTR_MAP;//将空变量存入BNTrainingReduce对应空间并用attr_map_指针保存
OUTPUT_MAP(BNTrainingReduce) = {{0, OUTPUT_DESC(sum)}, {1, OUTPUT_DESC(square_sum)}};
//将变量sum、square_sum处理并存入对应OutputDesc结构体的相应变量中，存入BNTrainingReduce对应空间内并用output_map_指针保存
REG_ADPT_DESC(BNTrainingReduce, kNameBNTrainingReduce, ADPT_DESC(BNTrainingReduce))
//构造指向BNTrainingReduce的指针并储存，创建结构体RegAdptDescBNTrainingReduce

// BNTrainingReduceGrad
INPUT_MAP(BNTrainingReduceGrad) = {{1, INPUT_DESC(grads)},         {2, INPUT_DESC(x)},     {3, INPUT_DESC(diff_scale)},
                                   {4, INPUT_DESC(diff_offset)},   {5, INPUT_DESC(scale)}, {6, INPUT_DESC(batch_mean)},
                                   {7, INPUT_DESC(batch_variance)}};
//将变量grads、x、diff_scale、diff_offset、scale、batch_mean、batch_variance处理并存入对应InputDesc结构体的相应变量中
//存入BNTrainingReduceGrad对应空间内并用input_map指针保存
ATTR_MAP(BNTrainingReduceGrad) = {{"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                       存入BNTrainingReduceGrad对应空间并用attr_map_指针保存
//                                                                                       其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(BNTrainingReduceGrad) = {{0, OUTPUT_DESC(y)}};
//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入BNTrainingReduceGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(BNTrainingReduceGrad, kNameBNTrainingReduceGrad, ADPT_DESC(BNTrainingReduceGrad))
//构造指向BNTrainingReduceGrad的指针并储存，创建结构体RegAdptDescBNTrainingReduceGrad

// BNTrainingUpdate
INPUT_MAP(BNTrainingUpdate) = {{1, INPUT_DESC(x)},       {2, INPUT_DESC(sum)},    {3, INPUT_DESC(square_sum)},
                               {4, INPUT_DESC(scale)},   {5, INPUT_DESC(offset)}, {6, INPUT_DESC(mean)},
                               {7, INPUT_DESC(variance)}};
//将变量x、sum、square_sum、scale、offset、mean、variance处理并存入对应InputDesc结构体的相应变量中
//存入BNTrainingUpdate对应空间内并用input_map指针保存
ATTR_MAP(BNTrainingUpdate) = {{"factor", ATTR_DESC(factor, AnyTraits<float>())},
                              {"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                   存入BNTrainingUpdate对应空间并用attr_map_指针保存
//                                                                                   其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(BNTrainingUpdate) = {{0, OUTPUT_DESC(y)},
                                {1, OUTPUT_DESC(mean)},
                                {2, OUTPUT_DESC(variance)},
                                {3, OUTPUT_DESC(batch_mean)},
                                {4, OUTPUT_DESC(batch_variance)}};
//将变量y、mean、variance、batch_mean、batch_variance处理并存入对应OutputDesc结构体的相应变量中，存入BNTrainingUpdate对应空间内并用output_map_指针保存
REG_ADPT_DESC(BNTrainingUpdate, kNameBNTrainingUpdate, ADPT_DESC(BNTrainingUpdate))
//构造指向BNTrainingUpdate的指针并储存，创建结构体RegAdptDescBNTrainingUpdate

// BNTrainingUpdateGrad
INPUT_MAP(BNTrainingUpdateGrad) = {
  {1, INPUT_DESC(grads)}, {2, INPUT_DESC(x)}, {3, INPUT_DESC(batch_mean)}, {4, INPUT_DESC(batch_variance)}};
//将变量grads、x、batch_mean、batch_variance处理并存入对应InputDesc结构体的相应变量中
//存入BNTrainingUpdateGrad对应空间内并用input_map指针保存
ATTR_MAP(BNTrainingUpdateGrad) = {{"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                       存入BNTrainingUpdateGrad对应空间并用attr_map_指针保存
//                                                                                       其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(BNTrainingUpdateGrad) = {{0, OUTPUT_DESC(diff_scale)}, {1, OUTPUT_DESC(diff_offset)}};
//将变量diff_scale、diff_offset处理并存入对应OutputDesc结构体的相应变量中，存入BNTrainingUpdateGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(BNTrainingUpdateGrad, kNameBNTrainingUpdateGrad, ADPT_DESC(BNTrainingUpdateGrad))
//构造指向BNTrainingUpdateGrad的指针并储存，创建结构体RegAdptDescBNTrainingUpdateGrad

// ReduceAnyD
INPUT_MAP(ReduceAnyD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入ReduceAnyD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(ReduceAnyD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};//对实现ReduceAnyD的对象转换后转移出相应存储空间
ATTR_MAP(ReduceAnyD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                存入ReduceAnyD对应空间并用attr_map_指针保存
//                                                                                其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ReduceAnyD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入ReduceAnyD对应空间内并用output_map_指针保存
REG_ADPT_DESC(ReduceAnyD, kNameReduceAnyD, ADPT_DESC(ReduceAnyD))//构造指向ReduceAnyD的指针并储存，创建结构体RegAdptDescReduceAnyD

// ReduceSumD
INPUT_MAP(ReduceSumD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入ReduceSumD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(ReduceSumD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};//对实现ReduceSumD的对象转换后转移出相应存储空间
ATTR_MAP(ReduceSumD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                存入ReduceSumD对应空间并用attr_map_指针保存
//                                                                                其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ReduceSumD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入ReduceSumD对应空间内并用output_map_指针保存
REG_ADPT_DESC(ReduceSumD, prim::kPrimReduceSum->name(), ADPT_DESC(ReduceSumD))//构造指向ReduceSumD的指针并储存，创建结构体RegAdptDescReduceSumD

// ReduceProdD
INPUT_MAP(ReduceProdD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入ReduceProdD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(ReduceProdD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};//对实现ReduceProdD的对象转换后转移出相应存储空间
ATTR_MAP(ReduceProdD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                 存入ReduceProdD对应空间并用attr_map_指针保存
//                                                                                 其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ReduceProdD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入ReduceProdD对应空间内并用output_map_指针保存
REG_ADPT_DESC(ReduceProdD, kNameReduceProd, ADPT_DESC(ReduceProdD))//构造指向ReduceProdD的指针并储存，创建结构体RegAdptDescReduceProdD

// ReduceAllD
INPUT_MAP(ReduceAllD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入ReduceAllD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(ReduceAllD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};//对实现ReduceAllD的对象转换后转移出相应存储空间
ATTR_MAP(ReduceAllD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                存入ReduceAllD对应空间并用attr_map_指针保存
//                                                                                其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ReduceAllD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入ReduceAllD对应空间内并用output_map_指针保存
REG_ADPT_DESC(ReduceAllD, prim::kPrimReduceAll->name(), ADPT_DESC(ReduceAllD))//构造指向ReduceAllD的指针并储存，创建结构体RegAdptDescReduceAllD

// ReduceMeanD
INPUT_MAP(ReduceMeanD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入ReduceMeanD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(ReduceMeanD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};//对实现ReduceMeanD的对象转换后转移出相应存储空间
ATTR_MAP(ReduceMeanD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                 存入ReduceMeanD对应空间并用attr_map_指针保存
//                                                                                 其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ReduceMeanD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入ReduceMeanD对应空间内并用output_map_指针保存
REG_ADPT_DESC(ReduceMeanD, prim::kPrimReduceMean->name(), ADPT_DESC(ReduceMeanD))//构造指向ReduceMeanD的指针并储存，创建结构体RegAdptDescReduceMeanD

// ReduceMinD
INPUT_MAP(ReduceMinD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入ReduceMinD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(ReduceMinD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};//对实现ReduceMinD的对象转换后转移出相应存储空间
ATTR_MAP(ReduceMinD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                 存入ReduceMinD对应空间并用attr_map_指针保存
//                                                                                 其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ReduceMinD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入ReduceMinD对应空间内并用output_map_指针保存
REG_ADPT_DESC(ReduceMinD, prim::kPrimReduceMin->name(), ADPT_DESC(ReduceMinD))//构造指向ReduceMinD的指针并储存，创建结构体RegAdptDescReduceMinD

// ReduceMaxD
INPUT_MAP(ReduceMaxD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入ReduceMaxD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(ReduceMaxD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};//对实现ReduceMaxD的对象转换后转移出相应存储空间
ATTR_MAP(ReduceMaxD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                              存入ReduceMaxD对应空间并用attr_map_指针保存
//                                                                                              其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ReduceMaxD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入ReduceMaxD对应空间内并用output_map_指针保存
REG_ADPT_DESC(ReduceMaxD, prim::kPrimReduceMax->name(), ADPT_DESC(ReduceMaxD))//构造指向ReduceMaxD的指针并储存，创建结构体RegAdptDescReduceMaxD
}  // namespace mindspore::transform
