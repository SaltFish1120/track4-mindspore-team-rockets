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

#include "transform/graph_ir/op_declare/nn_pooling_ops_declare.h"//按照路径寻找以下文件，导入到本文件
#include <vector>//提供vector数组构建函数模版等

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
// MaxPool最大池化
INPUT_MAP(MaxPool) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入MaxPool对应空间内并用input_map指针保存
ATTR_MAP(MaxPool) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                     {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                     {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())},
                     {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                   存入MaxPool对应空间并用attr_map_指针保存
//                                                                                   其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                   std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(MaxPool) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入MaxPool对应空间内并用output_map_指针保存
REG_ADPT_DESC(MaxPool, kNameMaxPool, ADPT_DESC(MaxPool))//构造指向MaxPool的指针并储存，创建结构体RegAdptDescMaxPool

// MaxPool3D
INPUT_MAP(MaxPool3D) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入MaxPool3D对应空间内并用input_map指针保存
ATTR_MAP(MaxPool3D) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())},
                       {"pad_list", ATTR_DESC(pads, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"dilation", ATTR_DESC(dilation, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"ceil_mode", ATTR_DESC(ceil_mode, AnyTraits<int64_t>())},
                       {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                   存入MaxPool3D对应空间并用attr_map_指针保存
//                                                                                   其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                   std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(MaxPool3D) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入MaxPool3D对应空间内并用output_map_指针保存
REG_ADPT_DESC(MaxPool3D, kNameMaxPool3D, ADPT_DESC(MaxPool3D))//构造指向MaxPool3D的指针并储存，创建结构体RegAdptDescMaxPool3D

// MaxPool3DGrad
INPUT_MAP(MaxPool3DGrad) = {{1, INPUT_DESC(orig_x)}, {2, INPUT_DESC(orig_y)}, {3, INPUT_DESC(grads)}};
//将变量orig_x、orig_y、grads处理并存入对应InputDesc结构体的相应变量中，存入MaxPool3DGrad对应空间内并用input_map指针保存
ATTR_MAP(MaxPool3DGrad) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                           {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                           {"pad_list", ATTR_DESC(pads, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                           {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                         存入MaxPool3DGrad对应空间并用attr_map_指针保存
//                                                                                         其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                         std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(MaxPool3DGrad) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入MaxPool3DGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(MaxPool3DGrad, kNameMaxPool3DGrad, ADPT_DESC(MaxPool3DGrad))//构造指向MaxPool3DGrad的指针并储存，创建结构体RegAdptDescMaxPool3DGrad

// MaxPool3DGradGrad
INPUT_MAP(MaxPool3DGradGrad) = {{1, INPUT_DESC(orig_x)}, {2, INPUT_DESC(orig_y)}, {3, INPUT_DESC(grads)}};
//将变量orig_x、orig_y、grads处理并存入对应InputDesc结构体的相应变量中，存入MaxPool3DGradGrad对应空间内并用input_map指针保存
ATTR_MAP(MaxPool3DGradGrad) = {
  {"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                存入MaxPool3DGradGrad对应空间并用attr_map_指针保存
//                                                                其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(MaxPool3DGradGrad) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入MaxPool3DGradGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(MaxPool3DGradGrad, kNameMaxPool3DGradGrad, ADPT_DESC(MaxPool3DGradGrad))//构造指向MaxPool3DGrad的指针并储存，创建结构体RegAdptDescMaxPool3DGradGrad

// AvgPool平均池化
INPUT_MAP(AvgPool) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入AvgPool对应空间内并用input_map指针保存
ATTR_MAP(AvgPool) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                     {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                     {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())},
                     {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                  存入AvgPool对应空间并用attr_map_指针保存
//                                                                                  其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                   std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(AvgPool) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入AvgPool对应空间内并用output_map_指针保存
REG_ADPT_DESC(AvgPool, kNameAvgPool, ADPT_DESC(AvgPool))//构造指向AvgPool的指针并储存，创建结构体RegAdptDescAvgPool

// MaxPoolGrad
INPUT_MAP(MaxPoolGrad) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(grad)}};
//将变量x1、x2、x3处理并存入对应InputDesc结构体的相应变量中，存入MaxPoolGrad对应空间内并用input_map指针保存
ATTR_MAP(MaxPoolGrad) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                         {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                         {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())},
                         {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                      存入MaxPoolGrad对应空间并用attr_map_指针保存
//                                                                                      其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                      std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(MaxPoolGrad) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入MaxPoolGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(MaxPoolGrad, kNameMaxPoolGrad, ADPT_DESC(MaxPoolGrad))//构造指向MaxPoolGrad的指针并储存，创建结构体RegAdptDescMaxPoolGrad

// MaxPoolGradGrad
INPUT_MAP(MaxPoolGradGrad) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(grad)}};
//将变量x1、x2、x3处理并存入对应InputDesc结构体的相应变量中，存入MaxPoolGraddGrad对应空间内并用input_map指针保存
ATTR_MAP(MaxPoolGradGrad) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                             {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                             {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())},
                             {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                           存入MaxPoolGradGrad对应空间并用attr_map_指针保存
//                                                                                          其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                          std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(MaxPoolGradGrad) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入MaxPoolGradGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(MaxPoolGradGrad, kNameMaxPoolGradGrad, ADPT_DESC(MaxPoolGradGrad))//构造指向MaxPoolGradGrad的指针并储存，创建结构体RegAdptDescMaxPoolGradGrad

// avgpoolgrad
INPUT_MAP(AvgPoolGrad) = {{1, INPUT_DESC(orig_input_shape)}, {2, INPUT_DESC(input_grad)}};
//将变量orig_input_shape、input_grad处理并存入对应InputDesc结构体的相应变量中，存入avgpoolgrad对应空间内并用input_map指针保存
ATTR_MAP(AvgPoolGrad) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                         {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                         {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())},
                         {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                       存入AvgPoolGrad对应空间并用attr_map_指针保存
//                                                                                       其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                       std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(AvgPoolGrad) = {{0, OUTPUT_DESC(out_grad)}};//将变量out_grad处理并存入对应OutputDesc结构体的相应变量中，存入AvgPoolGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(AvgPoolGrad, kNameAvgPoolGrad, ADPT_DESC(AvgPoolGrad))//构造指向AvgPoolGrad的指针并储存，创建结构体RegAdptDescAvgPoolGrad

// MaxPoolWithArgmax
INPUT_MAP(MaxPoolWithArgmax) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入MaxPoolWithArgmax对应空间内并用input_map指针保存
ATTR_MAP(MaxPoolWithArgmax) = {
  {"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                              存入MaxPoolWithArgmax对应空间并用attr_map_指针保存
//                                                              其中AnyTraits<>的作用为将<>内类型进行构建
//                                                              std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(MaxPoolWithArgmax) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(argmax)}};
//将变量y、argmax处理并存入对应OutputDesc结构体的相应变量中，存入MaxPoolWithArgmax对应空间内并用output_map_指针保存
REG_ADPT_DESC(MaxPoolWithArgmax, kNameMaxPoolWithArgmax, ADPT_DESC(MaxPoolWithArgmax))
//构造指向MaxPoolWithArgmax的指针并储存，创建结构体RegAdptDescMaxPoolWithArgmax

// MaxPoolGradWithArgmax
INPUT_MAP(MaxPoolGradWithArgmax) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(grad)}, {3, INPUT_DESC(argmax)}};
//将变量x、grad、argmax处理并存入对应InputDesc结构体的相应变量中，存入MaxPoolGradWithArgmax对应空间内并用input_map指针保存
ATTR_MAP(MaxPoolGradWithArgmax) = {
  {"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                              存入MaxPoolGradWithArgmax对应空间并用attr_map_指针保存
//                                                              其中AnyTraits<>的作用为将<>内类型进行构建
//                                                              std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(MaxPoolGradWithArgmax) = {{0, OUTPUT_DESC(y)}};
//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入MaxPoolGradWithArgmax对应空间内并用output_map_指针保存
REG_ADPT_DESC(MaxPoolGradWithArgmax, kNameMaxPoolGradWithArgmax, ADPT_DESC(MaxPoolGradWithArgmax))
//构造指向MaxPoolGradWithArgmax的指针并储存，创建结构体RegAdptDescMaxPoolGradWithArgmax

// MaxPoolGradGradWithArgmax
INPUT_MAP(MaxPoolGradGradWithArgmax) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(grad)}, {3, INPUT_DESC(argmax)}};
//将变量x、grad、argmax处理并存入对应InputDesc结构体的相应变量中，存入MaxPoolGradGradWithArgmax对应空间内并用input_map指针保存
ATTR_MAP(MaxPoolGradGradWithArgmax) = {
  {"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                              存入MaxPoolGradGradWithArgmax对应空间并用attr_map_指针保存
//                                                              其中AnyTraits<>的作用为将<>内类型进行构建
//                                                              std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(MaxPoolGradGradWithArgmax) = {{0, OUTPUT_DESC(y)}};
//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入MaxPoolGradGradWithArgmax对应空间内并用output_map_指针保存
REG_ADPT_DESC(MaxPoolGradGradWithArgmax, kNameMaxPoolGradGradWithArgmax, ADPT_DESC(MaxPoolGradGradWithArgmax))
//构造指向MaxPoolGradGradWithArgmax的指针并储存，创建结构体RegAdptDescMaxPoolGradGradWithArgmax

// Pooling
INPUT_MAP(Pooling) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入Pooling对应空间内并用input_map指针保存
ATTR_MAP(Pooling) = {{"mode", ATTR_DESC(mode, AnyTraits<int64_t>())},
                     {"global", ATTR_DESC(global_pooling, AnyTraits<bool>())},
                     {"kernel_size", ATTR_DESC(window, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                     {"strides", ATTR_DESC(stride, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                     {"pad", ATTR_DESC(pad, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                     {"dilation", ATTR_DESC(dilation, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                     {"round_mode", ATTR_DESC(ceil_mode, AnyTraits<int64_t>())},
                     {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                   存入Pooling对应空间并用attr_map_指针保存
//                                                                                  其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                  std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(Pooling) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Pooling对应空间内并用output_map_指针保存
REG_ADPT_DESC(Pooling, kNamePooling, ADPT_DESC(Pooling))//构造指向Pooling的指针并储存，创建结构体RegAdptDescPooling

// MaxPoolV3
INPUT_MAP(MaxPoolV3) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入MaxPoolV3对应空间内并用input_map指针保存
ATTR_MAP(MaxPoolV3) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"padding_mode", ATTR_DESC(padding_mode, AnyTraits<std::string>())},
                       {"pad", ATTR_DESC(pads, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
                       {"global", ATTR_DESC(global_pooling, AnyTraits<bool>())},
                       {"ceil_mode", ATTR_DESC(ceil_mode, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                               存入MaxPoolV3对应空间并用attr_map_指针保存
//                                                                               其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                               std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(MaxPoolV3) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入MaxPoolV3对应空间内并用output_map_指针保存
REG_ADPT_DESC(MaxPoolV3, kNameMaxPoolV3, ADPT_DESC(MaxPoolV3))//构造指向MaxPoolV3的指针并储存，创建结构体RegAdptDescMaxPoolV3

// AvgPoolV2
INPUT_MAP(AvgPoolV2) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入AvgPoolV2对应空间内并用input_map指针保存
ATTR_MAP(AvgPoolV2) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"padding_mode", ATTR_DESC(padding_mode, AnyTraits<std::string>())},
                       {"pad", ATTR_DESC(pads, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
                       {"global", ATTR_DESC(global_pooling, AnyTraits<bool>())},
                       {"ceil_mode", ATTR_DESC(ceil_mode, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                               存入AvgPoolV2对应空间并用attr_map_指针保存
//                                                                               其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                               std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(AvgPoolV2) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入AvgPoolV2对应空间内并用output_map_指针保存
REG_ADPT_DESC(AvgPoolV2, kNameAvgPoolV2, ADPT_DESC(AvgPoolV2))//构造指向AvgPoolV2的指针并储存，创建结构体RegAdptDescAvgPoolV2

// GlobalAveragePool
INPUT_MAP(GlobalAveragePool) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入GlobalAveragePool对应空间内并用input_map指针保存
ATTR_MAP(GlobalAveragePool) = EMPTY_ATTR_MAP;//将空变量存入AvgPoolV2对应空间并用attr_map_指针保存
OUTPUT_MAP(GlobalAveragePool) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入GlobalAveragePool对应空间内并用output_map_指针保存
REG_ADPT_DESC(GlobalAveragePool, kNameGlobalAvgPool, ADPT_DESC(GlobalAveragePool))//构造指向GlobalAveragePool的指针并储存，创建结构体RegAdptDescGlobalAveragePool

// Upsample
INPUT_MAP(Upsample) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入Upsample对应空间内并用input_map指针保存
ATTR_MAP(Upsample) = {{"scale", ATTR_DESC(scale, AnyTraits<float>())},
                      {"stride_h", ATTR_DESC(stride_h, AnyTraits<int64_t>())},
                      {"stride_w", ATTR_DESC(stride_w, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                               存入Upsample对应空间并用attr_map_指针保存
//                                                                               其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                               std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(Upsample) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Upsample对应空间内并用output_map_指针保存
REG_ADPT_DESC(Upsample, kNameUpsample, ADPT_DESC(Upsample))//构造指向Upsample的指针并储存，创建结构体RegAdptDescUpsample
}  // namespace mindspore::transform
