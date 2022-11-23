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

#include "transform/graph_ir/op_declare/nn_norm_ops_declare.h"
#include <vector>

namespace mindspore::transform {
// SoftmaxV2
INPUT_MAP(SoftmaxV2) = {{1, INPUT_DESC(x)}};//将SoftmaxV2与标准进行比较，以input_map_为key并构造名为x的INPUT_DESC结构体并与标准比较其容量
ATTR_MAP(SoftmaxV2) = {
  {"axis", ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
};//将SoftmaxV2与标准进行比较，以attr_map_为key并构造储存了name变量axes的ATTR_DESC结构体
//其中AnyTraits<>的作用为将<>内类型进行构建
//std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(SoftmaxV2) = {{0, OUTPUT_DESC(y)}};//将SoftmaxV2与标准进行比较，以output_map_为key，并构造储存了name变量y的OUTPUT_DESC结构体并与标准比较其容量
REG_ADPT_DESC(SoftmaxV2, kNameSoftmax, ADPT_DESC(SoftmaxV2))//在ADPT_DESC结构体的基础上创建名为kNameSoftmax的RED_ADPT_DESC结构体
                                                            //将SoftmaxV2转为字符变量存入结构体的name变量中
                                                            //并与标准比较其容量，返回对应内容
//以下部分根据引用的不同变量对其进行相应操作
// SoftmaxGrad
INPUT_MAP(SoftmaxGrad) = {{1, INPUT_DESC(softmax)}, {2, INPUT_DESC(grad_softmax)}};
OUTPUT_MAP(SoftmaxGrad) = {{0, OUTPUT_DESC(grad_x)}};
ATTR_MAP(SoftmaxGrad) = EMPTY_ATTR_MAP;
REG_ADPT_DESC(SoftmaxGrad, kNameSoftmaxGrad, ADPT_DESC(SoftmaxGrad))

// SoftmaxCrossEntropyWithLogits
INPUT_MAP(SoftmaxCrossEntropyWithLogits) = {{1, INPUT_DESC(features)}, {2, INPUT_DESC(labels)}};
ATTR_MAP(SoftmaxCrossEntropyWithLogits) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SoftmaxCrossEntropyWithLogits) = {{0, OUTPUT_DESC(loss)}, {1, OUTPUT_DESC(backprop)}};
REG_ADPT_DESC(SoftmaxCrossEntropyWithLogits, prim::kPrimSoftmaxCrossEntropyWithLogits->name(),
              ADPT_DESC(SoftmaxCrossEntropyWithLogits))

// SmoothL1Loss
INPUT_MAP(SmoothL1Loss) = {{1, INPUT_DESC(predict)}, {2, INPUT_DESC(label)}};
ATTR_MAP(SmoothL1Loss) = {{"beta", ATTR_DESC(sigma, AnyTraits<float>())}};
OUTPUT_MAP(SmoothL1Loss) = {{0, OUTPUT_DESC(loss)}};
REG_ADPT_DESC(SmoothL1Loss, kNameSmoothL1Loss, ADPT_DESC(SmoothL1Loss))

// SmoothL1LossGrad
INPUT_MAP(SmoothL1LossGrad) = {{1, INPUT_DESC(predict)}, {2, INPUT_DESC(label)}, {3, INPUT_DESC(dout)}};
ATTR_MAP(SmoothL1LossGrad) = {{"beta", ATTR_DESC(sigma, AnyTraits<float>())}};
OUTPUT_MAP(SmoothL1LossGrad) = {{0, OUTPUT_DESC(gradient)}};
REG_ADPT_DESC(SmoothL1LossGrad, kNameSmoothL1LossGrad, ADPT_DESC(SmoothL1LossGrad))

// SigmoidCrossEntropyWithLogits
INPUT_MAP(SigmoidCrossEntropyWithLogits) = {{1, INPUT_DESC(predict)}, {2, INPUT_DESC(target)}};
ATTR_MAP(SigmoidCrossEntropyWithLogits) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SigmoidCrossEntropyWithLogits) = {{0, OUTPUT_DESC(loss)}};
REG_ADPT_DESC(SigmoidCrossEntropyWithLogits, kNameSigmoidCrossEntropyWithLogits,
              ADPT_DESC(SigmoidCrossEntropyWithLogits))

// SigmoidCrossEntropyWithLogitsGrad
INPUT_MAP(SigmoidCrossEntropyWithLogitsGrad) = {
  {1, INPUT_DESC(predict)}, {2, INPUT_DESC(target)}, {3, INPUT_DESC(dout)}};
ATTR_MAP(SigmoidCrossEntropyWithLogitsGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SigmoidCrossEntropyWithLogitsGrad) = {{0, OUTPUT_DESC(gradient)}};
REG_ADPT_DESC(SigmoidCrossEntropyWithLogitsGrad, kNameSigmoidCrossEntropyWithLogitsGrad,
              ADPT_DESC(SigmoidCrossEntropyWithLogitsGrad))

// SigmoidCrossEntropyWithLogitsV2
INPUT_MAP(SigmoidCrossEntropyWithLogitsV2) = {
  {1, INPUT_DESC(predict)}, {2, INPUT_DESC(target)}, {3, INPUT_DESC(weight)}, {4, INPUT_DESC(pos_weight)}};
ATTR_MAP(SigmoidCrossEntropyWithLogitsV2) = {{"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())}};
OUTPUT_MAP(SigmoidCrossEntropyWithLogitsV2) = {{0, OUTPUT_DESC(loss)}};
REG_ADPT_DESC(SigmoidCrossEntropyWithLogitsV2, kNameSigmoidCrossEntropyWithLogitsV2,
              ADPT_DESC(SigmoidCrossEntropyWithLogitsV2))

// LogSoftmaxGrad
INPUT_MAP(LogSoftmaxGrad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(grad)}};
ATTR_MAP(LogSoftmaxGrad) = {
  {"axis", ATTR_DESC(axis, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(LogSoftmaxGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(LogSoftmaxGrad, prim::kPrimLogSoftmaxGrad->name(), ADPT_DESC(LogSoftmaxGrad))

// LogSoftmaxV2
INPUT_MAP(LogSoftmaxV2) = {{1, INPUT_DESC(logits)}};
ATTR_MAP(LogSoftmaxV2) = {
  {"axis", ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(LogSoftmaxV2) = {{0, OUTPUT_DESC(logsoftmax)}};
REG_ADPT_DESC(LogSoftmaxV2, prim::kPrimLogSoftmax->name(), ADPT_DESC(LogSoftmaxV2))

// LayerNorm
INPUT_MAP(LayerNorm) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(gamma)}, {3, INPUT_DESC(beta)}};
ATTR_MAP(LayerNorm) = {{"begin_norm_axis", ATTR_DESC(begin_norm_axis, AnyTraits<int64_t>())},
                       {"begin_params_axis", ATTR_DESC(begin_params_axis, AnyTraits<int64_t>())},
                       {"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())}};//将begin_norm_axis变量处理并存入结构体对应的变量中，转为字符串并存至name变量中，
                                                                            //收录begin_+norm_axis变量和结构体，并与标准进行比较，进行空间调整并用attr_map_为key储存相应内容
OUTPUT_MAP(LayerNorm) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(mean)}, {2, OUTPUT_DESC(variance)}};
REG_ADPT_DESC(LayerNorm, prim::kPrimLayerNorm->name(), ADPT_DESC(LayerNorm))

// LayerNormGrad
INPUT_MAP(LayerNormGrad) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(dy)}, {3, INPUT_DESC(variance)}, {4, INPUT_DESC(mean)}, {5, INPUT_DESC(gamma)}};
ATTR_MAP(LayerNormGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(LayerNormGrad) = {{0, OUTPUT_DESC(pd_x)}, {1, OUTPUT_DESC(pd_gamma)}, {2, OUTPUT_DESC(pd_beta)}};
REG_ADPT_DESC(LayerNormGrad, prim::kPrimLayerNormGrad->name(), ADPT_DESC(LayerNormGrad))

// LRN
INPUT_MAP(LRN) = {{1, INPUT_DESC(x)}};
ATTR_MAP(LRN) = {{"depth_radius", ATTR_DESC(depth_radius, AnyTraits<int64_t>())},
                 {"bias", ATTR_DESC(bias, AnyTraits<float>())},
                 {"alpha", ATTR_DESC(alpha, AnyTraits<float>())},
                 {"beta", ATTR_DESC(beta, AnyTraits<float>())},
                 {"norm_region", ATTR_DESC(norm_region, AnyTraits<string>())}};
OUTPUT_MAP(LRN) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(LRN, kNameLRN, ADPT_DESC(LRN))

// LRNGrad
INPUT_MAP(LRNGrad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(x)}, {3, INPUT_DESC(y)}};
ATTR_MAP(LRNGrad) = {{"depth_radius", ATTR_DESC(depth_radius, AnyTraits<int64_t>())},
                     {"bias", ATTR_DESC(bias, AnyTraits<float>())},
                     {"alpha", ATTR_DESC(alpha, AnyTraits<float>())},
                     {"beta", ATTR_DESC(beta, AnyTraits<float>())}};
OUTPUT_MAP(LRNGrad) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(LRNGrad, kNameLRNGrad, ADPT_DESC(LRNGrad))

// DropoutDoMask
INPUT_MAP(DropOutDoMask) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(mask)}, {3, INPUT_DESC(keep_prob)}};
ATTR_MAP(DropOutDoMask) = EMPTY_ATTR_MAP;
OUTPUT_MAP(DropOutDoMask) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DropOutDoMask, kNameDropoutDoMask, ADPT_DESC(DropOutDoMask))

// BinaryCrossEntropy
INPUT_MAP(BinaryCrossEntropy) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(y)}, {3, INPUT_DESC(weight)}};
ATTR_MAP(BinaryCrossEntropy) = {{"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())}};
OUTPUT_MAP(BinaryCrossEntropy) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(BinaryCrossEntropy, kNameBinaryCrossEntropy, ADPT_DESC(BinaryCrossEntropy))

// BinaryCrossEntropyGrad
INPUT_MAP(BinaryCrossEntropyGrad) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(y)}, {3, INPUT_DESC(grad_output)}, {4, INPUT_DESC(weight)}};
ATTR_MAP(BinaryCrossEntropyGrad) = {{"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())}};
OUTPUT_MAP(BinaryCrossEntropyGrad) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(BinaryCrossEntropyGrad, kNameBinaryCrossEntropyGrad, ADPT_DESC(BinaryCrossEntropyGrad))

// Centralization
INPUT_MAP(Centralization) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Centralization) = {{"axes", ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(Centralization) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Centralization, kNameCentralization, ADPT_DESC(Centralization))

// Scale
INPUT_MAP(Scale) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(scale)}, {3, INPUT_DESC(bias)}};
ATTR_MAP(Scale) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>())},
                   {"num_axes", ATTR_DESC(num_axes, AnyTraits<int64_t>())},
                   {"scale_from_blob", ATTR_DESC(scale_from_blob, AnyTraits<bool>())}};

OUTPUT_MAP(Scale) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Scale, kNameScale, ADPT_DESC(Scale))
}  // namespace mindspore::transform
