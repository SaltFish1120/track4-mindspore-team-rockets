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

#include "transform/graph_ir/op_declare/nn_training_ops_declare.h"//按照路径寻找以下文件，导入到本文件

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
// ApplyMomentum
INPUT_MAP(ApplyMomentum) = {
  {1, INPUT_DESC(var)}, {2, INPUT_DESC(accum)}, {3, INPUT_DESC(lr)}, {4, INPUT_DESC(grad)}, {5, INPUT_DESC(momentum)}};
//将变量var、accum、lr、grad、momentum处理并存入对应InputDesc结构体的相应变量中，存入ApplyMomentum对应空间内并用input_map指针保存
ATTR_MAP(ApplyMomentum) = {{"use_nesterov", ATTR_DESC(use_nesterov, AnyTraits<bool>())},
                           {"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                      存入ApplyMomentum对应空间并用attr_map_指针保存
//                                                                                      其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ApplyMomentum) = {{0, OUTPUT_DESC(var)}};//将变量var处理并存入对应OutputDesc结构体的相应变量中，存入ApplyMomentum对应空间内并用output_map_指针保存
REG_ADPT_DESC(ApplyMomentum, kNameApplyMomentum, ADPT_DESC(ApplyMomentum))
//构造指向ApplyMomentum的指针并储存，创建结构体RegAdptDescApplyMomentum

// LarsV2Update
INPUT_MAP(LarsV2Update) = {{1, INPUT_DESC(w)},
                           {2, INPUT_DESC(g)},
                           {3, INPUT_DESC(w_square_sum)},
                           {4, INPUT_DESC(g_square_sum)},
                           {5, INPUT_DESC(weight_decay)},
                           {6, INPUT_DESC(learning_rate)}};
//将变量w、g、w_square_sum、g_square_sum、weight_decay、learning_rate处理并存入对应InputDesc结构体的相应变量中，存入arsV2Update对应空间内并用input_map指针保存
ATTR_MAP(LarsV2Update) = {{"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())},
                          {"hyperpara", ATTR_DESC(hyperpara, AnyTraits<float>())},
                          {"use_clip", ATTR_DESC(use_clip, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                存入LarsV2Update对应空间并用attr_map_指针保存
//                                                                                其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(LarsV2Update) = {{0, OUTPUT_DESC(g_new)}};//将变量g_new处理并存入对应OutputDesc结构体的相应变量中，存入LarsV2Update对应空间内并用output_map_指针保存
REG_ADPT_DESC(LarsV2Update, kNameLARSUpdate, ADPT_DESC(LarsV2Update))//构造指向LarsV2Update的指针并储存，创建结构体RegAdptDescLarsV2Update

// ApplyAdam
INPUT_MAP(ApplyAdam) = {{1, INPUT_DESC(var)},         {2, INPUT_DESC(m)},           {3, INPUT_DESC(v)},
                        {4, INPUT_DESC(beta1_power)}, {5, INPUT_DESC(beta2_power)}, {6, INPUT_DESC(lr)},
                        {7, INPUT_DESC(beta1)},       {8, INPUT_DESC(beta2)},       {9, INPUT_DESC(epsilon)},
                        {10, INPUT_DESC(grad)}};
//将变量var、m、v、beta1_power、beta2_power、lr、beta1、beta2、epsilon、grad处理并存入对应InputDesc结构体的相应变量中，存入ApplyAdam对应空间内并用input_map指针保存
ATTR_MAP(ApplyAdam) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())},
                       {"use_nesterov", ATTR_DESC(use_nesterov, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                     存入ApplyAdam对应空间并用attr_map_指针保存
//                                                                                    其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ApplyAdam) = {{0, OUTPUT_DESC(var)}};//将变量var处理并存入对应OutputDesc结构体的相应变量中，存入ApplyAdam对应空间内并用output_map_指针保存

// ApplyAdamD
INPUT_MAP(ApplyAdamD) = {{1, INPUT_DESC(var)},         {2, INPUT_DESC(m)},           {3, INPUT_DESC(v)},
                         {4, INPUT_DESC(beta1_power)}, {5, INPUT_DESC(beta2_power)}, {6, INPUT_DESC(lr)},
                         {7, INPUT_DESC(beta1)},       {8, INPUT_DESC(beta2)},       {9, INPUT_DESC(epsilon)},
                         {10, INPUT_DESC(grad)}};
//将变量var、m、v、beta1_power、beta2_power、lr、beta1、beta2、epsilon、grad处理并存入对应InputDesc结构体的相应变量中，存入ApplyAdamD对应空间内并用input_map指针保存
ATTR_MAP(ApplyAdamD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())},
                        {"use_nesterov", ATTR_DESC(use_nesterov, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                      存入ApplyAdamD对应空间并用attr_map_指针保存
//                                                                                      其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ApplyAdamD) = {{0, OUTPUT_DESC(var)}, {1, OUTPUT_DESC(m)}, {2, OUTPUT_DESC(v)}};
//将变量var、m、v处理并存入对应OutputDesc结构体的相应变量中，存入ApplyAdamD对应空间内并用output_map_指针保存
REG_ADPT_DESC(ApplyAdamD, kNameApplyAdam, ADPT_DESC(ApplyAdamD))//构造指向ApplyAdamD的指针并储存，创建结构体RegAdptDescApplyAdamD
REG_ADPT_DESC(ApplyAdam, kNameApplyAdam, ADPT_DESC(ApplyAdam))//构造指向ApplyAdam的指针并储存，创建结构体RegAdptDescApplyAdam

// ApplyAdagradD
INPUT_MAP(ApplyAdagradD) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(accum)}, {3, INPUT_DESC(lr)}, {4, INPUT_DESC(grad)}};
//将变量var、accum、lr、grad处理并存入对应InputDesc结构体的相应变量中，存入ApplyAdagradD对应空间内并用input_map指针保存
ATTR_MAP(ApplyAdagradD) = {{"update_slots", ATTR_DESC(update_slots, AnyTraits<bool>())},
                           {"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                       存入ApplyAdagradD对应空间并用attr_map_指针保存
//                                                                                       其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ApplyAdagradD) = {{0, OUTPUT_DESC(var)}, {1, OUTPUT_DESC(accum)}};
//将变量var、accum处理并存入对应OutputDesc结构体的相应变量中，存入ApplyAdagradD对应空间内并用output_map_指针保存
REG_ADPT_DESC(ApplyAdagradD, kNameApplyAdagrad, ADPT_DESC(ApplyAdagradD))//构造指向ApplyAdagradD的指针并储存，创建结构体RegAdptDescApplyAdagradD

// ApplyAdagradV2D
INPUT_MAP(ApplyAdagradV2D) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(accum)}, {3, INPUT_DESC(lr)}, {4, INPUT_DESC(grad)}};
//将变量var、accum、lr、grad处理并存入对应InputDesc结构体的相应变量中，存入ApplyAdagradV2D对应空间内并用input_map指针保存
ATTR_MAP(ApplyAdagradV2D) = {{"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())},
                             {"update_slots", ATTR_DESC(update_slots, AnyTraits<bool>())},
                             {"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                         存入ApplyAdagradV2D对应空间并用attr_map_指针保存
//                                                                                         其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ApplyAdagradV2D) = {{0, OUTPUT_DESC(var)}, {1, OUTPUT_DESC(accum)}};
//将变量var、accum处理并存入对应OutputDesc结构体的相应变量中，存入ApplyAdagradV2D对应空间内并用output_map_指针保存
REG_ADPT_DESC(ApplyAdagradV2D, kNameApplyAdagradV2D, ADPT_DESC(ApplyAdagradV2D))//构造指向ApplyAdagradV2D的指针并储存，创建结构体RegAdptDescApplyAdagradV2D

// ApplyAddSignD
INPUT_MAP(ApplyAddSignD) = {{1, INPUT_DESC(var)},   {2, INPUT_DESC(m)},          {3, INPUT_DESC(lr)},
                            {4, INPUT_DESC(alpha)}, {5, INPUT_DESC(sign_decay)}, {6, INPUT_DESC(beta)},
                            {7, INPUT_DESC(grad)}};
//将变量var、m、lr、alpha、sign_decay、beta、grad处理并存入对应InputDesc结构体的相应变量中，存入ApplyAddSignD对应空间内并用input_map指针保存
ATTR_MAP(ApplyAddSignD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                       存入ApplyAddSignD对应空间并用attr_map_指针保存
//                                                                                       其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ApplyAddSignD) = {{0, OUTPUT_DESC(var)}, {1, OUTPUT_DESC(m)}};
//将变量var、m处理并存入对应OutputDesc结构体的相应变量中，存入ApplyAddSignD对应空间内并用output_map_指针保存
REG_ADPT_DESC(ApplyAddSignD, kNameApplyAddSignD, ADPT_DESC(ApplyAddSignD))//构造指向ApplyAddSignD的指针并储存，创建结构体RegAdptDescApplyAddSignD

// SparseApplyAdagradV2D
INPUT_MAP(SparseApplyAdagradV2D) = {
  {1, INPUT_DESC(var)}, {2, INPUT_DESC(accum)}, {3, INPUT_DESC(grad)}, {4, INPUT_DESC(indices)}};
//将变量var、accum、grad、indices处理并存入对应InputDesc结构体的相应变量中，存入SparseApplyAdagradV2D对应空间内并用input_map指针保存
ATTR_MAP(SparseApplyAdagradV2D) = {{"lr", ATTR_DESC(lr, AnyTraits<float>())},
                                   {"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())},
                                   {"update_slots", ATTR_DESC(update_slots, AnyTraits<bool>())},
                                   {"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                              存入SparseApplyAdagradV2D对应空间并用attr_map_指针保存
//                                                                                              其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(SparseApplyAdagradV2D) = {{0, OUTPUT_DESC(var)}, {1, OUTPUT_DESC(accum)}};
//将变量var、accum处理并存入对应OutputDesc结构体的相应变量中，存入SparseApplyAdagradV2D对应空间内并用output_map_指针保存
REG_ADPT_DESC(SparseApplyAdagradV2D, kNameSparseApplyAdagradV2D, ADPT_DESC(SparseApplyAdagradV2D))
//构造指向SparseApplyAdagradV2D的指针并储存，创建结构体RegAdptDescSparseApplyAdagradV2D

// DataFormatDimMap
INPUT_MAP(DataFormatDimMap) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入DataFormatDimMap对应空间内并用input_map指针保存
ATTR_MAP(DataFormatDimMap) = {{"src_format", ATTR_DESC(src_format, AnyTraits<std::string>())},
                              {"dst_format", ATTR_DESC(dst_format, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                              存入DataFormatDimMap对应空间并用attr_map_指针保存
//                                                                                              其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(DataFormatDimMap) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入DataFormatDimMap对应空间内并用output_map_指针保存
REG_ADPT_DESC(DataFormatDimMap, kNameDataFormatDimMap, ADPT_DESC(DataFormatDimMap))
//构造指向DataFormatDimMap的指针并储存，创建结构体RegAdptDescDataFormatDimMap

// ApplyAdadeltaD
INPUT_MAP(ApplyAdadeltaD) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(accum)}, {3, INPUT_DESC(accum_update)},
                             {4, INPUT_DESC(lr)},  {5, INPUT_DESC(rho)},   {6, INPUT_DESC(epsilon)},
                             {7, INPUT_DESC(grad)}};
//将变量var、accum、accum_update、lr、rho、epsilon、grad处理并存入对应InputDesc结构体的相应变量中，存入ApplyAdadeltaD对应空间内并用input_map指针保存
ATTR_MAP(ApplyAdadeltaD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                        存入ApplyAdadeltaD对应空间并用attr_map_指针保存
//                                                                                        其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ApplyAdadeltaD) = {{0, OUTPUT_DESC(var)}, {1, OUTPUT_DESC(accum)}, {2, OUTPUT_DESC(accum_update)}};
//将变量var、accum、accum_update处理并存入对应OutputDesc结构体的相应变量中，存入ApplyAdadeltaD对应空间内并用output_map_指针保存
REG_ADPT_DESC(ApplyAdadeltaD, kNameApplyAdadelta, ADPT_DESC(ApplyAdadeltaD))
//构造指向ApplyAdadeltaD的指针并储存，创建结构体RegAdptDescApplyAdadeltaD

// ApplyAdaMaxD
INPUT_MAP(ApplyAdaMaxD) = {{1, INPUT_DESC(var)},         {2, INPUT_DESC(m)},       {3, INPUT_DESC(v)},
                           {4, INPUT_DESC(beta1_power)}, {5, INPUT_DESC(lr)},      {6, INPUT_DESC(beta1)},
                           {7, INPUT_DESC(beta2)},       {8, INPUT_DESC(epsilon)}, {9, INPUT_DESC(grad)}};
//将变量var、m、v、beta1_power、lr、beta1、beta2、epsilon、grad处理并存入对应InputDesc结构体的相应变量中，存入ApplyAdaMaxD对应空间内并用input_map指针保存
ATTR_MAP(ApplyAdaMaxD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                      存入ApplyAdaMaxD对应空间并用attr_map_指针保存
//                                                                                      其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ApplyAdaMaxD) = {{0, OUTPUT_DESC(var)}, {1, OUTPUT_DESC(m)}, {2, OUTPUT_DESC(v)}};
//将变量var、m、v处理并存入对应OutputDesc结构体的相应变量中，存入ApplyAdaMaxD对应空间内并用output_map_指针保存
REG_ADPT_DESC(ApplyAdaMaxD, kNameApplyAdaMax, ADPT_DESC(ApplyAdaMaxD))
//构造指向ApplyAdaMaxD的指针并储存，创建结构体RegAdptDescApplyAdaMaxD

// ApplyGradientDescent
INPUT_MAP(ApplyGradientDescent) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(alpha)}, {3, INPUT_DESC(delta)}};
//将变量var、alpha、delta处理并存入对应InputDesc结构体的相应变量中，存入ApplyGradientDescent对应空间内并用input_map指针保存
ATTR_MAP(ApplyGradientDescent) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                              存入ApplyGradientDescent对应空间并用attr_map_指针保存
//                                                                                              其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ApplyGradientDescent) = {{0, OUTPUT_DESC(var)}};
//将变量var处理并存入对应OutputDesc结构体的相应变量中，存入ApplyGradientDescent对应空间内并用output_map_指针保存
REG_ADPT_DESC(ApplyGradientDescent, kNameApplyGradientDescent, ADPT_DESC(ApplyGradientDescent))
//构造指向ApplyGradientDescent的指针并储存，创建结构体RegAdptDescApplyGradientDescent

// ApplyPowerSignD
INPUT_MAP(ApplyPowerSignD) = {{1, INPUT_DESC(var)},     {2, INPUT_DESC(m)},          {3, INPUT_DESC(lr)},
                              {4, INPUT_DESC(logbase)}, {5, INPUT_DESC(sign_decay)}, {6, INPUT_DESC(beta)},
                              {7, INPUT_DESC(grad)}};
//将变量var、m、lr、logbase、sign_decay、beta、grad处理并存入对应InputDesc结构体的相应变量中，存入ApplyPowerSignD对应空间内并用input_map指针保存
ATTR_MAP(ApplyPowerSignD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                         存入ApplyPowerSignD对应空间并用attr_map_指针保存
//                                                                                          其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ApplyGradientDescent) = {{0, OUTPUT_DESC(var)}};
OUTPUT_MAP(ApplyPowerSignD) = {{0, OUTPUT_DESC(var)}, {1, OUTPUT_DESC(m)}};
//将变量var、m处理并存入对应OutputDesc结构体的相应变量中，存入ApplyPowerSignDt对应空间内并用output_map_指针保存
REG_ADPT_DESC(ApplyPowerSignD, kNameApplyPowerSign, ADPT_DESC(ApplyPowerSignD))
//构造指向ApplyPowerSignD的指针并储存，创建结构体RegAdptDescApplyPowerSignD

// ApplyProximalGradientDescent
INPUT_MAP(ApplyProximalGradientDescent) = {
  {1, INPUT_DESC(var)}, {2, INPUT_DESC(alpha)}, {3, INPUT_DESC(l1)}, {4, INPUT_DESC(l2)}, {5, INPUT_DESC(delta)}};
//将变量var、alpha、l1、l2、delta处理并存入对应InputDesc结构体的相应变量中，存入ApplyProximalGradientDescent对应空间内并用input_map指针保存
ATTR_MAP(ApplyProximalGradientDescent) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                                存入ApplyProximalGradientDescent对应空间并用attr_map_指针保存
//                                                                                                      其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ApplyProximalGradientDescent) = {{0, OUTPUT_DESC(var)}};
//将变量var处理并存入对应OutputDesc结构体的相应变量中，存入ApplyProximalGradientDescent对应空间内并用output_map_指针保存
REG_ADPT_DESC(ApplyProximalGradientDescent, kNameApplyProximalGradientDescent, ADPT_DESC(ApplyProximalGradientDescent))
//构造指向ApplyProximalGradientDescent的指针并储存，创建结构体RegAdptDescApplyProximalGradientDescent

// SGD
INPUT_MAP(SGD) = {{1, INPUT_DESC(parameters)}, {2, INPUT_DESC(gradient)}, {3, INPUT_DESC(learning_rate)},
                  {4, INPUT_DESC(accum)},      {5, INPUT_DESC(momentum)}, {6, INPUT_DESC(stat)}};
//将变量parameters、gradient、learning_rate、accum、momentum、stat处理并存入对应InputDesc结构体的相应变量中，存入SGD对应空间内并用input_map指针保存
ATTR_MAP(SGD) = {{"dampening", ATTR_DESC(dampening, AnyTraits<float>())},
                 {"weight_decay", ATTR_DESC(weight_decay, AnyTraits<float>())},
                 {"nesterov", ATTR_DESC(nesterov, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                       存入SGD对应空间并用attr_map_指针保存
//                                                                       其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(SGD) = {{0, OUTPUT_DESC(parameters)}};//将变量parameters处理并存入对应OutputDesc结构体的相应变量中，存入SGD对应空间内并用output_map_指针保存
REG_ADPT_DESC(SGD, kNameSGD, ADPT_DESC(SGD))//构造指向SGD的指针并储存，创建结构体RegAdptDescSGD

// SparseApplyAdagradD
INPUT_MAP(SparseApplyAdagradD) = {
  {1, INPUT_DESC(var)}, {2, INPUT_DESC(accum)}, {3, INPUT_DESC(grad)}, {4, INPUT_DESC(indices)}};
//将变量var、accum、grad、indices处理并存入对应InputDesc结构体的相应变量中，存入SparseApplyAdagradD对应空间内并用input_map指针保存
ATTR_MAP(SparseApplyAdagradD) = {{"lr", ATTR_DESC(lr, AnyTraits<float>())},
                                 {"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                            存入SparseApplyAdagradD对应空间并用attr_map_指针保存
//                                                                                            其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(SparseApplyAdagradD) = {{0, OUTPUT_DESC(var)}};
//将变量var处理并存入对应OutputDesc结构体的相应变量中，存入SparseApplyAdagradD对应空间内并用output_map_指针保存
REG_ADPT_DESC(SparseApplyAdagradD, kNameSparseApplyAdagrad, ADPT_DESC(SparseApplyAdagradD))
//构造指向SparseApplyAdagradD的指针并储存，创建结构体RegAdptDescSparseApplyAdagradD

// ApplyProximalAdagradD
INPUT_MAP(ApplyProximalAdagradD) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(accum)}, {3, INPUT_DESC(lr)},
                                    {4, INPUT_DESC(l1)},  {5, INPUT_DESC(l2)},    {6, INPUT_DESC(grad)}};
//将变量var、accum、lr、l1、l2、grad处理并存入对应InputDesc结构体的相应变量中，存入ApplyProximalAdagradD对应空间内并用input_map指针保存
ATTR_MAP(ApplyProximalAdagradD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                               存入ApplyProximalAdagradD对应空间并用attr_map_指针保存
//                                                                                               其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ApplyProximalAdagradD) = {{0, OUTPUT_DESC(var)}, {1, OUTPUT_DESC(accum)}};
//将变量accum处理并存入对应OutputDesc结构体的相应变量中，存入ApplyProximalAdagradD对应空间内并用output_map_指针保存
REG_ADPT_DESC(ApplyProximalAdagradD, kNameApplyProximalAdagrad, ADPT_DESC(ApplyProximalAdagradD))
//构造指向ApplyProximalAdagradD的指针并储存，创建结构体RegAdptDescApplyProximalAdagradD

// SparseApplyProximalAdagradD
INPUT_MAP(SparseApplyProximalAdagradD) = {{1, INPUT_DESC(var)},    {2, INPUT_DESC(accum)}, {3, INPUT_DESC(lr)},
                                          {4, INPUT_DESC(l1)},     {5, INPUT_DESC(l2)},    {6, INPUT_DESC(grad)},
                                          {7, INPUT_DESC(indices)}};
//将变量var、accum、lr、l1、l2、grad、indices处理并存入对应InputDesc结构体的相应变量中，存入SparseApplyProximalAdagradD对应空间内并用input_map指针保存
ATTR_MAP(SparseApplyProximalAdagradD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                               存入SparseApplyProximalAdagradD对应空间并用attr_map_指针保存
//                                                                                                    其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(SparseApplyProximalAdagradD) = {{0, OUTPUT_DESC(var)}, {1, OUTPUT_DESC(accum)}};
//将变量accum处理并存入对应OutputDesc结构体的相应变量中，存入SparseApplyProximalAdagradD对应空间内并用output_map_指针保存
REG_ADPT_DESC(SparseApplyProximalAdagradD, kNameSparseApplyProximalAdagradD, ADPT_DESC(SparseApplyProximalAdagradD))
//构造指向SparseApplyProximalAdagradD的指针并储存，创建结构体RegAdptDescSparseApplyProximalAdagradD

// SparseApplyFtrlD
INPUT_MAP(SparseApplyFtrlD) = {{1, INPUT_DESC(var)},
                               {2, INPUT_DESC(accum)},
                               {3, INPUT_DESC(linear)},
                               {4, INPUT_DESC(grad)},
                               {5, INPUT_DESC(indices)}};
//将变量var、accum、linear、grad、indices处理并存入对应InputDesc结构体的相应变量中，存入SparseApplyFtrlD对应空间内并用input_map指针保存
ATTR_MAP(SparseApplyFtrlD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())},
                              {"lr", ATTR_DESC(lr, AnyTraits<float>())},
                              {"l1", ATTR_DESC(l1, AnyTraits<float>())},
                              {"l2", ATTR_DESC(l2, AnyTraits<float>())},
                              {"lr_power", ATTR_DESC(lr_power, AnyTraits<float>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                     存入SparseApplyFtrlD对应空间并用attr_map_指针保存
//                                                                                     其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(SparseApplyFtrlD) = {{0, OUTPUT_DESC(var)}};
//将变量var处理并存入对应OutputDesc结构体的相应变量中，存入SparseApplyFtrlD对应空间内并用output_map_指针保存
REG_ADPT_DESC(SparseApplyFtrlD, kNameSparseApplyFtrlD, ADPT_DESC(SparseApplyFtrlD))
//构造指向SparseApplyFtrlD的指针并储存，创建结构体RegAdptDescSparseApplyFtrlD

// SparseApplyFtrlV2D
INPUT_MAP(SparseApplyFtrlV2D) = {{1, INPUT_DESC(var)},
                                 {2, INPUT_DESC(accum)},
                                 {3, INPUT_DESC(linear)},
                                 {4, INPUT_DESC(grad)},
                                 {5, INPUT_DESC(indices)}};
//将变量var、accum、linear、grad、indices处理并存入对应InputDesc结构体的相应变量中，存入SparseApplyFtrlV2D对应空间内并用input_map指针保存
ATTR_MAP(SparseApplyFtrlV2D) = {{"lr", ATTR_DESC(lr, AnyTraits<float>())}, {"l1", ATTR_DESC(l1, AnyTraits<float>())}};
//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入SparseApplyFtrlV2D对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(SparseApplyFtrlV2D) = {{0, OUTPUT_DESC(var)}, {1, OUTPUT_DESC(accum)}, {2, OUTPUT_DESC(linear)}};
//将变量var、accum、linear处理并存入对应OutputDesc结构体的相应变量中，存入SparseApplyFtrlV2D对应空间内并用output_map_指针保存
REG_ADPT_DESC(SparseApplyFtrlV2D, kNameSparseApplyFtrlV2D, ADPT_DESC(SparseApplyFtrlV2D))
//构造指向SparseApplyFtrlV2D的指针并储存，创建结构体RegAdptDescSparseApplyFtrlV2D

// ApplyFtrl
INPUT_MAP(ApplyFtrl) = {{1, INPUT_DESC(var)},  {2, INPUT_DESC(accum)},   {3, INPUT_DESC(linear)},
                        {4, INPUT_DESC(grad)}, {5, INPUT_DESC(lr)},      {6, INPUT_DESC(l1)},
                        {7, INPUT_DESC(l2)},   {8, INPUT_DESC(lr_power)}};
//将变量var、accum、linear、grad、lr、l1、l2、lr_power处理并存入对应InputDesc结构体的相应变量中，存入ApplyFtrl对应空间内并用input_map指针保存
ATTR_MAP(ApplyFtrl) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                   存入ApplyFtrl对应空间并用attr_map_指针保存
//                                                                                   其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ApplyFtrl) = {{0, OUTPUT_DESC(var)}};//将变量var处理并存入对应OutputDesc结构体的相应变量中，存入ApplyFtrl对应空间内并用output_map_指针保存
REG_ADPT_DESC(ApplyFtrl, kNameApplyFtrl, ADPT_DESC(ApplyFtrl))//构造指向ApplyFtrl的指针并储存，创建结构体RegAdptDescApplyFtrl

// ApplyRMSPropD
INPUT_MAP(ApplyRMSPropD) = {
  {1, INPUT_DESC(var)}, {2, INPUT_DESC(ms)}, {3, INPUT_DESC(mom)}, {4, INPUT_DESC(lr)}, {5, INPUT_DESC(grad)}};
//将变量var、ms、mom、lr、grad处理并存入对应InputDesc结构体的相应变量中，存入ApplyRMSPropD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(ApplyRMSPropD) = {{6, ATTR_DESC(rho, AnyTraits<float>())},
                                 {7, ATTR_DESC(momentum, AnyTraits<float>())},
                                 {8, ATTR_DESC(epsilon, AnyTraits<float>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                              存入ApplyRMSPropD对应空间并用attr_map_指针保存
//                                                                              其中AnyTraits<>的作用为将<>内类型进行构建
ATTR_MAP(ApplyRMSPropD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ApplyRMSPropD) = {{0, OUTPUT_DESC(var)}};//将变量var处理并存入对应OutputDesc结构体的相应变量中，存入ApplyRMSPropD对应空间内并用output_map_指针保存
REG_ADPT_DESC(ApplyRMSPropD, kNameApplyRMSProp, ADPT_DESC(ApplyRMSPropD))//构造指向ApplyRMSPropD的指针并储存，创建结构体RegAdptDescApplyRMSPropD

// ApplyCenteredRMSProp
INPUT_MAP(ApplyCenteredRMSProp) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(mg)},       {3, INPUT_DESC(ms)},
                                   {4, INPUT_DESC(mom)}, {5, INPUT_DESC(grad)},     {6, INPUT_DESC(lr)},
                                   {7, INPUT_DESC(rho)}, {8, INPUT_DESC(momentum)}, {9, INPUT_DESC(epsilon)}};
//将变量var、mg、ms、mom、grad、lr、rho、momentum、epsilon处理并存入对应InputDesc结构体的相应变量中，存入ApplyCenteredRMSProp对应空间内并用input_map指针保存
ATTR_MAP(ApplyCenteredRMSProp) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                              存入ApplyCenteredRMSProp对应空间并用attr_map_指针保存
//                                                                                              其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ApplyCenteredRMSProp) = {{0, OUTPUT_DESC(var)}};
//将变量var处理并存入对应OutputDesc结构体的相应变量中，存入ApplyCenteredRMSProp对应空间内并用output_map_指针保存
REG_ADPT_DESC(ApplyCenteredRMSProp, kNameApplyCenteredRMSProp, ADPT_DESC(ApplyCenteredRMSProp))
//构造指向ApplyCenteredRMSProp的指针并储存，创建结构体RegAdptDescApplyCenteredRMSProp
}  // namespace mindspore::transform
