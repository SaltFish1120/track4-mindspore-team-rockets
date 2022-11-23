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

#include "transform/graph_ir/op_declare/nonlinear_fuc_ops_declare.h"//按照路径寻找以下文件，导入到本文件

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
// Relu
INPUT_MAP(Relu) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入Relu对应空间内并用input_map指针保存
ATTR_MAP(Relu) = EMPTY_ATTR_MAP;//将空变量存入Relu对应空间并用attr_map_指针保存
OUTPUT_MAP(Relu) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Relu对应空间内并用output_map_指针保存
REG_ADPT_DESC(Relu, prim::kPrimRelu->name(), ADPT_DESC(Relu))//构造指向Relu的指针并储存，创建结构体RegAdptDescRelu

// ReluV2
INPUT_MAP(ReluV2) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入ReluV2对应空间内并用input_map指针保存
ATTR_MAP(ReluV2) = EMPTY_ATTR_MAP;//将空变量存入ReluV2对应空间并用attr_map_指针保存
OUTPUT_MAP(ReluV2) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(mask)}};
//将变量y、mask处理并存入对应OutputDesc结构体的相应变量中，存入ReluV2对应空间内并用output_map_指针保存
REG_ADPT_DESC(ReluV2, kNameReluV2, ADPT_DESC(ReluV2))//构造指向ReluV2的指针并储存，创建结构体RegAdptDescReluV2

// Elu
INPUT_MAP(Elu) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入Elu对应空间内并用input_map指针保存
ATTR_MAP(Elu) = {{"alpha", ATTR_DESC(alpha, AnyTraits<float>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                  存入Elu对应空间并用attr_map_指针保存
//                                                                  其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(Elu) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Elu对应空间内并用output_map_指针保存
REG_ADPT_DESC(Elu, kNameElu, ADPT_DESC(Elu))//构造指向Elu的指针并储存，创建结构体RegAdptDescElu

// EluGrad
INPUT_MAP(EluGrad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(activations)}};
//将变量grads、activations处理并存入对应InputDesc结构体的相应变量中，存入EluGrad对应空间内并用input_map指针保存
ATTR_MAP(EluGrad) = EMPTY_ATTR_MAP;//将空变量存入EluGrad对应空间并用attr_map_指针保存
OUTPUT_MAP(EluGrad) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入EluGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(EluGrad, kNameEluGrad, ADPT_DESC(EluGrad))//构造指向EluGrad的指针并储存，创建结构体RegAdptDescEluGrad

// PRelu
INPUT_MAP(PRelu) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(weight)}};
//将变量x、weight处理并存入对应InputDesc结构体的相应变量中，存入PRelu对应空间内并用input_map指针保存
ATTR_MAP(PRelu) = EMPTY_ATTR_MAP;//将空变量存入PRelu对应空间并用attr_map_指针保存
OUTPUT_MAP(PRelu) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入PRelu对应空间内并用output_map_指针保存
REG_ADPT_DESC(PRelu, kNamePrelu, ADPT_DESC(PRelu))//构造指向PRelu的指针并储存，创建结构体RegAdptDescPRelu

// PReluGrad
INPUT_MAP(PReluGrad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(features)}, {3, INPUT_DESC(weights)}};
//将变量grads、features、weights处理并存入对应InputDesc结构体的相应变量中，存入PReluGrad对应空间内并用input_map指针保存
ATTR_MAP(PReluGrad) = EMPTY_ATTR_MAP;//将空变量存入PReluGrad对应空间并用attr_map_指针保存
OUTPUT_MAP(PReluGrad) = {{0, OUTPUT_DESC(dx)}, {1, OUTPUT_DESC(da)}};
//将变量dx、da处理并存入对应OutputDesc结构体的相应变量中，存入PReluGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(PReluGrad, kNamePreluGrad, ADPT_DESC(PReluGrad))//构造指向PReluGrad的指针并储存，创建结构体RegAdptDescPReluGrad

// Selu
INPUT_MAP(Selu) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入Selu对应空间内并用input_map指针保存
ATTR_MAP(Selu) = EMPTY_ATTR_MAP;//将空变量存入Selu对应空间并用attr_map_指针保存
OUTPUT_MAP(Selu) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Selu对应空间内并用output_map_指针保存
REG_ADPT_DESC(Selu, kNameSelu, ADPT_DESC(Selu))//构造指向Selu的指针并储存，创建结构体RegAdptDescSelu

// Sigmoid
INPUT_MAP(Sigmoid) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入Sigmoid对应空间内并用input_map指针保存
ATTR_MAP(Sigmoid) = EMPTY_ATTR_MAP;//将空变量存入Sigmoid对应空间并用attr_map_指针保存
OUTPUT_MAP(Sigmoid) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Sigmoid对应空间内并用output_map_指针保存
REG_ADPT_DESC(Sigmoid, kNameSigmoid, ADPT_DESC(Sigmoid))//构造指向Sigmoid的指针并储存，创建结构体RegAdptDescSigmoid

// SigmoidGrad
INPUT_MAP(SigmoidGrad) = {{1, INPUT_DESC(y)}, {2, INPUT_DESC(dy)}};
//将变量y、dy处理并存入对应InputDesc结构体的相应变量中，存入SigmoidGrad对应空间内并用input_map指针保存
ATTR_MAP(SigmoidGrad) = EMPTY_ATTR_MAP;//将空变量存入SigmoidGrad对应空间并用attr_map_指针保存
OUTPUT_MAP(SigmoidGrad) = {{0, OUTPUT_DESC(z)}};//将变量z处理并存入对应OutputDesc结构体的相应变量中，存入SigmoidGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(SigmoidGrad, kNameSigmoidGrad, ADPT_DESC(SigmoidGrad))//构造指向SigmoidGrad的指针并储存，创建结构体RegAdptDescSigmoidGrad

// HardSwish
INPUT_MAP(HardSwish) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入HardSwish对应空间内并用input_map指针保存
ATTR_MAP(HardSwish) = EMPTY_ATTR_MAP;//将空变量存入HardSwish对应空间并用attr_map_指针保存
OUTPUT_MAP(HardSwish) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入HardSwish对应空间内并用output_map_指针保存
REG_ADPT_DESC(HardSwish, kNameHSwish, ADPT_DESC(HardSwish))//构造指向HardSwish的指针并储存，创建结构体RegAdptDescHardSwish

// HardSwishGrad
INPUT_MAP(HardSwishGrad) = {{1, INPUT_DESC(grad)}, {2, INPUT_DESC(x)}};
//将变量grad、x处理并存入对应InputDesc结构体的相应变量中，存入HardSwishGrad对应空间内并用input_map指针保存
ATTR_MAP(HardSwishGrad) = EMPTY_ATTR_MAP;//将空变量存入HardSwishGrad对应空间并用attr_map_指针保存
OUTPUT_MAP(HardSwishGrad) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入HardSwishGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(HardSwishGrad, kNameHSwishGrad, ADPT_DESC(HardSwishGrad))//构造指向HardSwishGrad的指针并储存，创建结构体RegAdptDescHardSwishGrad

// HSigmoid
INPUT_MAP(HardSigmoid) = {{1, INPUT_DESC(input_x)}};//将变量input_x处理并存入对应InputDesc结构体的相应变量中，存入HardSigmoid对应空间内并用input_map指针保存
ATTR_MAP(HardSigmoid) = {{"alpha", ATTR_DESC(alpha, AnyTraits<float>())},
                         {"beta", ATTR_DESC(beta, AnyTraits<float>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                        存入HardSigmoid对应空间并用attr_map_指针保存
//                                                                        其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(HardSigmoid) = {{0, OUTPUT_DESC(output_y)}};//将变量output_y处理并存入对应OutputDesc结构体的相应变量中，存入HardSigmoid对应空间内并用output_map_指针保存
REG_ADPT_DESC(HardSigmoid, kNameHSigmoid, ADPT_DESC(HardSigmoid))//构造指向HardSigmoid的指针并储存，创建结构体RegAdptDescHardSigmoid

// Relu6
INPUT_MAP(Relu6) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入Relu6对应空间内并用input_map指针保存
ATTR_MAP(Relu6) = EMPTY_ATTR_MAP;//将空变量存入Relu6对应空间并用attr_map_指针保存
OUTPUT_MAP(Relu6) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Relu6对应空间内并用output_map_指针保存
REG_ADPT_DESC(Relu6, kNameReLU6, ADPT_DESC(Relu6))//构造指向Relu6的指针并储存，创建结构体RegAdptDescRelu6

// Relu6Grad
INPUT_MAP(Relu6Grad) = {{1, INPUT_DESC(gradients)}, {2, INPUT_DESC(features)}};
//将变量gradients、features处理并存入对应InputDesc结构体的相应变量中，存入Relu6Grad对应空间内并用input_map指针保存
ATTR_MAP(Relu6Grad) = EMPTY_ATTR_MAP;//将空变量存入Relu6Grad对应空间并用attr_map_指针保存
OUTPUT_MAP(Relu6Grad) = {{0, OUTPUT_DESC(backprops)}};//将变量backprops处理并存入对应OutputDesc结构体的相应变量中，存入Relu6Grad对应空间内并用output_map_指针保存
REG_ADPT_DESC(Relu6Grad, kNameReLU6Grad, ADPT_DESC(Relu6Grad))//构造指向Relu6Grad的指针并储存，创建结构体RegAdptDescRelu6Grad

// Softsign
INPUT_MAP(Softsign) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入Softsign对应空间内并用input_map指针保存
ATTR_MAP(Softsign) = EMPTY_ATTR_MAP;//将空变量存入Softsign对应空间并用attr_map_指针保存
OUTPUT_MAP(Softsign) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Softsign对应空间内并用output_map_指针保存
REG_ADPT_DESC(Softsign, kNameSoftsign, ADPT_DESC(Softsign))//构造指向Softsign的指针并储存，创建结构体RegAdptDescSoftsign

// Softplus
INPUT_MAP(Softplus) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入Softplus对应空间内并用input_map指针保存
ATTR_MAP(Softplus) = EMPTY_ATTR_MAP;//将空变量存入Softplus对应空间并用attr_map_指针保存
OUTPUT_MAP(Softplus) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Softplus对应空间内并用output_map_指针保存
REG_ADPT_DESC(Softplus, kNameSoftplus, ADPT_DESC(Softplus))//构造指向Softplus的指针并储存，创建结构体RegAdptDescSoftplus

// SoftplusGrad
INPUT_MAP(SoftplusGrad) = {{1, INPUT_DESC(gradients)}, {2, INPUT_DESC(features)}};
//将变量gradients、features处理并存入对应InputDesc结构体的相应变量中，存入SoftplusGrad对应空间内并用input_map指针保存
ATTR_MAP(SoftplusGrad) = EMPTY_ATTR_MAP;//将空变量存入SoftplusGrad对应空间并用attr_map_指针保存
OUTPUT_MAP(SoftplusGrad) = {{0, OUTPUT_DESC(backprops)}};
//将变量backprops处理并存入对应OutputDesc结构体的相应变量中，存入SoftplusGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(SoftplusGrad, kNameSoftplusGrad, ADPT_DESC(SoftplusGrad))//构造指向SoftplusGrad的指针并储存，创建结构体RegAdptDescSoftplusGrad

// ReluGrad
INPUT_MAP(ReluGrad) = {{1, INPUT_DESC(gradients)}, {2, INPUT_DESC(features)}};
//将变量gradients、features处理并存入对应InputDesc结构体的相应变量中，存入ReluGrad对应空间内并用input_map指针保存
ATTR_MAP(ReluGrad) = EMPTY_ATTR_MAP;//将空变量存入ReluGrad对应空间并用attr_map_指针保存
OUTPUT_MAP(ReluGrad) = {{0, OUTPUT_DESC(backprops)}};//将变量backprops处理并存入对应OutputDesc结构体的相应变量中，存入ReluGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(ReluGrad, prim::kPrimReluGrad->name(), ADPT_DESC(ReluGrad))//构造指向ReluGrad的指针并储存，创建结构体RegAdptDescReluGrad

// ReluGradV2
INPUT_MAP(ReluGradV2) = {{1, INPUT_DESC(gradients)}, {2, INPUT_DESC(mask)}};
//将变量gradients、mask处理并存入对应InputDesc结构体的相应变量中，存入ReluGradV2对应空间内并用input_map指针保存
ATTR_MAP(ReluGradV2) = EMPTY_ATTR_MAP;//将空变量存入ReluGradV2对应空间并用attr_map_指针保存
OUTPUT_MAP(ReluGradV2) = {{0, OUTPUT_DESC(backprops)}};//将变量backprops处理并存入对应OutputDesc结构体的相应变量中，存入ReluGradV2对应空间内并用output_map_指针保存
REG_ADPT_DESC(ReluGradV2, kNameReluGradV2, ADPT_DESC(ReluGradV2))//构造指向ReluGradV2的指针并储存，创建结构体RegAdptDescReluGradV2

// Tanh
INPUT_MAP(Tanh) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入Tanh对应空间内并用input_map指针保存
ATTR_MAP(Tanh) = EMPTY_ATTR_MAP;//将空变量存入ReluGradV2对应空间并用attr_map_指针保存
OUTPUT_MAP(Tanh) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Tanh对应空间内并用output_map_指针保存
REG_ADPT_DESC(Tanh, prim::kPrimTanh->name(), ADPT_DESC(Tanh))//构造指向Tanh的指针并储存，创建结构体RegAdptDescTanh

// TanhGrad
INPUT_MAP(TanhGrad) = {{1, INPUT_DESC(y)}, {2, INPUT_DESC(dy)}};
//将变量y、dy处理并存入对应InputDesc结构体的相应变量中，存入TanhGrad对应空间内并用input_map指针保存
ATTR_MAP(TanhGrad) = EMPTY_ATTR_MAP;//将空变量存入ReluGradV2对应空间并用attr_map_指针保存
OUTPUT_MAP(TanhGrad) = {{0, OUTPUT_DESC(z)}};//将变量z处理并存入对应OutputDesc结构体的相应变量中，存入TanhGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(TanhGrad, prim::kPrimTanhGrad->name(), ADPT_DESC(TanhGrad))//构造指向TanhGrad的指针并储存，创建结构体RegAdptDescTanhGrad

// Mish
INPUT_MAP(Mish) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入Mish对应空间内并用input_map指针保存
ATTR_MAP(Mish) = EMPTY_ATTR_MAP;//将空变量存入Mish对应空间并用attr_map_指针保存
OUTPUT_MAP(Mish) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Mish对应空间内并用output_map_指针保存
REG_ADPT_DESC(Mish, kNameMish, ADPT_DESC(Mish))//构造指向Mish的指针并储存，创建结构体RegAdptDescMish

// GeLU
INPUT_MAP(Gelu) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入Gelu对应空间内并用input_map指针保存
ATTR_MAP(Gelu) = EMPTY_ATTR_MAP;//将空变量存入Gelu对应空间并用attr_map_指针保存
OUTPUT_MAP(Gelu) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Gelu对应空间内并用output_map_指针保存
REG_ADPT_DESC(Gelu, prim::kPrimGeLU->name(), ADPT_DESC(Gelu))//构造指向Gelu的指针并储存，创建结构体RegAdptDescGelu

// GeLUGrad
INPUT_MAP(GeluGrad) = {{1, INPUT_DESC(dy)}, {2, INPUT_DESC(x)}, {3, INPUT_DESC(y)}};
//将变量dy、x、y处理并存入对应InputDesc结构体的相应变量中，存入GeluGrad对应空间内并用input_map指针保存
ATTR_MAP(GeluGrad) = EMPTY_ATTR_MAP;//将空变量存入GeluGrad对应空间并用attr_map_指针保存
OUTPUT_MAP(GeluGrad) = {{0, OUTPUT_DESC(z)}};//将变量z处理并存入对应OutputDesc结构体的相应变量中，存入GeluGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(GeluGrad, prim::kPrimGeLUGrad->name(), ADPT_DESC(GeluGrad))//构造指向GeluGrad的指针并储存，创建结构体RegAdptDescGeluGrad

// FastGeLU
INPUT_MAP(FastGelu) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入FastGelu对应空间内并用input_map指针保存
ATTR_MAP(FastGelu) = EMPTY_ATTR_MAP;//将空变量存入FastGelu对应空间并用attr_map_指针保存
OUTPUT_MAP(FastGelu) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入FastGelu对应空间内并用output_map_指针保存
REG_ADPT_DESC(FastGelu, prim::kPrimFastGeLU->name(), ADPT_DESC(FastGelu))//构造指向FastGelu的指针并储存，创建结构体RegAdptDescFastGelu

// FastGeLUGrad
INPUT_MAP(FastGeluGrad) = {{1, INPUT_DESC(dy)}, {2, INPUT_DESC(x)}};
//将变量dy、x处理并存入对应InputDesc结构体的相应变量中，存入FastGeluGrad对应空间内并用input_map指针保存
ATTR_MAP(FastGeluGrad) = EMPTY_ATTR_MAP;//将空变量存入FastGeluGrad对应空间并用attr_map_指针保存
OUTPUT_MAP(FastGeluGrad) = {{0, OUTPUT_DESC(z)}};//将变量z处理并存入对应OutputDesc结构体的相应变量中，存入FastGeluGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(FastGeluGrad, prim::kPrimFastGeLUGrad->name(), ADPT_DESC(FastGeluGrad))
//构造指向FastGeluGrad的指针并储存，创建结构体RegAdptDescFastGeluGrad

// LeakyRelu
INPUT_MAP(LeakyRelu) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入LeakyRelu对应空间内并用input_map指针保存
ATTR_MAP(LeakyRelu) = {{"alpha", ATTR_DESC(negative_slope, AnyTraits<float>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                 存入LeakyRelu对应空间并用attr_map_指针保存
//                                                                                 其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(LeakyRelu) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入LeakyRelu对应空间内并用output_map_指针保存
REG_ADPT_DESC(LeakyRelu, prim::kPrimLeakyRelu->name(), ADPT_DESC(LeakyRelu))//构造指向LeakyRelu的指针并储存，创建结构体RegAdptDescLeakyRelu
}  // namespace mindspore::transform
