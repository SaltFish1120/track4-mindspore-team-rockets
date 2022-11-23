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

#include "transform/graph_ir/op_declare/rnn_declare.h"//按照路径寻找以下文件，导入到本文件

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
// BasicLSTMCell
INPUT_MAP(BasicLSTMCell) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(h)}, {3, INPUT_DESC(c)}, {4, INPUT_DESC(w)}, {5, INPUT_DESC(b)}};
//将变量x、h、c、w、b处理并存入对应InputDesc结构体的相应变量中，存入BNTrainingReduce对应空间内并用input_map指针保存
ATTR_MAP(BasicLSTMCell) = {{"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())},
                           {"forget_bias", ATTR_DESC(forget_bias, AnyTraits<float>())},
                           {"state_is_tuple", ATTR_DESC(state_is_tuple, AnyTraits<bool>())},
                           {"activation", ATTR_DESC(activation, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                            存入BasicLSTMCell对应空间并用attr_map_指针保存
//                                                                                            其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(BasicLSTMCell) = {{0, OUTPUT_DESC(ct)}, {1, OUTPUT_DESC(ht)}, {2, OUTPUT_DESC(it)},    {3, OUTPUT_DESC(jt)},
                             {4, OUTPUT_DESC(ft)}, {5, OUTPUT_DESC(ot)}, {6, OUTPUT_DESC(tanhct)}};
//将变量ct、ht、it、jt、ft、ot、tanhct处理并存入对应OutputDesc结构体的相应变量中，存入BasicLSTMCell对应空间内并用output_map_指针保存
REG_ADPT_DESC(BasicLSTMCell, kNameBasicLSTMCell, ADPT_DESC(BasicLSTMCell))
//构造指向BasicLSTMCell的指针并储存，创建结构体RegAdptDescBasicLSTMCell

// BasicLSTMCellInputGrad
INPUT_MAP(BasicLSTMCellInputGrad) = {{1, INPUT_DESC(dgate)}, {2, INPUT_DESC(w)}};
//将变量dgate、w处理并存入对应InputDesc结构体的相应变量中，存入BasicLSTMCellInputGrad对应空间内并用input_map指针保存
ATTR_MAP(BasicLSTMCellInputGrad) = {{"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                             存入BasicLSTMCellInputGrad对应空间并用attr_map_指针保存
//                                                                                             其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(BasicLSTMCellInputGrad) = {{0, OUTPUT_DESC(dxt)}, {1, OUTPUT_DESC(dht)}};
//将变量dxt、dht处理并存入对应OutputDesc结构体的相应变量中，存入BasicLSTMCellInputGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(BasicLSTMCellInputGrad, kNameBasicLSTMCellInputGrad, ADPT_DESC(BasicLSTMCellInputGrad))
//构造指向BasicLSTMCellInputGrad的指针并储存，创建结构体RegAdptDescBasicLSTMCellInputGrad

// BasicLSTMCellWeightGrad
INPUT_MAP(BasicLSTMCellWeightGrad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(h)}, {3, INPUT_DESC(dgate)}};
//将变量x、h、dgate处理并存入对应InputDesc结构体的相应变量中，存入BasicLSTMCellWeightGrad对应空间内并用input_map指针保存
ATTR_MAP(BasicLSTMCellWeightGrad) = EMPTY_ATTR_MAP;//将空变量存入BasicLSTMCellWeightGrad对应空间并用attr_map_指针保存
OUTPUT_MAP(BasicLSTMCellWeightGrad) = {{0, OUTPUT_DESC(dw)}, {1, OUTPUT_DESC(db)}};
//将变量dw、db处理并存入对应OutputDesc结构体的相应变量中，存入BasicLSTMCellWeightGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(BasicLSTMCellWeightGrad, kNameBasicLSTMCellWeightGrad, ADPT_DESC(BasicLSTMCellWeightGrad))
//构造指向BasicLSTMCellWeightGrad的指针并储存，创建结构体RegAdptDescBasicLSTMCellWeightGrad

// BasicLSTMCellCStateGrad
INPUT_MAP(BasicLSTMCellCStateGrad) = {{1, INPUT_DESC(c)},  {2, INPUT_DESC(dht)},   {3, INPUT_DESC(dct)},
                                      {4, INPUT_DESC(it)}, {5, INPUT_DESC(jt)},    {6, INPUT_DESC(ft)},
                                      {7, INPUT_DESC(ot)}, {8, INPUT_DESC(tanhct)}};
//将变量c、dht、dct、it、jt、ft、ot、tanhct处理并存入对应InputDesc结构体的相应变量中，存入BasicLSTMCellCStateGrad对应空间内并用input_map指针保存
ATTR_MAP(BasicLSTMCellCStateGrad) = {{"forget_bias", ATTR_DESC(forget_bias, AnyTraits<float>())},
                                     {"activation", ATTR_DESC(activation, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                                      存入BasicLSTMCellCStateGrad对应空间并用attr_map_指针保存
//                                                                                                      其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(BasicLSTMCellCStateGrad) = {{0, OUTPUT_DESC(dgate)}, {1, OUTPUT_DESC(dct_1)}};
//将变量dgate、dct_1处理并存入对应OutputDesc结构体的相应变量中，存入BasicLSTMCellCStateGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(BasicLSTMCellCStateGrad, kNameBasicLSTMCellCStateGrad, ADPT_DESC(BasicLSTMCellCStateGrad))
//构造指向BasicLSTMCellCStateGrad的指针并储存，创建结构体RegAdptDescBasicLSTMCellCStateGrad

// LSTMInputGrad
INPUT_MAP(LSTMInputGrad) = {{1, INPUT_DESC(w)},  {2, INPUT_DESC(init_c)}, {3, INPUT_DESC(c)},      {4, INPUT_DESC(dy)},
                            {5, INPUT_DESC(dh)}, {6, INPUT_DESC(dc)},     {7, INPUT_DESC(i)},      {8, INPUT_DESC(j)},
                            {9, INPUT_DESC(f)},  {10, INPUT_DESC(o)},     {11, INPUT_DESC(tanhct)}};
//将变量w、init_c、c、dy、dh、dc、i、j、f、o、tanhct处理并存入对应InputDesc结构体的相应变量中，存入LSTMInputGrad对应空间内并用input_map指针保存
ATTR_MAP(LSTMInputGrad) = EMPTY_ATTR_MAP;//将空变量存入LSTMInputGrad对应空间并用attr_map_指针保存
OUTPUT_MAP(LSTMInputGrad) = {
  {0, OUTPUT_DESC(dx)}, {1, OUTPUT_DESC(dh_prev)}, {2, OUTPUT_DESC(dc_prev)}, {4, OUTPUT_DESC(dgate)}};
//将变量dx、dh_prev、dc_prev、dgate处理并存入对应OutputDesc结构体的相应变量中，存入LSTMInputGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(LSTMInputGrad, kNameLSTMInputGrad, ADPT_DESC(LSTMInputGrad))
//构造指向LSTMInputGrad的指针并储存，创建结构体RegAdptDescLSTMInputGrad

// DynamicRNN
INPUT_MAP(DynamicRNN) = {{1, INPUT_DESC(x)},          {2, INPUT_DESC(w)},      {3, INPUT_DESC(b)},
                         {4, INPUT_DESC(seq_length)}, {5, INPUT_DESC(init_h)}, {6, INPUT_DESC(init_c)},
                         {7, INPUT_DESC(wci)},        {8, INPUT_DESC(wcf)},    {9, INPUT_DESC(wco)},
                         {10, INPUT_DESC(mask)}};
//将变量x、w、c、b、seq_length、init_h、init_c、wci、wcf、wco、mask处理并存入对应InputDesc结构体的相应变量中，存入DynamicRNN对应空间内并用input_map指针保存
ATTR_MAP(DynamicRNN) = {{"cell_type", ATTR_DESC(cell_type, AnyTraits<std::string>())},
                        {"direction", ATTR_DESC(direction, AnyTraits<std::string>())},
                        {"cell_depth", ATTR_DESC(cell_depth, AnyTraits<int64_t>())},
                        {"use_peephole", ATTR_DESC(use_peephole, AnyTraits<bool>())},
                        {"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())},
                        {"cell_clip", ATTR_DESC(cell_clip, AnyTraits<float>())},
                        {"num_proj", ATTR_DESC(num_proj, AnyTraits<int64_t>())},
                        {"time_major", ATTR_DESC(time_major, AnyTraits<bool>())},
                        {"ivation", ATTR_DESC(activation, AnyTraits<std::string>())},
                        {"forget_bias", ATTR_DESC(forget_bias, AnyTraits<float>())},
                        {"is_training", ATTR_DESC(is_training, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                    存入DynamicRNN对应空间并用attr_map_指针保存
//                                                                                    其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(DynamicRNN) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(output_h)}, {2, OUTPUT_DESC(output_c)},
                          {3, OUTPUT_DESC(i)}, {4, OUTPUT_DESC(j)},        {5, OUTPUT_DESC(f)},
                          {6, OUTPUT_DESC(o)}, {7, OUTPUT_DESC(tanhc)}};
//将变量y、output_h、output_c、i、j、f、o、tanhc处理并存入对应OutputDesc结构体的相应变量中，存入DynamicRNN对应空间内并用output_map_指针保存
REG_ADPT_DESC(DynamicRNN, kNameDynamicRNN, ADPT_DESC(DynamicRNN))
//构造指向DynamicRNN的指针并储存，创建结构体RegAdptDescDynamicRNN

// DynamicRNNGrad
INPUT_MAP(DynamicRNNGrad) = {
  {1, INPUT_DESC(x)},      {2, INPUT_DESC(w)},      {3, INPUT_DESC(b)},   {4, INPUT_DESC(y)},
  {5, INPUT_DESC(init_h)}, {6, INPUT_DESC(init_c)}, {7, INPUT_DESC(h)},   {8, INPUT_DESC(c)},
  {9, INPUT_DESC(dy)},     {10, INPUT_DESC(dh)},    {11, INPUT_DESC(dc)}, {12, INPUT_DESC(i)},
  {13, INPUT_DESC(j)},     {14, INPUT_DESC(f)},     {15, INPUT_DESC(o)},  {16, INPUT_DESC(tanhct)}};
//将变量x、w、c、b、y、init_h、init_c、h、c、dy、dh、dc、i、j、f、o、tanhct处理并存入对应InputDesc结构体的相应变量中，存入DynamicRNNGrad对应空间内并用input_map指针保存
ATTR_MAP(DynamicRNNGrad) = {{"cell_type", ATTR_DESC(cell_type, AnyTraits<std::string>())},
                            {"direction", ATTR_DESC(direction, AnyTraits<std::string>())},
                            {"cell_depth", ATTR_DESC(cell_depth, AnyTraits<int64_t>())},
                            {"use_peephole", ATTR_DESC(use_peephole, AnyTraits<bool>())},
                            {"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())},
                            {"cell_clip", ATTR_DESC(cell_clip, AnyTraits<float>())},
                            {"num_proj", ATTR_DESC(num_proj, AnyTraits<int64_t>())},
                            {"time_major", ATTR_DESC(time_major, AnyTraits<bool>())},
                            {"forget_bias", ATTR_DESC(forget_bias, AnyTraits<float>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                        存入DynamicRNNGrad对应空间并用attr_map_指针保存
//                                                                                        其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(DynamicRNNGrad) = {{0, OUTPUT_DESC(dw)},
                              {1, OUTPUT_DESC(db)},
                              {2, OUTPUT_DESC(dx)},
                              {3, OUTPUT_DESC(dh_prev)},
                              {4, OUTPUT_DESC(dc_prev)}};
//将变量dw、db、dx、dh_prev、dc_prev处理并存入对应OutputDesc结构体的相应变量中，存入DynamicRNNGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(DynamicRNNGrad, kNameDynamicRNNGrad, ADPT_DESC(DynamicRNNGrad))
//构造指向DynamicRNNGrad的指针并储存，创建结构体RegAdptDescDynamicRNNGrad

// DynamicGRUV2
INPUT_MAP(DynamicGRUV2) = {{1, INPUT_DESC(x)},          {2, INPUT_DESC(weight_input)}, {3, INPUT_DESC(weight_hidden)},
                           {4, INPUT_DESC(bias_input)}, {5, INPUT_DESC(bias_hidden)},  {6, INPUT_DESC(seq_length)},
                           {7, INPUT_DESC(init_h)}};
//将变量x、weight_input、weight_hidden、bias_input、bias_hidden、seq_length、init_h处理并存入对应InputDesc结构体的相应变量中
//存入DynamicGRUV2对应空间内并用input_map指针保存
ATTR_MAP(DynamicGRUV2) = {{"direction", ATTR_DESC(direction, AnyTraits<std::string>())},
                          {"cell_depth", ATTR_DESC(cell_depth, AnyTraits<int64_t>())},
                          {"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())},
                          {"cell_clip", ATTR_DESC(cell_clip, AnyTraits<float>())},
                          {"num_proj", ATTR_DESC(num_proj, AnyTraits<int64_t>())},
                          {"time_major", ATTR_DESC(time_major, AnyTraits<bool>())},
                          {"activation", ATTR_DESC(activation, AnyTraits<std::string>())},
                          {"gate_order", ATTR_DESC(gate_order, AnyTraits<std::string>())},
                          {"reset_after", ATTR_DESC(reset_after, AnyTraits<bool>())},
                          {"is_training", ATTR_DESC(is_training, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                      存入DynamicGRUV2对应空间并用attr_map_指针保存
//                                                                                      其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(DynamicGRUV2) = {{0, OUTPUT_DESC(y)},     {1, OUTPUT_DESC(output_h)}, {2, OUTPUT_DESC(update)},
                            {3, OUTPUT_DESC(reset)}, {4, OUTPUT_DESC(new)},      {5, OUTPUT_DESC(hidden_new)}};
//将变量y、output_h、update、reset、new、hidden_new处理并存入对应OutputDesc结构体的相应变量中，存入DynamicGRUV2对应空间内并用output_map_指针保存
REG_ADPT_DESC(DynamicGRUV2, kNameDynamicGRUV2, ADPT_DESC(DynamicGRUV2))
//构造指向DynamicGRUV2的指针并储存，创建结构体RegAdptDescDynamicGRUV2

// DynamicGRUV2Grad
INPUT_MAP(DynamicGRUV2Grad) = {
  {1, INPUT_DESC(x)},           {2, INPUT_DESC(weight_input)}, {3, INPUT_DESC(weight_hidden)},
  {4, INPUT_DESC(y)},           {5, INPUT_DESC(init_h)},       {6, INPUT_DESC(h)},
  {7, INPUT_DESC(dy)},          {8, INPUT_DESC(dh)},           {9, INPUT_DESC(update)},
  {10, INPUT_DESC(reset)},      {11, INPUT_DESC(new)},         {12, INPUT_DESC(hidden_new)},
  {13, INPUT_DESC(seq_length)}, {14, INPUT_DESC(mask)}};
//将变量x、weight_input、weight_hidden、y、init_h、h、dy、dh、update、reset、new、hidden_new、seq_length、mask处理并存入对应InputDesc结构体的相应变量中
//存入DynamicGRUV2Grad对应空间内并用input_map指针保存
ATTR_MAP(DynamicGRUV2Grad) = {{"direction", ATTR_DESC(direction, AnyTraits<std::string>())},
                              {"cell_depth", ATTR_DESC(cell_depth, AnyTraits<int64_t>())},
                              {"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())},
                              {"cell_clip", ATTR_DESC(cell_clip, AnyTraits<float>())},
                              {"num_proj", ATTR_DESC(num_proj, AnyTraits<int64_t>())},
                              {"time_major", ATTR_DESC(time_major, AnyTraits<bool>())},
                              {"gate_order", ATTR_DESC(gate_order, AnyTraits<std::string>())},
                              {"reset_after", ATTR_DESC(reset_after, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                          存入DynamicGRUV2Grad对应空间并用attr_map_指针保存
//                                                                                          其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(DynamicGRUV2Grad) = {{0, OUTPUT_DESC(dw_input)},  {1, OUTPUT_DESC(dw_hidden)}, {2, OUTPUT_DESC(db_input)},
                                {3, OUTPUT_DESC(db_hidden)}, {4, OUTPUT_DESC(dx)},        {5, OUTPUT_DESC(dh_prev)}};
//将变量dw_input、dw_hidden、db_input、db_hidden、dx、dh_prev处理并存入对应OutputDesc结构体的相应变量中，存入DynamicGRUV2Grad对应空间内并用output_map_指针保存
REG_ADPT_DESC(DynamicGRUV2Grad, kNameDynamicGRUV2Grad, ADPT_DESC(DynamicGRUV2Grad))
//构造指向DynamicGRUV2Grad的指针并储存，创建结构体RegAdptDescDynamicGRUV2Grad
}  // namespace mindspore::transform
