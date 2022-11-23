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

#include "minddata/mindrecord/include/shard_pk_sample.h"//按照路径寻找以下文件，导入到本文件

using mindspore::LogStream;//声明mindspore空间下的LogStream
using mindspore::ExceptionType::NoExceptionType;//声明mindspore空间下ExceptionType类中的NoExceptionType
using mindspore::MsLogLevel::ERROR;//声明mindspore空间下MsLogLevel类中的ERROR

namespace mindspore {//创建名为mindspore的空间
namespace mindrecord {//创建名为mindrecord的空间
//引用ShardPkSample空间中ShardPkSample函数,输入三种参数调用指定函数
//对参数ShardCategory、shuffle_、num_samples_进行进行参数初始化
ShardPkSample::ShardPkSample(const std::string &category_field, int64_t num_elements, int64_t num_samples)
    : ShardCategory(category_field, num_elements, std::numeric_limits<int64_t>::max(), true),
      shuffle_(false),
      num_samples_(num_samples) {}
//引用ShardPkSample空间中ShardPkSample函数,输入四种参数调用指定函数
//对参数ShardCategory进行进行参数初始化
ShardPkSample::ShardPkSample(const std::string &category_field, int64_t num_elements, int64_t num_categories,
                             int64_t num_samples)
    : ShardCategory(category_field, num_elements, num_categories, true), shuffle_(false), num_samples_(num_samples) {}
//引用ShardPkSample空间中ShardPkSample函数,输入四种参数调用指定函数
//对参数ShardCategory、shuffle_op_进行进行参数初始化
ShardPkSample::ShardPkSample(const std::string &category_field, int64_t num_elements, int64_t num_categories,
                             uint32_t seed, int64_t num_samples)
    : ShardCategory(category_field, num_elements, num_categories, true), shuffle_(true), num_samples_(num_samples) {
  shuffle_op_ = std::make_shared<ShardShuffle>(seed, kShuffleSample);  // 进行重新排序和替换
}
//在ShardDistributedSample空间中创建Status型PreExecute函数,返回值为Status变量
//判断shuffle_是否为真,若是则调用RETURN_IF_NOT_OK函数
//返回Status中Ok函数的返回值
Status ShardPkSample::SufExecute(ShardTaskList &tasks) {
  if (shuffle_ == true) {
    RETURN_IF_NOT_OK((*shuffle_op_)(tasks));
  }
  return Status::OK();
}
}  // namespace mindrecord
}  // namespace mindspore
