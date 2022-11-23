/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "minddata/mindrecord/include/shard_sequential_sample.h"//按照路径寻找以下文件，导入到本文件

using mindspore::LogStream;//声明mindspore空间下的LogStream
using mindspore::ExceptionType::NoExceptionType;//声明mindspore空间下ExceptionType类中的NoExceptionType
using mindspore::MsLogLevel::ERROR;//声明mindspore空间下MsLogLevel类中的ERROR

namespace mindspore {//创建名为mindspore的空间
namespace mindrecord {//创建名为mindrecord的空间
//引用ShardSequentialSample空间中ShardSequentialSample函数,输入两种参数调用指定函数
//对参数ShardSample、offset_、per_、per_offset_进行参数初始化
ShardSequentialSample::ShardSequentialSample(int64_t n, int64_t offset)
    : ShardSample(n), offset_(offset), per_(0.0f), per_offset_(0.0f) {}
//引用ShardSequentialSample空间中ShardSequentialSample函数,输入两种参数调用指定函数
//对参数ShardSample、offset_、per_、per_offset_进行参数初始化
ShardSequentialSample::ShardSequentialSample(float per, float per_offset)
    : ShardSample(0), offset_(0), per_(per), per_offset_(per_offset) {}
//引用ShardSequentialSample空间中GetNumSamples函数
//判断no_of_samples_是否等于0且per_是否在-kEpsilon到kEpsilon区间内，若是则返回dataset_size
//判断per_是否在kEpsilon到1.0f区间内，若是则返回dataset_size与kEpsilon的乘积
//若均不符合则返回dataset_size和no_of_samples_之间较小的值
int64_t ShardSequentialSample::GetNumSamples(int64_t dataset_size, int64_t num_classes) {
  if (no_of_samples_ == 0 && (per_ >= -kEpsilon && per_ <= kEpsilon)) {
    return dataset_size;
  }
  if (per_ > kEpsilon && per_ <= 1.0f) {
    return dataset_size * kEpsilon;
  }
  return std::min(static_cast<int64_t>(no_of_samples_), dataset_size);
}
//引用ShardSequentialSample空间中Execute函数
Status ShardSequentialSample::Execute(ShardTaskList &tasks) {
  int64_t taking;
  int64_t total_no = static_cast<int64_t>(tasks.sample_ids_.size());
  //判断no_of_samples_是否等于0且per_是否在-kEpsilon到kEpsilon区间内，若是则给taking赋值，值为total_no
  //判断per_是否在kEpsilon到1.0f区间内，若是则给taking赋值，值为total_no与kEpsilon的乘积
  //若均不符合则返回total_no和no_of_samples_之间较小的值
  if (no_of_samples_ == 0 && (per_ >= -kEpsilon && per_ <= kEpsilon)) {
    taking = total_no;
  } else if (per_ > kEpsilon && per_ <= 1.0f) {
    taking = total_no * kEpsilon;
  } else {
    taking = std::min(static_cast<int64_t>(no_of_samples_), total_no);
  }
  //判断tasks中permutation_是否为空，若是则给total_no赋值，值为tasks.Size()，并进入循环，依次给new_tasks赋值
  //引用ShardTaskList空间中的TaskListSwap函数
  if (tasks.permutation_.empty()) {
    ShardTaskList new_tasks;
    total_no = static_cast<int64_t>(tasks.Size());
    for (int64_t i = offset_; i < taking + offset_; ++i) {
      new_tasks.AssignTask(tasks, i % total_no);
    }
    ShardTaskList::TaskListSwap(tasks, new_tasks);
  } else {  // shuffled洗牌
    ShardTaskList new_tasks;
    total_no = static_cast<int64_t>(tasks.permutation_.size());
    for (int64_t i = offset_; i < taking + offset_; ++i) {
      new_tasks.AssignTask(tasks, tasks.permutation_[i % total_no]);
    }
    ShardTaskList::TaskListSwap(tasks, new_tasks);
  }
  return Status::OK();
}

}  // namespace mindrecord
}  // namespace mindspore
