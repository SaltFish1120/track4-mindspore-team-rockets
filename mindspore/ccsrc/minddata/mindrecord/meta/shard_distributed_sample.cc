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

#include "minddata/mindrecord/include/shard_distributed_sample.h"//按照路径寻找以下文件，导入到本文件

using mindspore::LogStream;//声明mindspore空间下的LogStream
using mindspore::ExceptionType::NoExceptionType;//声明mindspore空间下ExceptionType类中的NoExceptionType
using mindspore::MsLogLevel::ERROR;//声明mindspore空间下MsLogLevel类中的ERROR

namespace mindspore {//创建名为mindspore的空间
namespace mindrecord {//创建名为mindrecord的空间
//引用ShardDistributedSample空间中ShardDistributedSample函数,输入七种参数调用指定函数
//对参数ShardSample、shuffle_、no_of_padded_samples_、first_epoch_、shuffle_op_进行进行参数初始化
ShardDistributedSample::ShardDistributedSample(int num_shards, int shard_id, int64_t no_of_padded_samples, bool shuffle,
                                               uint32_t seed, int64_t no_of_samples, int64_t offset)
    : ShardSample(1, num_shards, shard_id, no_of_samples, offset),
      shuffle_(shuffle),
      no_of_padded_samples_(no_of_padded_samples),
      first_epoch_(true) {
  shuffle_op_ = std::make_shared<ShardShuffle>(seed, kShuffleSample);
}
//引用ShardDistributedSample空间中ShardDistributedSample函数,输入六种参数调用指定函数
//对参数ShardDistributedSample进行进行参数初始化
ShardDistributedSample::ShardDistributedSample(int num_shards, int shard_id, bool shuffle, uint32_t seed,
                                               int64_t no_of_samples, int64_t offset)
    : ShardDistributedSample(num_shards, shard_id, 0, shuffle, seed, no_of_samples, offset) {}
//在ShardDistributedSample空间中创建int64_t型GetNumSamples函数,返回值为0或-1
//判断no_of_padded_samples_的值
//若no_of_padded_samples_的值小于等于0,令res变量等于0。若no_of_padded_samples_大于0,则返回0。
//判断dataset_size与denominator_的模是否等于0,若相等则将dataset_size与denominator_和numerator_的乘积的商赋给res
//若不相等,则将dataset_size与denominator_和numerator_的乘积的商+1后赋给res
//返回no_of_samples_,no_of_samples_的取值取决于res是否等于0,若等于0则返回0,若不等于0则返回no_of_samples_与res的最小值
//若no_of_padded_samples_的值大于0,将dataset_size和no_of_padded_samples_的和赋给padded_size
//判断padded_size与denominator_的模是否为0,若是则返回padded_size与denominator_和numerator_的乘积的商,若不是则返回-1
int64_t ShardDistributedSample::GetNumSamples(int64_t dataset_size, int64_t num_classes) {
  if (no_of_padded_samples_ <= 0) {
    int64_t res = 0;
    if (dataset_size % denominator_ == 0) {
      res = dataset_size / denominator_ * numerator_;
    } else {
      res = dataset_size / denominator_ * numerator_ + 1;
    }
    return no_of_samples_ == 0 ? res : std::min(no_of_samples_, res);
  } else {
    auto padded_size = dataset_size + no_of_padded_samples_;
    if (padded_size % denominator_ == 0) {
      return padded_size / denominator_ * numerator_;
    } else {
      return -1;
    }
  }
  return 0;
}
//在ShardDistributedSample空间中创建Status型PreExecute函数,返回值为Status变量
//将tasks.Size()函数的返回值赋给total_no
//判断no_of_padded_samples_是否大于0且first_epoch_是否为真,若是则调用CHECK_FAIL_RETURN_UNEXPECTED函数并输出非有效数据警告
//若no_of_padded_samples_小于0,则判断first_epoch_是否为真,若是则将first_epoch_改为否,将tasks赋给task_。若不是则将task_赋给tasks 
//判断shuffle_是否为真，若是则调用shuffle_op_指针中SetShardSampleCount和UpdateShuffleMode的函数并调用RETURN_IF_NOT_OK函数
//返回Status中Ok函数的返回值
Status ShardDistributedSample::PreExecute(ShardTaskList &tasks) {
  auto total_no = tasks.Size();
  if (no_of_padded_samples_ > 0 && first_epoch_) {
    CHECK_FAIL_RETURN_UNEXPECTED(total_no % denominator_ == 0,
                                 "Invalid data, the size of dataset and padded samples: " + std::to_string(total_no) +
                                   " can not be divisible by the value of 'num_shards': " +
                                   std::to_string(denominator_) + ".\n Please adjust the value of 'num_padded'.");
  }
  if (first_epoch_) {
    first_epoch_ = false;
    task_ = tasks;
  } else {
    tasks = task_;
  }
  if (shuffle_ == true) {
    shuffle_op_->SetShardSampleCount(GetShardSampleCount());
    shuffle_op_->UpdateShuffleMode(GetShuffleMode());
    RETURN_IF_NOT_OK((*shuffle_op_)(tasks));
  }
  return Status::OK();
}
}  // namespace mindrecord
}  // namespace mindspore
