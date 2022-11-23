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

#include "minddata/mindrecord/include/shard_sample.h"//按照路径寻找以下文件，导入到本文件

using mindspore::LogStream;//声明mindspore空间下的LogStream
using mindspore::ExceptionType::NoExceptionType;//声明mindspore空间下ExceptionType类中的NoExceptionType
using mindspore::MsLogLevel::ERROR;//声明mindspore空间下MsLogLevel类中的ERROR

namespace mindspore {//创建名为mindspore的空间
namespace mindrecord {//创建名为mindrecord的空间
//引用ShardSample空间中ShardSample函数,输入参数n调用指定函数
//对参数numerator_、denominator_、partition_id_、no_of_samples_、indices_、sampler_type_、offset_进行参数初始化
ShardSample::ShardSample(int64_t n)
    : numerator_(0),
      denominator_(0),
      partition_id_(0),
      no_of_samples_(n),
      indices_({}),
      sampler_type_(kCustomTopNSampler),
      offset_(-1) {}
//引用ShardSample空间中ShardSample函数,输入两种参数调用指定函数
//对参数numerator_、denominator_、partition_id_、no_of_samples_、indices_、sampler_type_、offset_进行参数初始化
ShardSample::ShardSample(int64_t num, int64_t den)
    : numerator_(num),
      denominator_(den),
      partition_id_(0),
      no_of_samples_(0),
      indices_({}),
      sampler_type_(kCustomTopPercentSampler),
      offset_(-1) {}
//引用ShardSample空间中ShardSample函数,输入五种参数调用指定函数
//对参数numerator_、denominator_、partition_id_、no_of_samples_、indices_、sampler_type_、offset_进行参数初始化
ShardSample::ShardSample(int64_t num, int64_t den, int64_t par, int64_t no_of_samples, int64_t offset)
    : numerator_(num),
      denominator_(den),
      partition_id_(par),
      no_of_samples_(no_of_samples),
      indices_({}),
      sampler_type_(kCustomTopPercentSampler),
      offset_(offset) {}
//引用ShardSample空间中ShardSample函数,输入参数indices调用指定函数
//对参数numerator_、denominator_、partition_id_、no_of_samples_、indices_、sampler_type_、offset_进行参数初始化
ShardSample::ShardSample(const std::vector<int64_t> &indices)
    : numerator_(0),
      denominator_(0),
      partition_id_(0),
      no_of_samples_(0),
      indices_(indices),
      sampler_type_(kSubsetSampler) {}
//引用ShardSample空间中ShardSample函数,输入参数indices调用指定函数，此函数继承ShardSample函数
//对参数sampler_type_、shuffle_op_进行参数初始化
ShardSample::ShardSample(const std::vector<int64_t> &indices, uint32_t seed) : ShardSample(indices) {
  sampler_type_ = kSubsetRandomSampler;
  shuffle_op_ = std::make_shared<ShardShuffle>(seed);
}
//在ShardCategory空间中创建int64_t型GetNumSamples函数,返回值为int64_t型参数
//判断sampler_type_与kCustomTopNSampler是否赋值相同，若是则返回no_of_samples_
//判断sampler_type_与kCustomTopPercentSampler是否复制相同，若是则再次判断dataset_size与denominator_的模是否为0，若是则返回dataset_size与denominator_和numerator_的乘积的商
//若不是则返回dataset_size与denominator_和numerator_的乘积的商+1的值
//判断sampler_type_与kSubsetRandomSampler的值或者sampler_type_与kSubsetSampler的值是否相等，若任一等式成立则返回indices_.size函数的返回值
//若均不符合条件，最后返回0
int64_t ShardSample::GetNumSamples(int64_t dataset_size, int64_t num_classes) {
  if (sampler_type_ == kCustomTopNSampler) {
    return no_of_samples_;
  }

  if (sampler_type_ == kCustomTopPercentSampler) {
    if (dataset_size % denominator_ == 0) {
      return dataset_size / denominator_ * numerator_;
    } else {
      return dataset_size / denominator_ * numerator_ + 1;
    }
  }
  if (sampler_type_ == kSubsetRandomSampler || sampler_type_ == kSubsetSampler) {
    return indices_.size();
  }
  return 0;
}
//在ShardDistributedSample空间中创建Status型PreExecute函数,返回值为Status变量
//判断tasks.permutation_.empty函数的返回值是否为真，若是则将tasks.sample_ids_.size()的返回值赋给total_no并调用CHECK_FAIL_RETURN_UNEXPECTED函数
//判断sampler_type_与kSubsetRandomSampler的值或者sampler_type_与kSubsetSampler的值是否相等，若任一等式成立则进入循环，修改index的赋值并调用new_tasks.AssignTask函数
//若两等式均不成立，则判断nums_per_shard_.empty函数的返回值是否为真，若是则进入循环，调用new_tasks.AssignTask函数并累加count变量，直至no_of_samples_不等于0且count与no_of_samples_相等均成立时跳出循环
//若不是，则进入另一循环，循环次数与上式不同，循环运行内容相同
//若tasks.permutation_.empty函数的返回值为假，则将tasks.sample_ids_.size()的返回值赋给total_no并调用CHECK_FAIL_RETURN_UNEXPECTED函数，直接进入执行相同操作的循环，次数不同
//在输出返回值前调用ShardTaskList类中的TaskListSwap函数
//返回Status中Ok函数的返回值
Status ShardSample::UpdateTasks(ShardTaskList &tasks, int64_t taking) {
  if (tasks.permutation_.empty()) {
    ShardTaskList new_tasks;
    auto total_no = tasks.sample_ids_.size();
    CHECK_FAIL_RETURN_UNEXPECTED(total_no > 0,
                                 "[Internal ERROR] 'total_no' should be positive but got: " + std::to_string(total_no));
    if (sampler_type_ == kSubsetRandomSampler || sampler_type_ == kSubsetSampler) {
      for (int64_t i = 0; i < indices_.size(); ++i) {
        int64_t index = ((indices_[i] % total_no) + total_no) % total_no;
        new_tasks.AssignTask(tasks, index);  // different mod result between c and python c和python有不同mod结果
      }
    } else {
      int64_t count = 0;
      if (nums_per_shard_.empty()) {
        for (int64_t i = partition_id_ * taking; i < (partition_id_ + 1) * taking; i++) {
          if (no_of_samples_ != 0 && count == no_of_samples_) break;
          new_tasks.AssignTask(tasks, i % total_no);  // rounding up. if overflow, go back to start四舍五入。如果溢出，返回开始
          count++;
        }
      } else {
        // Get samples within a specific range获取特定范围内的样本
        int64_t i = partition_id_ - 1 >= 0 ? nums_per_shard_[partition_id_ - 1] : 0;
        for (; i < nums_per_shard_[partition_id_]; i++) {
          if (no_of_samples_ != 0 && count == no_of_samples_) break;
          new_tasks.AssignTask(tasks, i % total_no);
          count++;
        }
      }
    }
    ShardTaskList::TaskListSwap(tasks, new_tasks);
  } else {
    ShardTaskList new_tasks;
    int64_t total_no = tasks.permutation_.size();
    CHECK_FAIL_RETURN_UNEXPECTED(total_no > 0,
                                 "[Internal ERROR] 'total_no' should be positive but got: " + std::to_string(total_no));
    int64_t cnt = 0;
    for (int64_t i = partition_id_ * taking; i < (partition_id_ + 1) * taking; i++) {
      if (no_of_samples_ != 0 && cnt == no_of_samples_) break;
      new_tasks.AssignTask(tasks, tasks.permutation_[i % total_no]);
      cnt++;
    }
    ShardTaskList::TaskListSwap(tasks, new_tasks);
  }
  return Status::OK();
}
//在ShardDistributedSample空间中创建Status型PreExecute函数,返回值为Status变量
//判断offset_是否为-1，若不是则进入循环，对samples_per_buffer_、remainder进行赋值并对remainder、offset_进行判断后调整samples_per_buffer_的赋值，最终调用nums_per_shard_.push_back函数
//判断sampler_type_与kCustomTopNSampler是否相等，若相等则对no_of_samples_、taking = no_of_samples_赋值
//若不相等则判断sampler_type_ 与kSubsetRandomSampler是否相等或sampler_type_与kSubsetSampler是否相等，其中任一成立即调用CHECK_FAIL_RETURN_UNEXPECTED函数并输出非法输入警告
//若均不符合则判断numerator_、denominator_是否均大于0且numerator_是否小于等于denominator_ ，若是则继续判断numerator_是否等于1且denominator_是否大于1，若是则对taking赋值
//若不是则对taking进行其他操作的赋值
//若不符合第一条件，则调用RETURN_STATUS_UNEXPECTED函数输出标准变量不符合警告
//返回UpdateTasks函数的返回值
Status ShardSample::Execute(ShardTaskList &tasks) {
  if (offset_ != -1) {
    int64_t old_v = 0;
    int64_t num_rows_ = tasks.sample_ids_.size();
    for (int64_t x = 0; x < denominator_; x++) {
      int64_t samples_per_buffer_ = (num_rows_ + offset_) / denominator_;
      int64_t remainder = (num_rows_ + offset_) % denominator_;
      if (x < remainder) samples_per_buffer_++;
      if (x < offset_) samples_per_buffer_--;
      old_v += samples_per_buffer_;
      // nums_per_shard_ is used to save the current shard's ending index nums_per_shard用于保存当前碎片的结束索引
      nums_per_shard_.push_back(old_v);
    }
  }
  int no_of_categories = static_cast<int>(tasks.categories);
  int64_t total_no = tasks.sample_ids_.size();
  int64_t taking = 0;
  if (sampler_type_ == kCustomTopNSampler) {  // non sharding case constructor #1非分片case构造函数#1
    no_of_samples_ = std::min(no_of_samples_, total_no);
    taking = no_of_samples_ - no_of_samples_ % no_of_categories;
  } else if (sampler_type_ == kSubsetRandomSampler || sampler_type_ == kSubsetSampler) {
    CHECK_FAIL_RETURN_UNEXPECTED(static_cast<int64_t>(indices_.size()) <= total_no,
                                 "Invalid input, indices size: " + std::to_string(indices_.size()) +
                                   " should be less than or equal to database size: " + std::to_string(total_no) + ".");
  } else {  // constructor TopPercent顶部百分比
    if (numerator_ > 0 && denominator_ > 0 && numerator_ <= denominator_) {
      if (numerator_ == 1 && denominator_ > 1) {  // sharding分片
        taking = (total_no + denominator_ - 1) / denominator_;
      } else {  // non sharding不分片
        taking = total_no * numerator_ / denominator_;
        taking -= (taking % no_of_categories);
      }
    } else {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] 'numerator_': " + std::to_string(numerator_) +
                               " should be positive and less than denominator_: " + std::to_string(denominator_) + ".");
    }
  }
  return UpdateTasks(tasks, taking);
}
//在ShardDistributedSample空间中创建Status型PreExecute函数,返回值为Status变量
//判断sampler_type_和kSubsetRandomSampler的值是否相等，若是则调用RETURN_IF_NOT_OK函数
//返回Status中Ok函数的返回值
Status ShardSample::SufExecute(ShardTaskList &tasks) {
  if (sampler_type_ == kSubsetRandomSampler) {
    RETURN_IF_NOT_OK((*shuffle_op_)(tasks));
  }
  return Status::OK();
}
}  // namespace mindrecord
}  // namespace mindspore
