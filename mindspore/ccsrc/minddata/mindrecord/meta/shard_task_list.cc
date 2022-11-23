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

#include "minddata/dataset/util/random.h"//按照路径寻找以下文件，导入到本文件,以下同理
#include "minddata/mindrecord/include/shard_task_list.h"
#include "utils/ms_utils.h"
#include "minddata/mindrecord/include/common/shard_utils.h"

using mindspore::LogStream;//声明mindspore空间下的LogStream
using mindspore::ExceptionType::NoExceptionType;//声明mindspore空间下ExceptionType类中的NoExceptionType
using mindspore::MsLogLevel::DEBUG;//声明mindspore空间下MsLogLevel类中的ERROR

namespace mindspore {//创建名为mindspore的空间
namespace mindrecord {//创建名为mindrecord的空间
//引用ShardTaskList空间中ShardTaskList函数,不输入参数调用指定函数
//对参数categories进行参数初始化
ShardTaskList::ShardTaskList() : categories(1) {}
//引用ShardTaskList空间中ShardTaskList函数,输入一种参数调用指定函数
//对参数categories、permutation_、sample_ids_、task_list_进行参数初始化
ShardTaskList::ShardTaskList(const ShardTaskList &other)
    : categories(other.categories),
      permutation_(other.permutation_),
      sample_ids_(other.sample_ids_),
      task_list_(other.task_list_) {}
  //重载ShardTaskList空间中的TaskList中的other
  //交换多种值的内容
ShardTaskList &ShardTaskList::operator=(const ShardTaskList &other) {
  ShardTaskList tmp(other);
  std::swap(categories, tmp.categories);
  permutation_.swap(tmp.permutation_);
  sample_ids_.swap(tmp.sample_ids_);
  task_list_.swap(tmp.task_list_);
  return *this;
}
//引用ShardTaskList空间中InitSampleIds函数
//进入循环，创建列表
void ShardTaskList::InitSampleIds() {
  // no-op if there already exists sample ids.  Do not clobber previous list如果已经存在示例id，则无操作。不要破坏上一个列表
  if (sample_ids_.empty()) {
    sample_ids_ = std::vector<int64_t>(task_list_.size());
    for (auto i = 0; i < task_list_.size(); i++) {
      sample_ids_[i] = i;
    }
  }
}
//引用ShardTaskList空间中MakePerm函数
//创建permutation_列表
void ShardTaskList::MakePerm() {
  int64_t perm_size = sample_ids_.size();
  permutation_ = std::vector<int64_t>(perm_size);
  for (int64_t i = 0; i < perm_size; i++) {
    permutation_[i] = i;
  }
}
//引用ShardTaskList空间中TaskListSwap函数
// Swap the new_tasks with orig_tasks将新任务与orig_tasks交换
void ShardTaskList::TaskListSwap(ShardTaskList &orig_tasks, ShardTaskList &new_tasks) {
  // When swapping, if the orig_tasks contains fields that need to be preserved after the swap, then swapping with a交换时，如果orig_tasks包含交换后需要保留的字段，则使用
  // new_tasks that does not have those fields will result in clobbering/losing the data after the swap.没有这些字段的new_tasks将导致交换后数据丢失。
  // The task_list_ should not be lost/clobbered.task_list_不应丢失/丢失。
  // This function can be called in the middle of mindrecord's epoch, when orig_tasks.task_list_ is still being这个函数可以在mindrecord的时代中期调用，当orig_任务时。task_list_仍在
  // used by mindrecord op's worker threads. So don't touch its task_list_ since this field should be preserved anyways.
  //由mindrecord op的工作线程使用。因此，不要触摸其task_list_，因为无论如何都应该保留此字段。
  std::swap(orig_tasks.categories, new_tasks.categories);
  std::swap(orig_tasks.permutation_, new_tasks.permutation_);
  std::swap(orig_tasks.sample_ids_, new_tasks.sample_ids_);
}
//引用ShardTaskList空间中PopBack函数
//调用task_list_类中pop_back函数
void ShardTaskList::PopBack() { task_list_.pop_back(); }
//引用ShardTaskList空间中Size函数
//返回task_list_空间中size函数的返回值
int64_t ShardTaskList::Size() const { return static_cast<int64_t>(task_list_.size()); }
//引用ShardTaskList空间中SizeOfRows函数
int64_t ShardTaskList::SizeOfRows() const {
  //判断task_list_的长度是否为0，若是则返回0
  if (task_list_.size() == 0) return static_cast<int64_t>(0);

  // 1 task is 1 page1个任务是1页
  const size_t kBlobInfoIndex = 2;
  auto sum_num_rows = [](int64_t x, ShardTask y) { return x + std::get<kBlobInfoIndex>(y)[0]; };
  int64_t nRows = std::accumulate(task_list_.begin(), task_list_.end(), 0, sum_num_rows);
  return nRows;
}
//引用ShardTaskList空间中GetTaskByID函数
//返回task_list_数组中id的返回值
ShardTask &ShardTaskList::GetTaskByID(int64_t id) { return task_list_[id]; }
//引用ShardTaskList空间中GetTaskSampleByID函数
//返回sample_ids_数组中id的返回值
int64_t ShardTaskList::GetTaskSampleByID(int64_t id) { return sample_ids_[id]; }
//引用ShardTaskList空间中GetRandomTaskID函数
int64_t ShardTaskList::GetRandomTaskID() {
  std::mt19937 gen = mindspore::dataset::GetRandomDevice();
  std::uniform_int_distribution<> dis(0, sample_ids_.size() - 1);
  return dis(gen);
}
//引用ShardTaskList空间中GetRandomTask函数
ShardTask &ShardTaskList::GetRandomTask() {
  std::mt19937 gen = mindspore::dataset::GetRandomDevice();
  std::uniform_int_distribution<> dis(0, task_list_.size() - 1);
  return task_list_[dis(gen)];
}
//引用ShardTaskList空间中Combine函数
ShardTaskList ShardTaskList::Combine(std::vector<ShardTaskList> &category_tasks, bool replacement, int64_t num_elements,
                                     int64_t num_samples) {
  ShardTaskList res;
  //判断category_tasks是否为空，若是则返回res的值
  if (category_tasks.empty()) return res;
  auto total_categories = category_tasks.size();
  res.categories = static_cast<int64_t>(total_categories);
  //判断resplacement是否为false，若是则判断category_tasks列表中的最小值
  //在0至category_tasks最小值的区间内反复调用InsertTask函数进行数据处理，直到num_samples等于0且count等于num_samples
  //若resplacement不为0，则判断category_tasks列表中的最大值
  //在0至category_tasks最小值的区间内反复调用InsertTask函数进行数据处理，直到num_samples等于0且count等于num_samples
  if (replacement == false) {
    auto minTasks = category_tasks[0].Size();
    for (int64_t i = 1; i < total_categories; i++) {
      minTasks = std::min(minTasks, category_tasks[i].Size());
    }
    int64_t count = 0;
    for (int64_t task_no = 0; task_no < minTasks; task_no++) {
      for (int64_t i = 0; i < total_categories; i++) {
        if (num_samples != 0 && count == num_samples) break;
        res.InsertTask(std::move(category_tasks[i].GetTaskByID(task_no)));
        count++;
      }
    }
  } else {
    auto maxTasks = category_tasks[0].Size();
    for (int64_t i = 1; i < total_categories; i++) {
      maxTasks = std::max(maxTasks, category_tasks[i].Size());
    }
    if (num_elements != std::numeric_limits<int64_t>::max()) {
      maxTasks = static_cast<decltype(maxTasks)>(num_elements);
    }
    int64_t count = 0;
    for (int64_t i = 0; i < total_categories; i++) {
      for (int64_t j = 0; j < maxTasks; j++) {
        if (num_samples != 0 && count == num_samples) break;
        res.InsertTask(category_tasks[i].GetRandomTask());
        count++;
      }
    }
  }
  //返回res的值
  return res;
}
}  // namespace mindrecord
}  // namespace mindspore
