/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_WRITER_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_WRITER_H_

#include <libgen.h>
#include <sys/file.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "minddata/mindrecord/include/shard_column.h"
#include "minddata/mindrecord/include/shard_error.h"
#include "minddata/mindrecord/include/shard_header.h"
#include "minddata/mindrecord/include/shard_index.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace mindrecord {
class __attribute__((visibility("default"))) ShardWriter {
 public:
  ShardWriter();

  ~ShardWriter();

  /// \brief Open file at the beginning 在开头打开文件
  /// \param[in] paths the file names list 文件名列表
  /// \param[in] append new data at the end of file if true, otherwise try to overwrite file 如果为 true，则文件末尾的新数据，否则尝试覆盖文件
  /// \param[in] overwrite a file with the same name if true 具有相同名称的文件（如果为 true）
  /// \return Status
  Status Open(const std::vector<std::string> &paths, bool append = false, bool overwrite = false);

  /// \brief Open file at the ending 在结尾处打开文件
  /// \param[in] paths the file names list 文件名列表
  /// \return MSRStatus the status of MSRStatus MSR状态
  Status OpenForAppend(const std::string &path);

  /// \brief Write header to disk 将标头写入磁盘
  /// \return MSRStatus the status of MSRStatus MSR状态
  Status Commit();

  /// \brief Set file size 设置文件大小
  /// \param[in] header_size the size of header, only (1<<N) is accepted 标头的大小，仅接受 （1<<N）
  /// \return MSRStatus the status of MSRStatus MSR状态
  Status SetHeaderSize(const uint64_t &header_size);

  /// \brief Set page size 设置页面大小
  /// \param[in] page_size the size of page, only (1<<N) is accepted 页面大小，仅接受 （1<<N）
  /// \return MSRStatus the status of MSRStatus MSR状态
  Status SetPageSize(const uint64_t &page_size);

  /// \brief Set shard header 设置分片头
  /// \param[in] header_data the info of header 标题的信息
  ///        WARNING, only called when file is empty 警告，仅在文件为空时调用
  /// \return MSRStatus the status of MSRStatus MSR状态
  Status SetShardHeader(std::shared_ptr<ShardHeader> header_data);

  /// \brief write raw data by group size 按组大小写入原始数据
  /// \param[in] raw_data the vector of raw json data, vector format 原始 json 数据的向量，向量格式
  /// \param[in] blob_data the vector of image data 图像数据的向量
  /// \param[in] sign validate data or not
  /// \return MSRStatus the status of MSRStatus to judge if write successfully MSRStatus 判断写入是否成功的 MSRStatus 状态
  Status WriteRawData(std::map<uint64_t, std::vector<json>> &raw_data, vector<vector<uint8_t>> &blob_data,
                      bool sign = true, bool parallel_writer = false);

  /// \brief write raw data by group size for call from python 按组大小写入原始数据，以便从 python 调用
  /// \param[in] raw_data the vector of raw json data, python-handle format 原始 json 数据的向量，python 句柄格式
  /// \param[in] blob_data the vector of blob json data, python-handle format blob json 数据的向量，python-handle 格式
  /// \param[in] sign validate data or not 验证数据与否
  /// \return MSRStatus the status of MSRStatus to judge if write successfully  MSRStatus 判断写入是否成功的 MSRStatus 状态
  Status WriteRawData(std::map<uint64_t, std::vector<py::handle>> &raw_data,
                      std::map<uint64_t, std::vector<py::handle>> &blob_data, bool sign = true,
                      bool parallel_writer = false);

  Status MergeBlobData(const std::vector<string> &blob_fields,
                       const std::map<std::string, std::unique_ptr<std::vector<uint8_t>>> &row_bin_data,
                       std::shared_ptr<std::vector<uint8_t>> *output);

  static Status Initialize(const std::unique_ptr<ShardWriter> *writer_ptr, const std::vector<std::string> &file_names);

 private:
  /// \brief write shard header data to disk 将分片头数据写入磁盘
  Status WriteShardHeader();

  /// \brief erase error data 擦除错误数据
  void DeleteErrorData(std::map<uint64_t, std::vector<json>> &raw_data, std::vector<std::vector<uint8_t>> &blob_data);

  /// \brief populate error data 填充错误数据
  void PopulateMutexErrorData(const int &row, const std::string &message, std::map<int, std::string> &err_raw_data);

  /// \brief check data 检查数据
  void CheckSliceData(int start_row, int end_row, json schema, const std::vector<json> &sub_raw_data,
                      std::map<int, std::string> &err_raw_data);

  /// \brief write shard header data to disk 将分片头数据写入磁盘
  Status ValidateRawData(std::map<uint64_t, std::vector<json>> &raw_data, std::vector<std::vector<uint8_t>> &blob_data,
                         bool sign, std::shared_ptr<std::pair<int, int>> *count_ptr);

  /// \brief fill data array in multiple thread run 在多线程运行中填充数据数组
  void FillArray(int start, int end, std::map<uint64_t, vector<json>> &raw_data,
                 std::vector<std::vector<uint8_t>> &bin_data);

  /// \brief serialized raw data 序列化的原始数据
  Status SerializeRawData(std::map<uint64_t, std::vector<json>> &raw_data, std::vector<std::vector<uint8_t>> &bin_data,
                          uint32_t row_count);

  /// \brief write all data parallel 并行写入所有数据
  Status ParallelWriteData(const std::vector<std::vector<uint8_t>> &blob_data,
                           const std::vector<std::vector<uint8_t>> &bin_raw_data);

  /// \brief write data shard by shard 逐个分片写入数据分片
  Status WriteByShard(int shard_id, int start_row, int end_row, const std::vector<std::vector<uint8_t>> &blob_data,
                      const std::vector<std::vector<uint8_t>> &bin_raw_data);

  /// \brief break image data up into multiple row groups 将图像数据分解为多个行组
  Status CutRowGroup(int start_row, int end_row, const std::vector<std::vector<uint8_t>> &blob_data,
                     std::vector<std::pair<int, int>> &rows_in_group, const std::shared_ptr<Page> &last_raw_page,
                     const std::shared_ptr<Page> &last_blob_page);

  /// \brief append partial blob data to previous page 将部分 Blob 数据追加到上一页
  Status AppendBlobPage(const int &shard_id, const std::vector<std::vector<uint8_t>> &blob_data,
                        const std::vector<std::pair<int, int>> &rows_in_group,
                        const std::shared_ptr<Page> &last_blob_page);

  /// \brief write new blob data page to disk 将新的 Blob 数据页写入磁盘
  Status NewBlobPage(const int &shard_id, const std::vector<std::vector<uint8_t>> &blob_data,
                     const std::vector<std::pair<int, int>> &rows_in_group,
                     const std::shared_ptr<Page> &last_blob_page);

  /// \brief shift last row group to next raw page for new appending 将最后一行组移动到下一个原始页面以进行新的追加
  Status ShiftRawPage(const int &shard_id, const std::vector<std::pair<int, int>> &rows_in_group,
                      std::shared_ptr<Page> &last_raw_page);

  /// \brief write raw data page to disk 将原始数据页写入磁盘
  Status WriteRawPage(const int &shard_id, const std::vector<std::pair<int, int>> &rows_in_group,
                      std::shared_ptr<Page> &last_raw_page, const std::vector<std::vector<uint8_t>> &bin_raw_data);

  /// \brief generate empty raw data page 生成空的原始数据页面
  Status EmptyRawPage(const int &shard_id, std::shared_ptr<Page> &last_raw_page);

  /// \brief append a row group at the end of raw page 在原始页面末尾追加行组
  Status AppendRawPage(const int &shard_id, const std::vector<std::pair<int, int>> &rows_in_group, const int &chunk_id,
                       int &last_row_groupId, std::shared_ptr<Page> last_raw_page,
                       const std::vector<std::vector<uint8_t>> &bin_raw_data);

  /// \brief write blob chunk to disk 将 blob 块写入磁盘
  Status FlushBlobChunk(const std::shared_ptr<std::fstream> &out, const std::vector<std::vector<uint8_t>> &blob_data,
                        const std::pair<int, int> &blob_row);

  /// \brief write raw chunk to disk 将原始块写入磁盘
  Status FlushRawChunk(const std::shared_ptr<std::fstream> &out, const std::vector<std::pair<int, int>> &rows_in_group,
                       const int &chunk_id, const std::vector<std::vector<uint8_t>> &bin_raw_data);

  /// \brief break up into tasks by shard 按分片分解为任务
  std::vector<std::pair<int, int>> BreakIntoShards();

  /// \brief calculate raw data size row by row 逐行计算原始数据大小
  Status SetRawDataSize(const std::vector<std::vector<uint8_t>> &bin_raw_data);

  /// \brief calculate blob data size row by row 逐行计算 Blob 数据大小
  Status SetBlobDataSize(const std::vector<std::vector<uint8_t>> &blob_data);

  /// \brief populate last raw page pointer 填充最后一个原始页面指针
  Status SetLastRawPage(const int &shard_id, std::shared_ptr<Page> &last_raw_page);

  /// \brief populate last blob page pointer 填充最后一个 blob 页指针
  Status SetLastBlobPage(const int &shard_id, std::shared_ptr<Page> &last_blob_page);

  /// \brief check the data by schema 按架构检查数据
  Status CheckData(const std::map<uint64_t, std::vector<json>> &raw_data);

  /// \brief check the data and type 检查数据和类型
  Status CheckDataTypeAndValue(const std::string &key, const json &value, const json &data, const int &i,
                               std::map<int, std::string> &err_raw_data);

  /// \brief Lock writer and save pages info 锁定编写器并保存页面信息
  Status LockWriter(bool parallel_writer, std::unique_ptr<int> *fd_ptr);

  /// \brief Unlock writer and save pages info 解锁作家并保存页面信息
  Status UnlockWriter(int fd, bool parallel_writer = false);

  /// \brief Check raw data before writing 写入前检查原始数据
  Status WriteRawDataPreCheck(std::map<uint64_t, std::vector<json>> &raw_data, vector<vector<uint8_t>> &blob_data,
                              bool sign, int *schema_count, int *row_count);

  /// \brief Get full path from file name 从文件名获取完整路径
  Status GetFullPathFromFileName(const std::vector<std::string> &paths);

  /// \brief Open files 打开文件
  Status OpenDataFiles(bool append, bool overwrite);

  /// \brief Remove lock file 删除锁定文件
  Status RemoveLockFile();

  /// \brief Remove lock file 删除锁定文件
  Status InitLockFile();

 private:
  const std::string kLockFileSuffix = "_Locker";
  const std::string kPageFileSuffix = "_Pages";
  std::string lock_file_;   // lock file for parallel run 锁定文件以进行并行运行
  std::string pages_file_;  // temporary file of pages info for parallel run 用于并行运行的页面信息的临时文件

  int shard_count_;        // number of files 文件数
  uint64_t header_size_;   // header size 页眉大小
  uint64_t page_size_;     // page size  页面大小
  uint32_t row_count_;     // count of rows 行数
  uint32_t schema_count_;  // count of schemas 架构计数

  std::vector<uint64_t> raw_data_size_;   // Raw data size 原始数据大小
  std::vector<uint64_t> blob_data_size_;  // Blob data size Blob 数据大小

  std::vector<std::string> file_paths_;                      // file paths 文件路径
  std::vector<std::shared_ptr<std::fstream>> file_streams_;  // file handles 文件句柄
  std::shared_ptr<ShardHeader> shard_header_;                // shard header 分片头
  std::shared_ptr<ShardColumn> shard_column_;                // shard columns 分片列

  std::map<uint64_t, std::map<int, std::string>> err_mg_;  // used for storing error raw_data info 用于存储错误raw_data信息

  std::mutex check_mutex_;  // mutex for data check 用于数据检查的互斥锁
  std::atomic<bool> flag_{false};
  std::atomic<int64_t> compression_size_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_WRITER_H_
