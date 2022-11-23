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

#include "minddata/mindrecord/include/shard_header.h"//按照路径寻找以下文件，导入到本文件

#include <map>//
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "utils/file_utils.h"//按照路径寻找以下文件，导入到本文件
#include "utils/ms_utils.h"
#include "minddata/mindrecord/include/shard_error.h"
#include "minddata/mindrecord/include/shard_page.h"

using mindspore::LogStream;//声明mindspore空间下的LogStream
using mindspore::ExceptionType::NoExceptionType;//声明mindspore空间下ExceptionType类中的NoExceptionType
using mindspore::MsLogLevel::ERROR;//声明mindspore空间下MsLogLevel类中的ERROR

namespace mindspore {//创建名为mindspore的空间
namespace mindrecord {//创建名为mindrecord的空间
//在atomic空间中创建bool型thread_status函数
std::atomic<bool> thread_status(false);
//引用ShardHeader空间中ShardHeader函数,输入三种参数调用指定函数
//对参数shard_count_、header_size_、page_size_、compression_size_进行进行参数初始化
//给index_变量赋值，值为访问Index空间的数量
ShardHeader::ShardHeader() : shard_count_(0), header_size_(0), page_size_(0), compression_size_(0) {
  index_ = std::make_shared<Index>();
}
//在ShardHeader空间中创建InitializeHeader函数
Status ShardHeader::InitializeHeader(const std::vector<json> &headers, bool load_dataset) {
  shard_count_ = headers.size();
  int shard_index = 0;
  bool first = true;
  //进入循环，判断first是否为真，若是则进行header列表的创建
  for (const auto &header : headers) {
    if (first) {
      first = false;
      RETURN_IF_NOT_OK(ParseSchema(header["schema"]));
      RETURN_IF_NOT_OK(ParseIndexFields(header["index_fields"]));
      RETURN_IF_NOT_OK(ParseStatistics(header["statistics"]));
      ParseShardAddress(header["shard_addresses"]);
      header_size_ = header["header_size"].get<uint64_t>();
      page_size_ = header["page_size"].get<uint64_t>();
      compression_size_ = header.contains("compression_size") ? header["compression_size"].get<uint64_t>() : 0;
    }
    RETURN_IF_NOT_OK(ParsePage(header["page"], shard_index, load_dataset));
    shard_index++;
  }
  return Status::OK();
}
//在ShardHeader空间中创建CheckFileStatus函数
Status ShardHeader::CheckFileStatus(const std::string &path) {
  auto realpath = FileUtils::GetRealPath(path.c_str());
  //捕捉错误，并输出错误信息
  CHECK_FAIL_RETURN_UNEXPECTED(
    realpath.has_value(),
    "Invalid file, failed to get the realpath of mindrecord files. Please check file path: " + path);
  std::ifstream fin(realpath.value(), std::ios::in | std::ios::binary);
  CHECK_FAIL_RETURN_UNEXPECTED(fin.is_open(),
                               "Invalid file, failed to open files for loading mindrecord files. Please check file "
                               "path, permission and open file limit: " +
                                 path);
  // fetch file size获取文件大小
  auto &io_seekg = fin.seekg(0, std::ios::end);
  if (!io_seekg.good() || io_seekg.fail() || io_seekg.bad()) {
    fin.close();
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] failed to seekg file, file path: " + path);
  }
  //捕捉错误，并输出错误信息
  size_t file_size = fin.tellg();
  if (file_size < kMinFileSize) {
    fin.close();
    RETURN_STATUS_UNEXPECTED("Invalid file, the size of mindrecord file: " + std::to_string(file_size) +
                             " is smaller than the lower limit: " + std::to_string(kMinFileSize) +
                             ".\n Please check file path: " + path +
                             " and use 'FileWriter' to generate valid mindrecord files.");
  }
  fin.close();
  return Status::OK();
}
//在ShardHeader空间中创建ValidateHeader函数
Status ShardHeader::ValidateHeader(const std::string &path, std::shared_ptr<json> *header_ptr) {
  RETURN_UNEXPECTED_IF_NULL(header_ptr);
  RETURN_IF_NOT_OK(CheckFileStatus(path));
  //
  auto realpath = FileUtils::GetRealPath(path.c_str());
  CHECK_FAIL_RETURN_UNEXPECTED(
    realpath.has_value(),
    "Invalid file, failed to get the realpath of mindrecord files. Please check file path: " + path);

  // read header size读取标头大小
  json json_header;
  std::ifstream fin(realpath.value(), std::ios::in | std::ios::binary);
  CHECK_FAIL_RETURN_UNEXPECTED(fin.is_open(),
                               "Invalid file, failed to open files for loading mindrecord files. Please check file "
                               "path, permission and open file limit: " +
                                 path);
  //判断io_rea的相关参数，若符合则关闭fin，并返回错误信息
  uint64_t header_size = 0;
  auto &io_read = fin.read(reinterpret_cast<char *>(&header_size), kInt64Len);
  if (!io_read.good() || io_read.fail() || io_read.bad()) {
    fin.close();
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] failed to read file, file path: " + path);
  }
  //判断header列表的大小是否大于header应有的大小
  //若符合则关闭fin并返回错误信息
  if (header_size > kMaxHeaderSize) {
    fin.close();
    RETURN_STATUS_UNEXPECTED(
      "Invalid file, the size of mindrecord file header is larger than the upper limit. \nPlease use 'FileWriter' to "
      "generate valid mindrecord files.");
  }

  // read header content读取标题内容
  std::vector<uint8_t> header_content(header_size);
  auto &io_read_content = fin.read(reinterpret_cast<char *>(&header_content[0]), header_size);
  if (!io_read_content.good() || io_read_content.fail() || io_read_content.bad()) {
    fin.close();
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Failed to read file, file path: " + path);
  }

  fin.close();
  std::string raw_header_content = std::string(header_content.begin(), header_content.end());
  // parse json content解析json内容
  try {
    json_header = json::parse(raw_header_content);
  } catch (json::parse_error &e) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Failed to parse the metadata in JSON format in the mindrecord files: " +
                             std::string(e.what()));
  }
  *header_ptr = std::make_shared<json>(json_header);
  return Status::OK();
}
//引用ShardHeader空间中BuildSingleHeader函数
//创建singleheader列表
Status ShardHeader::BuildSingleHeader(const std::string &file_path, std::shared_ptr<json> *header_ptr) {
  RETURN_UNEXPECTED_IF_NULL(header_ptr);
  std::shared_ptr<json> raw_header;
  RETURN_IF_NOT_OK(ValidateHeader(file_path, &raw_header));
  uint64_t compression_size =
    raw_header->contains("compression_size") ? (*raw_header)["compression_size"].get<uint64_t>() : 0;
  json header = {{"shard_addresses", (*raw_header)["shard_addresses"]},
                 {"header_size", (*raw_header)["header_size"]},
                 {"page_size", (*raw_header)["page_size"]},
                 {"compression_size", compression_size},
                 {"index_fields", (*raw_header)["index_fields"]},
                 {"blob_fields", (*raw_header)["schema"][0]["blob_fields"]},
                 {"schema", (*raw_header)["schema"][0]["schema"]},
                 {"version", (*raw_header)["version"]}};
  *header_ptr = std::make_shared<json>(header);
  return Status::OK();
}
//引用ShardHeader空间中BuildDataset函数
//创建dataset列表
Status ShardHeader::BuildDataset(const std::vector<std::string> &file_paths, bool load_dataset) {
  uint32_t thread_num = std::thread::hardware_concurrency();
  if (thread_num == 0) {
    thread_num = kThreadNumber;
  }
  uint32_t work_thread_num = 0;
  uint32_t shard_count = file_paths.size();
  int group_num = ceil(shard_count * 1.0 / thread_num);
  std::vector<std::thread> thread_set(thread_num);
  std::vector<json> headers(shard_count);
  for (uint32_t x = 0; x < thread_num; ++x) {
    int start_num = x * group_num;
    int end_num = ((x + 1) * group_num > shard_count) ? shard_count : (x + 1) * group_num;
    if (start_num >= end_num) {
      continue;
    }

    thread_set[x] =
      std::thread(&ShardHeader::GetHeadersOneTask, this, start_num, end_num, std::ref(headers), file_paths);
    work_thread_num++;
  }

  for (uint32_t x = 0; x < work_thread_num; ++x) {
    thread_set[x].join();
  }
  if (thread_status) {
    thread_status = false;
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Error raised in GetHeadersOneTask function.");
  }
  RETURN_IF_NOT_OK(InitializeHeader(headers, load_dataset));
  return Status::OK();
}
//引用ShardHeader空间中GetHeadersOneTask函数
void ShardHeader::GetHeadersOneTask(int start, int end, std::vector<json> &headers,
                                    const vector<string> &realAddresses) {
  if (thread_status || end > realAddresses.size()) {
    return;
  }
  for (int x = start; x < end; ++x) {
    std::shared_ptr<json> header;
    auto status = ValidateHeader(realAddresses[x], &header);
    if (status.IsError()) {
      thread_status = true;
      return;
    }
    (*header)["shard_addresses"] = realAddresses;
    if (std::find(kSupportedVersion.begin(), kSupportedVersion.end(), (*header)["version"]) ==
        kSupportedVersion.end()) {
      MS_LOG(ERROR) << "Invalid file, the version of mindrecord files" << (*header)["version"].dump()
                    << " is not supported.\nPlease use 'FileWriter' to generate valid mindrecord files.";
      thread_status = true;
      return;
    }
    headers[x] = *header;
  }
}
//引用ShardHeader空间中InitByFiles函数
Status ShardHeader::InitByFiles(const std::vector<std::string> &file_paths) {
  std::vector<std::string> file_names(file_paths.size());
  std::transform(file_paths.begin(), file_paths.end(), file_names.begin(), [](std::string fp) -> std::string {
    std::shared_ptr<std::string> fn;
    return GetFileName(fp, &fn).IsOk() ? *fn : "";
  });

  shard_addresses_ = std::move(file_names);
  shard_count_ = file_paths.size();
  CHECK_FAIL_RETURN_UNEXPECTED(shard_count_ != 0 && (shard_count_ <= kMaxShardCount),
                               "[Internal ERROR] 'shard_count_': " + std::to_string(shard_count_) +
                                 "is not in range (0, " + std::to_string(kMaxShardCount) + "].");
  pages_.resize(shard_count_);
  return Status::OK();
}
//引用ShardHeader空间中ParseIndexFields函数
//进入循环遍历index_field列表，获得parsed_index_fields列表
Status ShardHeader::ParseIndexFields(const json &index_fields) {
  std::vector<std::pair<uint64_t, std::string>> parsed_index_fields;
  for (auto &index_field : index_fields) {
    auto schema_id = index_field["schema_id"].get<uint64_t>();
    std::string field_name = index_field["index_field"].get<std::string>();
    std::pair<uint64_t, std::string> parsed_index_field(schema_id, field_name);
    parsed_index_fields.push_back(parsed_index_field);
  }
  RETURN_IF_NOT_OK(AddIndexFields(parsed_index_fields));
  return Status::OK();
}
//引用ShardHeader空间中ParsePage函数
Status ShardHeader::ParsePage(const json &pages, int shard_index, bool load_dataset) {
  // set shard_index when load_dataset is false当load_dataset为false时设置shard_index
  CHECK_FAIL_RETURN_UNEXPECTED(shard_count_ <= kMaxFileCount,
                               "Invalid file, the number of mindrecord files: " + std::to_string(shard_count_) +
                                 "is not in range (0, " + std::to_string(kMaxFileCount) +
                                 "].\nPlease use 'FileWriter' to generate fewer mindrecord files.");
  //判断pages_列表是否为空，若是则更改pages_列表的大小
  if (pages_.empty()) {
    pages_.resize(shard_count_);
  }
  //进入循环遍历pages_列表，设置pages_列表内容
  for (auto &page : pages) {
    int page_id = page["page_id"];
    int shard_id = page["shard_id"];
    std::string page_type = page["page_type"];
    int page_type_id = page["page_type_id"];
    auto start_row_id = page["start_row_id"].get<uint64_t>();
    auto end_row_id = page["end_row_id"].get<uint64_t>();

    std::vector<std::pair<int, uint64_t>> row_group_ids(page["row_group_ids"].size());
    std::transform(page["row_group_ids"].begin(), page["row_group_ids"].end(), row_group_ids.begin(),
                   [](json rg) { return std::make_pair(rg["id"], rg["offset"].get<uint64_t>()); });

    auto page_size = page["page_size"].get<uint64_t>();

    std::shared_ptr<Page> parsed_page = std::make_shared<Page>(page_id, shard_id, page_type, page_type_id, start_row_id,
                                                               end_row_id, row_group_ids, page_size);
    if (load_dataset == true) {
      pages_[shard_id].push_back(std::move(parsed_page));
    } else {
      pages_[shard_index].push_back(std::move(parsed_page));
    }
  }
  return Status::OK();
}
//引用ShardHeader空间中ParseStatistics函数
Status ShardHeader::ParseStatistics(const json &statistics) {
  //进入循环，遍历statistics列表，若有错误输出错误信息
  for (auto &statistic : statistics) {
    CHECK_FAIL_RETURN_UNEXPECTED(
      statistic.find("desc") != statistic.end() && statistic.find("statistics") != statistic.end(),
      "[Internal ERROR] Failed to deserialize statistics: " + statistics.dump());
    std::string statistic_description = statistic["desc"].get<std::string>();
    json statistic_body = statistic["statistics"];
    std::shared_ptr<Statistics> parsed_statistic = Statistics::Build(statistic_description, statistic_body);
    RETURN_UNEXPECTED_IF_NULL(parsed_statistic);
    AddStatistic(parsed_statistic);
  }
  return Status::OK();
}
//引用ShardHeader空间中ParseShardAddress函数
Status ShardHeader::ParseSchema(const json &schemas) {
  for (auto &schema : schemas) {
    // change how we get schemaBody once design is finalized设计完成后，更改获取schemaBody的方式
    CHECK_FAIL_RETURN_UNEXPECTED(schema.find("desc") != schema.end() && schema.find("blob_fields") != schema.end() &&
                                   schema.find("schema") != schema.end(),
                                 "[Internal ERROR] Failed to deserialize schema: " + schema.dump());
    std::string schema_description = schema["desc"].get<std::string>();
    std::vector<std::string> blob_fields = schema["blob_fields"].get<std::vector<std::string>>();
    json schema_body = schema["schema"];
    std::shared_ptr<Schema> parsed_schema = Schema::Build(schema_description, schema_body);
    RETURN_UNEXPECTED_IF_NULL(parsed_schema);
    AddSchema(parsed_schema);
  }
  return Status::OK();
}
//引用ShardHeader空间中ParseShardAddress函数
//复制address列表到shard_addresses_
void ShardHeader::ParseShardAddress(const json &address) {
  std::copy(address.begin(), address.end(), std::back_inserter(shard_addresses_));
}
//引用ShardHeader空间中SerializeHeader函数
std::vector<std::string> ShardHeader::SerializeHeader() {
  std::vector<std::string> header;
  auto index = SerializeIndexFields();
  auto stats = SerializeStatistics();
  auto schema = SerializeSchema();
  auto pages = SerializePage();
  auto address = SerializeShardAddress();
  //判断shard_count_是否符合条件，若符合则返回string类型
  if (shard_count_ > static_cast<int>(pages.size())) {
    return std::vector<string>{};
  }
  //判断shard_count_是否小于shard_count_最大值，若是则进入循环，编辑s字符串，最后返回字符串信息
  if (shard_count_ <= kMaxShardCount) {
    for (int shardId = 0; shardId < shard_count_; shardId++) {
      string s;
      s += "{\"header_size\":" + std::to_string(header_size_) + ",";
      s += "\"index_fields\":" + index + ",";
      s += "\"page\":" + pages[shardId] + ",";
      s += "\"page_size\":" + std::to_string(page_size_) + ",";
      s += "\"compression_size\":" + std::to_string(compression_size_) + ",";
      s += "\"schema\":" + schema + ",";
      s += "\"shard_addresses\":" + address + ",";
      s += "\"shard_id\":" + std::to_string(shardId) + ",";
      s += "\"statistics\":" + stats + ",";
      s += "\"version\":\"" + std::string(kVersion) + "\"";
      s += "}";
      header.emplace_back(s);
    }
  }
  return header;
}
//引用ShardHeader空间中SerializeIndexFields函数
//遍历fields列表，返回schema_id、index_field的值
std::string ShardHeader::SerializeIndexFields() {
  json j;
  auto fields = index_->GetFields();
  (void)std::transform(fields.begin(), fields.end(), std::back_inserter(j),
                       [](const std::pair<uint64_t, std::string> &field) -> json {
                         return {{"schema_id", field.first}, {"index_field", field.second}};
                       });
  return j.dump();
}
//引用ShardHeader空间中SerializePage函数
//遍历shard_pages列表，返回GetPage函数的值
std::vector<std::string> ShardHeader::SerializePage() {
  std::vector<string> pages;
  for (auto &shard_pages : pages_) {
    json j;
    (void)std::transform(shard_pages.begin(), shard_pages.end(), std::back_inserter(j),
                         [](const std::shared_ptr<Page> &p) { return p->GetPage(); });
    pages.emplace_back(j.dump());
  }
  return pages;
}
//引用ShardHeader空间中SerializeStatistics函数
//遍历statistics_列表，返回GetStatistics函数的值
std::string ShardHeader::SerializeStatistics() {
  json j;
  (void)std::transform(statistics_.begin(), statistics_.end(), std::back_inserter(j),
                       [](const std::shared_ptr<Statistics> &stats) { return stats->GetStatistics(); });
  return j.dump();
}
//引用ShardHeader空间中SerializeSchema函数
//遍历schema_列表，返回GetSchema函数的值
std::string ShardHeader::SerializeSchema() {
  json j;
  (void)std::transform(schema_.begin(), schema_.end(), std::back_inserter(j),
                       [](const std::shared_ptr<Schema> &schema) { return schema->GetSchema(); });
  return j.dump();
}
//引用ShardHeader空间中SerializeShardAddress函数
//进入循环，一次根据地址获得文件名并判断是否为空
std::string ShardHeader::SerializeShardAddress() {
  json j;
  std::shared_ptr<std::string> fn_ptr;
  for (const auto &addr : shard_addresses_) {
    (void)GetFileName(addr, &fn_ptr);
    (void)j.emplace_back(*fn_ptr);
  }
  return j.dump();
}
//引用ShardHeader空间中GetPage函数
//判断shard_id、page_id是否在标准范围内，若是则获取pages_列表中的对应信息，若不是则返回错误信息
Status ShardHeader::GetPage(const int &shard_id, const int &page_id, std::shared_ptr<Page> *page_ptr) {
  RETURN_UNEXPECTED_IF_NULL(page_ptr);
  if (shard_id < static_cast<int>(pages_.size()) && page_id < static_cast<int>(pages_[shard_id].size())) {
    *page_ptr = pages_[shard_id][page_id];
    return Status::OK();
  }
  page_ptr = nullptr;
  RETURN_STATUS_UNEXPECTED("[Internal ERROR] Failed to get Page, 'page_id': " + std::to_string(page_id));
}
//引用ShardHeader空间中SetPage函数
//判断shard_id、page_id是否在标准范围内，若是则在pages_列表中加入新值，若不是则返回错误信息
Status ShardHeader::SetPage(const std::shared_ptr<Page> &new_page) {
  int shard_id = new_page->GetShardID();
  int page_id = new_page->GetPageID();
  if (shard_id < static_cast<int>(pages_.size()) && page_id < static_cast<int>(pages_[shard_id].size())) {
    pages_[shard_id][page_id] = new_page;
    return Status::OK();
  }
  RETURN_STATUS_UNEXPECTED("[Internal ERROR] Failed to set Page, 'page_id': " + std::to_string(page_id));
}
//引用ShardHeader空间中AddPage函数
//判断shard_id、page_id是否在标准范围内，若是则在pages_列表中添加page，若不是则返回错误信息
Status ShardHeader::AddPage(const std::shared_ptr<Page> &new_page) {
  int shard_id = new_page->GetShardID();
  int page_id = new_page->GetPageID();
  if (shard_id < static_cast<int>(pages_.size()) && page_id == static_cast<int>(pages_[shard_id].size())) {
    pages_[shard_id].push_back(new_page);
    return Status::OK();
  }
  RETURN_STATUS_UNEXPECTED("[Internal ERROR] Failed to add Page, 'page_id': " + std::to_string(page_id));
}
//引用ShardHeader空间中GetLastPageId函数
//判断shard_id是否在范围内，若是则返回0，若不是则返回pages_列表的大小-1的值
int64_t ShardHeader::GetLastPageId(const int &shard_id) {
  if (shard_id >= static_cast<int>(pages_.size())) {
    return 0;
  }
  return pages_[shard_id].size() - 1;
}
//引用ShardHeader空间中GetLastPageIdByType函数
int ShardHeader::GetLastPageIdByType(const int &shard_id, const std::string &page_type) {
  //判断shard_id是否在范围内，若是则返回0
  if (shard_id >= static_cast<int>(pages_.size())) {
    return 0;
  }
  int last_page_id = -1;
  //进入循环，若pages_列表中存在page_type则给last_page_id 赋值
  for (uint64_t i = pages_[shard_id].size(); i >= 1; i--) {
    if (pages_[shard_id][i - 1]->GetPageType() == page_type) {
      last_page_id = pages_[shard_id][i - 1]->GetPageID();
      return last_page_id;
    }
  }
  return last_page_id;
}
//引用ShardHeader空间中GetPageByGroupId函数
//若发现错误则输出错误信息
//进入循环，建立page_ptr链表
Status ShardHeader::GetPageByGroupId(const int &group_id, const int &shard_id, std::shared_ptr<Page> *page_ptr) {
  RETURN_UNEXPECTED_IF_NULL(page_ptr);
  CHECK_FAIL_RETURN_UNEXPECTED(shard_id < static_cast<int>(pages_.size()),
                               "[Internal ERROR] 'shard_id': " + std::to_string(shard_id) +
                                 " should be smaller than the size of 'pages_': " + std::to_string(pages_.size()) +
                                 ".");
  for (uint64_t i = pages_[shard_id].size(); i >= 1; i--) {
    auto page = pages_[shard_id][i - 1];
    if (page->GetPageType() == kPageTypeBlob && page->GetPageTypeID() == group_id) {
      *page_ptr = std::make_shared<Page>(*page);
      return Status::OK();
    }
  }
  page_ptr = nullptr;
  RETURN_STATUS_UNEXPECTED("[Internal ERROR] Failed to get Page, 'group_id': " + std::to_string(group_id));
}
//引用ShardHeader空间中AddSchema函数
int ShardHeader::AddSchema(std::shared_ptr<Schema> schema) {
  //判断schema列表是否为空，若是则返回错误信息
  if (schema == nullptr) {
    MS_LOG(ERROR) << "[Internal ERROR] The pointer of schema is NULL.";
    return -1;
  }
  //继续判断schema空间中的存储空间是否为空，若是则返回错误信息
  if (!schema_.empty()) {
    MS_LOG(ERROR) << "The schema is added repeatedly. Please remove the redundant 'add_schema' function.";
    return -1;
  }
  //判断schema_id是否为-1，若是则给schema_id再赋值，后加入到schema列表
  int64_t schema_id = schema->GetSchemaID();
  if (schema_id == -1) {
    schema_id = schema_.size();
    schema->SetSchemaID(schema_id);
  }
  schema_.push_back(schema);
  return schema_id;
}
//引用ShardHeader空间中AddStatistic函数
//判断statistic是否为真，若是则对statistic_id赋值，并将值列入statistic列表
void ShardHeader::AddStatistic(std::shared_ptr<Statistics> statistic) {
  if (statistic) {
    int64_t statistics_id = statistic->GetStatisticsID();
    if (statistics_id == -1) {
      statistics_id = statistics_.size();
      statistic->SetStatisticsID(statistics_id);
    }
    statistics_.push_back(statistic);
  }
}
//引用ShardHeader空间中InitIndexPtr函数
//判断index_是否为真，若是则进行操作
std::shared_ptr<Index> ShardHeader::InitIndexPtr() {
  std::shared_ptr<Index> index = index_;
  if (!index_) {
    index = std::make_shared<Index>();
    index_ = index;
  }
  return index;
}
//引用ShardHeader空间中CheckIndexField函数,输入参数n调用指定函数
Status ShardHeader::CheckIndexField(const std::string &field, const json &schema) {
  // check field name is or is not valid检查字段名称无效
  CHECK_FAIL_RETURN_UNEXPECTED(schema.find(field) != schema.end(),
                               "Invalid input, 'index_fields': " + field + " can not found in schema: " +
                                 schema.dump() + ".\n Please use 'add_index' function to add proper 'index_fields'.");
  CHECK_FAIL_RETURN_UNEXPECTED(schema[field]["type"] != "Bytes",
                               "Invalid input, type of 'index_fields': " + field +
                                 " is bytes and can not set as an 'index_fields'.\n Please use 'add_index' function to "
                                 "add the other 'index_fields'.");
  CHECK_FAIL_RETURN_UNEXPECTED(schema.find(field) == schema.end() || schema[field].find("shape") == schema[field].end(),
                               "Invalid input, type of 'index_fields': " + field +
                                 " is array and can not set as an 'index_fields'.\n Please use 'add_index' function to "
                                 "add the other 'index_fields'.");
  return Status::OK();
}
//引用ShardHeader空间中AddIndexFields函数,输入参数n调用指定函数
Status ShardHeader::AddIndexFields(const std::vector<std::string> &fields) {
  //判断fields是否为空
  if (fields.empty()) {
    return Status::OK();
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!GetSchemas().empty(),
                               "Invalid data, schema is empty. Please use 'add_schema' function to add schema first.");
  // create index Object创建索引对象
  std::shared_ptr<Index> index = InitIndexPtr();
  for (const auto &schemaPtr : schema_) {
    std::shared_ptr<Schema> schema_ptr;
    RETURN_IF_NOT_OK(GetSchemaByID(schemaPtr->GetSchemaID(), &schema_ptr));
    json schema = schema_ptr->GetSchema().at("schema");
    // checkout and add fields for each schema签出并为每个架构添加字段
    std::set<std::string> field_set;
    for (const auto &item : index->GetFields()) {
      field_set.insert(item.second);
    }
    for (const auto &field : fields) {
      CHECK_FAIL_RETURN_UNEXPECTED(
        field_set.find(field) == field_set.end(),
        "The 'index_fields': " + field + " is added repeatedly. Please remove the redundant 'add_index' function.");
      // check field name is or is not valid检查字段名称无效
      RETURN_IF_NOT_OK(CheckIndexField(field, schema));
      field_set.insert(field);
      // add field into index将字段添加到索引中
      index.get()->AddIndexField(schemaPtr->GetSchemaID(), field);
    }
  }
  index_ = index;
  return Status::OK();
}
//引用ShardHeader空间中GetAllSchemaID函数,输入参数n调用指定函数
Status ShardHeader::GetAllSchemaID(std::set<uint64_t> &bucket_count) {
  // get all schema id获取所有架构id
  for (const auto &schema : schema_) {
    auto schema_id = schema->GetSchemaID();
    CHECK_FAIL_RETURN_UNEXPECTED(bucket_count.find(schema_id) == bucket_count.end(),
                                 "[Internal ERROR] duplicate schema exist, schema id: " + std::to_string(schema_id));
    bucket_count.insert(schema_id);
  }
  return Status::OK();
}
//引用ShardHeader空间中AddIndexFields函数,输入参数n调用指定函数
Status ShardHeader::AddIndexFields(std::vector<std::pair<uint64_t, std::string>> fields) {
  //判断fields是否为空
  if (fields.empty()) {
    return Status::OK();
  }
  // create index Object创建索引对象
  std::shared_ptr<Index> index = InitIndexPtr();
  // get all schema id获取所有架构id
  std::set<uint64_t> bucket_count;
  RETURN_IF_NOT_OK(GetAllSchemaID(bucket_count));
  // check and add fields for each schema检查并添加每个模式的字段
  std::set<std::pair<uint64_t, std::string>> field_set;
  for (const auto &item : index->GetFields()) {
    field_set.insert(item);
  }
  for (const auto &field : fields) {
    CHECK_FAIL_RETURN_UNEXPECTED(field_set.find(field) == field_set.end(),
                                 "The 'index_fields': " + field.second +
                                   " is added repeatedly. Please remove the redundant 'add_index' function.");
    uint64_t schema_id = field.first;
    std::string field_name = field.second;

    // check schemaId is or is not valid check schemaId无效
    CHECK_FAIL_RETURN_UNEXPECTED(bucket_count.find(schema_id) != bucket_count.end(),
                                 "[Internal ERROR] 'schema_id': " + std::to_string(schema_id) + " can not found.");
    // check field name is or is not valid check field name无效
    std::shared_ptr<Schema> schema_ptr;
    RETURN_IF_NOT_OK(GetSchemaByID(schema_id, &schema_ptr));
    json schema = schema_ptr->GetSchema().at("schema");
    CHECK_FAIL_RETURN_UNEXPECTED(schema.find(field_name) != schema.end(),
                                 "Invalid input, 'index_fields': " + field_name + " can not found in schema: " +
                                   schema.dump() + ".\n Please use 'add_index' function to add proper 'index_fields'.");
    RETURN_IF_NOT_OK(CheckIndexField(field_name, schema));
    field_set.insert(field);
    // add field into index将字段添加到索引中
    index->AddIndexField(schema_id, field_name);
  }
  index_ = index;
  return Status::OK();
}
//引用ShardHeader空间中GetShardAddressByID函数,输入参数n调用指定函数
//判断shard_id是否大于shard_addresses_列表的大小，若是则返回“”，若不是则返回shard_addresses_列表中at函数的返回值
std::string ShardHeader::GetShardAddressByID(int64_t shard_id) {
  if (shard_id >= shard_addresses_.size()) {
    return "";
  }
  return shard_addresses_.at(shard_id);
}
//引用ShardHeader空间中GetSchemas函数,返回schema_的值
std::vector<std::shared_ptr<Schema>> ShardHeader::GetSchemas() { return schema_; }
//引用ShardHeader空间中GetStatistics函数,返回statistics_的值
std::vector<std::shared_ptr<Statistics>> ShardHeader::GetStatistics() { return statistics_; }
//引用ShardHeader空间中GetFields函数,返回index_中GetFields的函数返回值
std::vector<std::pair<uint64_t, std::string>> ShardHeader::GetFields() { return index_->GetFields(); }
//引用ShardHeader空间中GetIndex函数,返回index_的值
std::shared_ptr<Index> ShardHeader::GetIndex() { return index_; }
//引用ShardHeader空间中GetSchemaByID函数
//判断schema列表中id是否在标准范围内
Status ShardHeader::GetSchemaByID(int64_t schema_id, std::shared_ptr<Schema> *schema_ptr) {
  RETURN_UNEXPECTED_IF_NULL(schema_ptr);
  int64_t schema_size = schema_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(schema_id >= 0 && schema_id < schema_size,
                               "[Internal ERROR] 'schema_id': " + std::to_string(schema_id) + " is not in range [0, " +
                                 std::to_string(schema_size) + ").");
  *schema_ptr = schema_.at(schema_id);
  return Status::OK();
}
//引用ShardHeader空间中GetStatisticByID函数
//判断statistic列表中id的值是否在标准范围内
Status ShardHeader::GetStatisticByID(int64_t statistic_id, std::shared_ptr<Statistics> *statistics_ptr) {
  RETURN_UNEXPECTED_IF_NULL(statistics_ptr);
  int64_t statistics_size = statistics_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(statistic_id >= 0 && statistic_id < statistics_size,
                               "[Internal ERROR] 'statistic_id': " + std::to_string(statistic_id) +
                                 " is not in range [0, " + std::to_string(statistics_size) + ").");
  *statistics_ptr = statistics_.at(statistic_id);
  return Status::OK();
}
//在ShardHeader空间中创建PagesToFile函数
Status ShardHeader::PagesToFile(const std::string dump_file_name) {
  auto realpath = FileUtils::GetRealPath(dump_file_name.c_str());
  CHECK_FAIL_RETURN_UNEXPECTED(realpath.has_value(),
                               "[Internal ERROR] Failed to get the realpath of Pages file, path: " + dump_file_name);
  // write header content to file, dump whatever is in the file before将头内容写入文件，转储之前文件中的任何内容
  std::ofstream page_out_handle(realpath.value(), std::ios_base::trunc | std::ios_base::out);
  CHECK_FAIL_RETURN_UNEXPECTED(page_out_handle.good(),
                               "[Internal ERROR] Failed to open Pages file, path: " + dump_file_name);
  auto pages = SerializePage();
  for (const auto &shard_pages : pages) {
    page_out_handle << shard_pages << "\n";
  }
  page_out_handle.close();
  return Status::OK();
}
//在ShardHeader空间中创建FileToPages函数
Status ShardHeader::FileToPages(const std::string dump_file_name) {
  for (auto &v : pages_) {  // clean pages清理页面
    v.clear();
  }
  auto realpath = FileUtils::GetRealPath(dump_file_name.c_str());
  CHECK_FAIL_RETURN_UNEXPECTED(realpath.has_value(),
                               "[Internal ERROR] Failed to get the realpath of Pages file, path: " + dump_file_name);
  // attempt to open the file contains the page in json试图打开包含json页面的文件
  std::ifstream page_in_handle(realpath.value());
  CHECK_FAIL_RETURN_UNEXPECTED(page_in_handle.good(),
                               "[Internal ERROR] Pages file does not exist, path: " + dump_file_name);
  std::string line;
  while (std::getline(page_in_handle, line)) {
    RETURN_IF_NOT_OK(ParsePage(json::parse(line), -1, true));
  }
  page_in_handle.close();
  return Status::OK();
}
//在ShardHeader空间中创建Initialize函数
Status ShardHeader::Initialize(const std::shared_ptr<ShardHeader> *header_ptr, const json &schema,
                               const std::vector<std::string> &index_fields, std::vector<std::string> &blob_fields,
                               uint64_t &schema_id) {
  RETURN_UNEXPECTED_IF_NULL(header_ptr);
  auto schema_ptr = Schema::Build("mindrecord", schema);
  CHECK_FAIL_RETURN_UNEXPECTED(schema_ptr != nullptr, "[Internal ERROR] Failed to build schema: " + schema.dump() +
                                                        "." + "Check the [ERROR] logs before for more details.");
  schema_id = (*header_ptr)->AddSchema(schema_ptr);
  // create index创建索引
  std::vector<std::pair<uint64_t, std::string>> id_index_fields;
  if (!index_fields.empty()) {
    (void)transform(index_fields.begin(), index_fields.end(), std::back_inserter(id_index_fields),
                    [schema_id](const std::string &el) { return std::make_pair(schema_id, el); });
    RETURN_IF_NOT_OK((*header_ptr)->AddIndexFields(id_index_fields));
  }

  auto build_schema_ptr = (*header_ptr)->GetSchemas()[0];
  blob_fields = build_schema_ptr->GetBlobFields();
  return Status::OK();
}
}  // namespace mindrecord
}  // namespace mindspore
