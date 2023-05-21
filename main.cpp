// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "triton/core/tritonserver.h"
#include "triton/utils/server_wrapper.h"
#include "utils.h"

#ifdef TRITON_ENABLE_GPU

#include <cuda_runtime_api.h>

#endif  // TRITON_ENABLE_GPU

namespace ts = triton::server;
namespace ts_utils = triton::utils::server;

namespace {

bool enforce_memory_type = false;
ts_utils::MemoryType requested_memory_type;

#ifdef TRITON_ENABLE_GPU
// static auto cuda_data_deleter = [](void* data) {
//   if (data != nullptr) {
//     cudaPointerAttributes attr;
//     auto cuerr = cudaPointerGetAttributes(&attr, data);
//     if (cuerr != cudaSuccess) {
//       std::cerr << "error: failed to get CUDA pointer attribute of " << data
//                 << ": " << cudaGetErrorString(cuerr) << std::endl;
//     }
//     if (attr.type == cudaMemoryTypeDevice) {
//       cuerr = cudaFree(data);
//     } else if (attr.type == cudaMemoryTypeHost) {
//       cuerr = cudaFreeHost(data);
//     }
//     if (cuerr != cudaSuccess) {
//       std::cerr << "error: failed to release CUDA pointer " << data << ": "
//                 << cudaGetErrorString(cuerr) << std::endl;
//     }
//   }
// };
#endif  // TRITON_ENABLE_GPU

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-m <\"system\"|\"pinned\"|gpu>"
            << " Enforce the memory type for input and output tensors."
            << " If not specified, inputs will be in system memory and outputs"
            << " will be based on the model's preferred type." << std::endl;
  std::cerr << "\t-v Enable verbose logging" << std::endl;
  std::cerr << "\t-r [model repository absolute path]" << std::endl;

  exit(1);
}

// TRITONSERVER_Error*
// ResponseAlloc(
//     TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
//     size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
//     int64_t preferred_memory_type_id, void* userp, void** buffer,
//     void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
//     int64_t* actual_memory_type_id)
//{
//   // Initially attempt to make the actual memory type and id that we
//   // allocate be the same as preferred memory type
//   *actual_memory_type = preferred_memory_type;
//   *actual_memory_type_id = preferred_memory_type_id;
//
//   // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
//   // need to do any other book-keeping.
//   if (byte_size == 0) {
//     *buffer = nullptr;
//     *buffer_userp = nullptr;
//     std::cout << "allocated " << byte_size << " bytes for result tensor "
//               << tensor_name << std::endl;
//   } else {
//     void* allocated_ptr = nullptr;
//     if (enforce_memory_type) {
//       *actual_memory_type = requested_memory_type;
//     }
//
//     switch (*actual_memory_type) {
// #ifdef TRITON_ENABLE_GPU
//       case TRITONSERVER_MEMORY_CPU_PINNED: {
//         auto err = cudaSetDevice(*actual_memory_type_id);
//         if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
//             (err != cudaErrorInsufficientDriver)) {
//           return TRITONSERVER_ErrorNew(
//               TRITONSERVER_ERROR_INTERNAL,
//               std::string(
//                   "unable to recover current CUDA device: " +
//                   std::string(cudaGetErrorString(err)))
//                   .c_str());
//         }
//
//         err = cudaHostAlloc(&allocated_ptr, byte_size,
//         cudaHostAllocPortable); if (err != cudaSuccess) {
//           return TRITONSERVER_ErrorNew(
//               TRITONSERVER_ERROR_INTERNAL,
//               std::string(
//                   "cudaHostAlloc failed: " +
//                   std::string(cudaGetErrorString(err)))
//                   .c_str());
//         }
//         break;
//       }
//
//       case TRITONSERVER_MEMORY_GPU: {
//         auto err = cudaSetDevice(*actual_memory_type_id);
//         if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
//             (err != cudaErrorInsufficientDriver)) {
//           return TRITONSERVER_ErrorNew(
//               TRITONSERVER_ERROR_INTERNAL,
//               std::string(
//                   "unable to recover current CUDA device: " +
//                   std::string(cudaGetErrorString(err)))
//                   .c_str());
//         }
//
//         err = cudaMalloc(&allocated_ptr, byte_size);
//         if (err != cudaSuccess) {
//           return TRITONSERVER_ErrorNew(
//               TRITONSERVER_ERROR_INTERNAL,
//               std::string(
//                   "cudaMalloc failed: " +
//                   std::string(cudaGetErrorString(err))) .c_str());
//         }
//         break;
//       }
// #endif  // TRITON_ENABLE_GPU
//
//         // Use CPU memory if the requested memory type is unknown
//         // (default case).
//       case TRITONSERVER_MEMORY_CPU:
//       default: {
//         *actual_memory_type = TRITONSERVER_MEMORY_CPU;
//         allocated_ptr = malloc(byte_size);
//         break;
//       }
//     }
//
//     // Pass the tensor name with buffer_userp so we can show it when
//     // releasing the buffer.
//     if (allocated_ptr != nullptr) {
//       *buffer = allocated_ptr;
//       *buffer_userp = new std::string(tensor_name);
//       std::cout << "allocated " << byte_size << " bytes in "
//                 << TRITONSERVER_MemoryTypeString(*actual_memory_type)
//                 << " for result tensor " << tensor_name << std::endl;
//     }
//   }
//
//   return nullptr;  // Success
// }

// TRITONSERVER_Error*
// ResponseRelease(
//     TRITONSERVER_ResponseAllocator* allocator, void* buffer, void*
//     buffer_userp, size_t byte_size, TRITONSERVER_MemoryType memory_type,
//     int64_t memory_type_id)
//{
//   std::string* name = nullptr;
//   if (buffer_userp != nullptr) {
//     name = reinterpret_cast<std::string*>(buffer_userp);
//   } else {
//     name = new std::string("<unknown>");
//   }
//
//   std::cout << "Releasing buffer " << buffer << " of size " << byte_size
//             << " in " << TRITONSERVER_MemoryTypeString(memory_type)
//             << " for result '" << *name << "'" << std::endl;
//   switch (memory_type) {
//     case TRITONSERVER_MEMORY_CPU:
//       free(buffer);
//       break;
// #ifdef TRITON_ENABLE_GPU
//     case TRITONSERVER_MEMORY_CPU_PINNED: {
//       auto err = cudaSetDevice(memory_type_id);
//       if (err == cudaSuccess) {
//         err = cudaFreeHost(buffer);
//       }
//       if (err != cudaSuccess) {
//         std::cerr << "error: failed to cudaFree " << buffer << ": "
//                   << cudaGetErrorString(err) << std::endl;
//       }
//       break;
//     }
//     case TRITONSERVER_MEMORY_GPU: {
//       auto err = cudaSetDevice(memory_type_id);
//       if (err == cudaSuccess) {
//         err = cudaFree(buffer);
//       }
//       if (err != cudaSuccess) {
//         std::cerr << "error: failed to cudaFree " << buffer << ": "
//                   << cudaGetErrorString(err) << std::endl;
//       }
//       break;
//     }
// #endif  // TRITON_ENABLE_GPU
//     default:
//       std::cerr << "error: unexpected buffer allocated in CUDA managed
//       memory"
//                 << std::endl;
//       break;
//   }
//
//   delete name;
//
//   return nullptr;  // Success
// }
//
// void
// InferRequestComplete(
//     TRITONSERVER_InferenceRequest* request, const uint32_t flags, void*
//     userp)
//{
//   // We reuse the request so we don't delete it here.
// }
//
// void
// InferResponseComplete(
//     TRITONSERVER_InferenceResponse* response, const uint32_t flags, void*
//     userp)
//{
//   if (response != nullptr) {
//     // Send 'response' to the future.
//     std::promise<TRITONSERVER_InferenceResponse*>* p =
//         reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
//     p->set_value(response);
//     delete p;
//   }
// }

TRITONSERVER_Error*
ParseModelMetadata(
    const rapidjson::Document& model_metadata, bool* is_int,
    bool* is_torch_model)
{
  std::string seen_data_type;
  for (const auto& input : model_metadata["inputs"].GetArray()) {
    if (strcmp(input["datatype"].GetString(), "INT32") != 0 &&
        strcmp(input["datatype"].GetString(), "FP32") != 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "simple lib example only supports model with data type INT32 or "
          "FP32");
    }
    if (seen_data_type.empty()) {
      seen_data_type = input["datatype"].GetString();
    } else if (
        strcmp(seen_data_type.c_str(), input["datatype"].GetString()) != 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "the inputs and outputs of 'simple' model must have the data type");
    }
  }
  for (const auto& output : model_metadata["outputs"].GetArray()) {
    if (strcmp(output["datatype"].GetString(), "INT32") != 0 &&
        strcmp(output["datatype"].GetString(), "FP32") != 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "simple lib example only supports model with data type INT32 or "
          "FP32");
    } else if (
        strcmp(seen_data_type.c_str(), output["datatype"].GetString()) != 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "the inputs and outputs of 'simple' model must have the data type");
    }
  }

  *is_int = (strcmp(seen_data_type.c_str(), "INT32") == 0);
  *is_torch_model =
      (strcmp(model_metadata["platform"].GetString(), "pytorch_libtorch") == 0);
  return nullptr;
}

template <typename T>
void
GenerateInputData(
    std::vector<char>* input0_data, std::vector<char>* input1_data)
{
  input0_data->resize(16 * sizeof(T));
  input1_data->resize(16 * sizeof(T));
  for (size_t i = 0; i < 16; ++i) {
    ((T*)input0_data->data())[i] = i;
    ((T*)input1_data->data())[i] = 1;
  }
}

template <typename T>
void
CompareResult(
    const std::string& output0_name, const std::string& output1_name,
    const void* input0, const void* input1, const char* output0,
    const char* output1)
{
  for (size_t i = 0; i < 16; ++i) {
    std::cout << ((T*)input0)[i] << " + " << ((T*)input1)[i] << " = "
              << ((T*)output0)[i] << std::endl;
    std::cout << ((T*)input0)[i] << " - " << ((T*)input1)[i] << " = "
              << ((T*)output1)[i] << std::endl;

    if ((((T*)input0)[i] + ((T*)input1)[i]) != ((T*)output0)[i]) {
      FAIL("incorrect sum in " + output0_name);
    }
    if ((((T*)input0)[i] - ((T*)input1)[i]) != ((T*)output1)[i]) {
      FAIL("incorrect difference in " + output1_name);
    }
  }
}

void
Check(
    std::shared_ptr<ts_utils::Tensor>& output0,
    std::shared_ptr<ts_utils::Tensor>& output1,
    const std::vector<char>& input0_data, const std::vector<char>& input1_data,
    const std::string& output0_name, const std::string& output1_name,
    const size_t expected_byte_size, const ts_utils::DataType expected_datatype,
    const std::string& model_name, const bool is_custom_alloc)
{
  std::unordered_map<std::string, std::vector<char>> output_data;
  for (auto& output :
       {std::make_pair(output0_name, output0),
        std::make_pair(output1_name, output1)}) {
    if (model_name == "add_sub") {
      if ((output.second->shape_.size() != 1) ||
          (output.second->shape_[0] != 16)) {
        FAIL("unexpected shape for '" + output.first + "'");
      }
    } else if (model_name == "simple") {
      if ((output.second->shape_.size() != 2) ||
          (output.second->shape_[0] != 1) || (output.second->shape_[1] != 16)) {
        FAIL("unexpected shape for '" + output.first + "'");
      }
    } else {
      FAIL("unexpected model name '" + model_name + "'");
    }

    if (output.second->data_type_ != expected_datatype) {
      FAIL(
          "unexpected datatype '" +
          std::string(DataTypeString(output.second->data_type_)) + "' for '" +
          output.first + "'");
    }

    if (output.second->byte_size_ != expected_byte_size) {
      FAIL(
          "unexpected byte-size, expected " +
          std::to_string(expected_byte_size) + ", got " +
          std::to_string(output.second->byte_size_) + " for " + output.first);
    }

    // For this example, we use default allocator and pre-allocated buffer in
    // the first and second infer requests, so the memory type for both cases
    // should be 'CPU'.
    if (is_custom_alloc) {
      if (enforce_memory_type &&
          (output.second->memory_type_ != requested_memory_type)) {
        FAIL(
            "unexpected memory type, expected to be allocated in " +
            std::string(MemoryTypeString(requested_memory_type)) + ", got " +
            std::string(MemoryTypeString(output.second->memory_type_)) +
            ", id " + std::to_string(output.second->memory_type_id_) + " for " +
            output.first);
      }
    } else {
      if (output.second->memory_type_ != ts_utils::MemoryType::CPU) {
        FAIL(
            "unexpected memory type, expected to be allocated in CPU, got " +
            std::string(MemoryTypeString(output.second->memory_type_)) +
            ", id " + std::to_string(output.second->memory_type_id_) + " for " +
            output.first);
      }
    }

    // We make a copy of the data here... which we could avoid for
    // performance reasons but ok for this simple example.
    std::vector<char>& odata = output_data[output.first];
    switch (output.second->memory_type_) {
      case ts_utils::MemoryType::CPU: {
        std::cout << output.first << " is stored in system memory" << std::endl;
        odata.assign(
            output.second->buffer_,
            output.second->buffer_ + output.second->byte_size_);
        break;
      }

      case ts_utils::MemoryType::CPU_PINNED: {
        std::cout << output.first << " is stored in pinned memory" << std::endl;
        odata.assign(
            output.second->buffer_,
            output.second->buffer_ + output.second->byte_size_);
        break;
      }

#ifdef TRITON_ENABLE_GPU
      case ts_utils::MemoryType::GPU: {
        std::cout << output.first << " is stored in GPU memory" << std::endl;
        odata.reserve(output.second->byte_size_);
        FAIL_IF_CUDA_ERR(
            cudaMemcpy(
                &odata[0], output.second->buffer_, output.second->byte_size_,
                cudaMemcpyDeviceToHost),
            "getting " + output.first + " data from GPU memory");
        break;
      }
#endif

      default:
        FAIL("unexpected memory type");
    }
  }

  CompareResult<int32_t>(
      output0_name, output1_name, &input0_data[0], &input1_data[0],
      output_data[output0_name].data(), output_data[output1_name].data());
}


}  // namespace

int
main(int argc, char** argv)
{
  std::string model_repository_path;
  int verbose_level = 0;

  // Parse commandline...
  // TODO, maybe need another modern commandline parser
  int opt;
  while ((opt = getopt(argc, argv, "vm:r:")) != -1) {
    switch (opt) {
      case 'm': {
        enforce_memory_type = true;
        if (!strcmp(optarg, "system")) {
          // requested_memory_type = TRITONSERVER_MEMORY_CPU;
          requested_memory_type = ts_utils::MemoryType::CPU;
        } else if (!strcmp(optarg, "pinned")) {
          // requested_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
          requested_memory_type = ts_utils::MemoryType::CPU_PINNED;
        } else if (!strcmp(optarg, "gpu")) {
          // requested_memory_type = TRITONSERVER_MEMORY_GPU;
          requested_memory_type = ts_utils::MemoryType::GPU;
        } else {
          Usage(
              argv,
              "-m must be used to specify one of the following types:"
              " <\"system\"|\"pinned\"|gpu>");
        }
        break;
      }
      case 'r':
        model_repository_path = optarg;
        break;
      case 'v':
        verbose_level = 1;
        break;
      case '?':
        Usage(argv);
        break;
      default:
        break;
    }
  }

  if (model_repository_path.empty()) {
    Usage(argv, "-r must be used to specify model repository path");
  }
#ifndef TRITON_ENABLE_GPU
  if (enforce_memory_type && requested_memory_type != tus::MemoryType::CPU) {
    Usage(argv, "-m can only be set to \"system\" without enabling GPU");
  }
#endif  // TRITON_ENABLE_GPU
  try {
    // Check API version. This compares the API version of the
    // triton-server library linked into this application against the
    // API version of the header file used when compiling this
    // application. The API version of the shared library must be >= the
    // API version used when compiling this application.
    //  uint32_t api_version_major, api_version_minor;
    //  FAIL_IF_ERR(
    //      TRITONSERVER_ApiVersion(&api_version_major, &api_version_minor),
    //      "getting Triton API version");
    //  if ((TRITONSERVER_API_VERSION_MAJOR != api_version_major) ||
    //      (TRITONSERVER_API_VERSION_MINOR > api_version_minor)) {
    //    FAIL("triton server API version mismatch");
    //  }
    ts_utils::ServerOptions options({model_repository_path});
    options.logging_.verbose_ =
        ts_utils::LoggingOptions::VerboseLevel(verbose_level);
    //  options.model_control_mode_ =
    //  triton_server_utils::ModelControlMode::EXPLICIT;

    options.trace_ = std::make_shared<ts_utils::Trace>(
        "trace_file", ts_utils::Trace::Level::TIMESTAMPS, 1, -1, 0);
    options.backend_dir_ = "/opt/tritonserver/backends";
    options.repo_agent_dir_ = "/opt/tritonserver/repoagents";
    options.strict_model_config_ = true;

#ifdef TRITON_ENABLE_GPU
    double min_compute_capability = TRITON_MIN_COMPUTE_CAPABILITY;
#else
    double min_compute_capability = 0;
#endif  // TRITON_ENABLE_GPU
    options.min_cuda_compute_capability_ = min_compute_capability;
    // Create the server object
    auto server = ts_utils::TritonServer::Create(options);
    // Wait until the server is both live and ready. The server will not
    // appear "ready" until all models are loaded and ready to receive
    // inference requests.
    size_t health_iters = 0;
    while (true) {
      auto live = server->IsServerLive();
      auto ready = server->IsServerReady();
      std::cout << "Server Health: live " << live << ", ready " << ready
                << std::endl;
      if (live && ready) {
        break;
      }
      if (++health_iters >= 10) {
        FAIL("failed to find healthy inference server");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Server metadata can be accessed using the server object. The
    // metadata is returned as an abstract TRITONSERVER_Message that can
    // be converted to JSON for further processing.
    {
      auto metadata = server->ServerMetadata();
      std::cout << "Server Metadata:" << std::endl;
      std::cout << metadata << std::endl;
    }

    const std::string model_name("simple");

    // We already waited for the server to be ready, above, so we know
    // that all models are also ready. But as an example we also wait
    // for a specific model to become available.
    bool is_torch_model = false;
    bool is_int = true;
    bool is_ready = false;
    health_iters = 0;
    while (!is_ready) {
      is_ready = server->IsModelReady(model_name, 1 /* model_version */);
      //    FAIL_IF_ERR(
      //        TRITONSERVER_ServerModelIsReady(
      //            server.get(), model_name.c_str(), 1 /* model_version */,
      //            &is_ready),
      //        "unable to get model readiness");
      if (!is_ready) {
        if (++health_iters >= 10) {
          FAIL("model failed to be ready in 10 iterations");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        continue;
      }

      auto json_model_metadata =
          server->ModelMetadata(model_name, 1 /* model_version */);
      //    TRITONSERVER_Message* model_metadata_message;
      //    FAIL_IF_ERR(
      //        TRITONSERVER_ServerModelMetadata(
      //            server.get(), model_name.c_str(), 1,
      //            &model_metadata_message),
      //        "unable to get model metadata message");
      //    const char* buffer;
      //    size_t byte_size;
      //    FAIL_IF_ERR(
      //        TRITONSERVER_MessageSerializeToJson(
      //            model_metadata_message, &buffer, &byte_size),
      //        "unable to serialize model metadata");

      // Parse the JSON string that represents the model metadata into a
      // JSON document. We use rapidjson for this parsing but any JSON
      // parser can be used.
      rapidjson::Document model_metadata;
      model_metadata.Parse(
          json_model_metadata.data(), json_model_metadata.length());
      if (model_metadata.HasParseError()) {
        FAIL(
            "error: failed to parse model metadata from JSON: " +
            std::string(GetParseError_En(model_metadata.GetParseError())) +
            " at " + std::to_string(model_metadata.GetErrorOffset()));
      }

      //    FAIL_IF_ERR(
      //        TRITONSERVER_MessageDelete(model_metadata_message),
      //        "deleting model metadata message");

      // Now that we have a document representation of the model
      // metadata, we can query it to extract some information about the
      // model.
      if (strcmp(model_metadata["name"].GetString(), model_name.c_str()) != 0) {
        FAIL("unable to find metadata for model");
      }

      bool found_version = false;
      if (model_metadata.HasMember("versions")) {
        for (const auto& version : model_metadata["versions"].GetArray()) {
          if (strcmp(version.GetString(), "1") == 0) {
            found_version = true;
            break;
          }
        }
      }
      if (!found_version) {
        FAIL("unable to find version 1 status for model");
      }

      FAIL_IF_ERR(
          ParseModelMetadata(model_metadata, &is_int, &is_torch_model),
          "parsing model metadata");
    }
    // Create the inference request on "simple" model
    auto request_1 =
        ts_utils::InferRequest::Create(ts_utils::InferOptions("simple"));
    // Add two input tensor to the request. The first is a 16-element
    std::vector<char> input0_data;
    std::vector<char> input1_data;
    GenerateInputData<int32_t>(&input0_data, &input1_data);
    size_t input0_size = input0_data.size();
    size_t input1_size = input1_data.size();

    ts_utils::Tensor input0(
        input0_data.data(), input0_data.size(), ts_utils::DataType::INT32,
        {1, 16}, ts_utils::MemoryType::CPU, 0);
    ts_utils::Tensor input1(
        input1_data.data(), input1_data.size(), ts_utils::DataType::INT32,
        {1, 16}, ts_utils::MemoryType::CPU, 0);

    request_1->AddInput("INPUT0", input0);
    request_1->AddInput("INPUT1", input1);

    request_1->AddRequestedOutput("OUTPUT0");
    request_1->AddRequestedOutput("OUTPUT1");

    auto result_future_1 = server->AsyncInfer(*request_1);

    // Get the result from the future and check the result
    auto result_1 = result_future_1.get();
    if (result_1->HasError()) {
      FAIL(result_1->ErrorMsg());
    }
    std::cout << "Ran inference on model '" << result_1->ModelName()
              << "', version '" << result_1->ModelVersion()
              << "', with request ID'" << result_1->Id() << "'\n";

    // Retrieve two outputs from InferResult object
    std::shared_ptr<ts_utils::Tensor> result_1_output_0 =
        result_1->Output("OUTPUT0");
    std::shared_ptr<ts_utils::Tensor> result_1_output_1 =
        result_1->Output("OUTPUT1");

    Check(
        result_1_output_0, result_1_output_1, input0_data, input1_data,
        "OUTPUT0", "OUTPUT1", input0_size, ts_utils::DataType::INT32,
        result_1->ModelName(), false);

    std::cout << result_1->DebugString() << std::endl;

    // Get the server metric
    auto metric = server->ServerMetrics();
    std::cout << "\n\n\n==================Server Metrics==================\n"
              << metric << std::endl;


    //
    //  // When triton needs a buffer to hold an output tensor, it will ask
    //  // us to provide the buffer. In this way we can have any buffer
    //  // management and sharing strategy that we want. To communicate to
    //  // triton the functions that we want it to call to perform the
    //  // allocations, we create a "response allocator" object. We pass
    //  // this response allocate object to triton when requesting
    //  // inference. We can reuse this response allocate object for any
    //  // number of inference requests.
    //  TRITONSERVER_ResponseAllocator* allocator = nullptr;
    //  FAIL_IF_ERR(
    //      TRITONSERVER_ResponseAllocatorNew(
    //          &allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn
    //          */),
    //      "creating response allocator");
    //
    //  // Create an inference request object. The inference request object
    //  // is where we set the name of the model we want to use for
    //  // inference and the input tensors.
    //  TRITONSERVER_InferenceRequest* irequest = nullptr;
    //  FAIL_IF_ERR(
    //      TRITONSERVER_InferenceRequestNew(
    //          &irequest, server.get(), model_name.c_str(), -1 /* model_version
    //          */),
    //      "creating inference request");
    //
    //  FAIL_IF_ERR(
    //      TRITONSERVER_InferenceRequestSetId(irequest, "my_request_id"),
    //      "setting ID for the request");
    //
    //  FAIL_IF_ERR(
    //      TRITONSERVER_InferenceRequestSetReleaseCallback(
    //          irequest, InferRequestComplete, nullptr /* request_release_userp
    //          */),
    //      "setting request release callback");
    //
    //  // Add the 2 input tensors to the request...
    //  auto input0 = "INPUT0";
    //  auto input1 = "INPUT1";
    //
    //  std::vector<int64_t> input0_shape({1, 16});
    //  std::vector<int64_t> input1_shape({1, 16});
    //
    //  const TRITONSERVER_DataType datatype =
    //      (is_int) ? TRITONSERVER_TYPE_INT32 : TRITONSERVER_TYPE_FP32;
    //
    //  FAIL_IF_ERR(
    //      TRITONSERVER_InferenceRequestAddInput(
    //          irequest, input0, datatype, &input0_shape[0],
    //          input0_shape.size()),
    //      "setting input 0 meta-data for the request");
    //  FAIL_IF_ERR(
    //      TRITONSERVER_InferenceRequestAddInput(
    //          irequest, input1, datatype, &input1_shape[0],
    //          input1_shape.size()),
    //      "setting input 1 meta-data for the request");
    //
    //  auto output0 = is_torch_model ? "OUTPUT__0" : "OUTPUT0";
    //  auto output1 = is_torch_model ? "OUTPUT__1" : "OUTPUT1";
    //
    //  // Indicate that we want both output tensors calculated and returned
    //  // for the inference request. These calls are optional, if no
    //  // output(s) are specifically requested then all outputs defined by
    //  // the model will be calculated and returned.
    //  FAIL_IF_ERR(
    //      TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output0),
    //      "requesting output 0 for the request");
    //  FAIL_IF_ERR(
    //      TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output1),
    //      "requesting output 1 for the request");
    //
    //  // Create the data for the two input tensors. Initialize the first
    //  // to unique values and the second to all ones.
    //  std::vector<char> input0_data;
    //  std::vector<char> input1_data;
    //  if (is_int) {
    //    GenerateInputData<int32_t>(&input0_data, &input1_data);
    //  } else {
    //    GenerateInputData<float>(&input0_data, &input1_data);
    //  }
    //
    //  size_t input0_size = input0_data.size();
    //  size_t input1_size = input1_data.size();
    //
    //  const void* input0_base = &input0_data[0];
    //  const void* input1_base = &input1_data[0];
    // #ifdef TRITON_ENABLE_GPU
    //  std::unique_ptr<void, decltype(cuda_data_deleter)> input0_gpu(
    //      nullptr, cuda_data_deleter);
    //  std::unique_ptr<void, decltype(cuda_data_deleter)> input1_gpu(
    //      nullptr, cuda_data_deleter);
    //  bool use_cuda_memory =
    //      (enforce_memory_type &&
    //       (requested_memory_type != ts_utils::MemoryType::CPU));
    //  if (use_cuda_memory) {
    //    FAIL_IF_CUDA_ERR(cudaSetDevice(0), "setting CUDA device to device 0");
    //    if (requested_memory_type != ts_utils::MemoryType::CPU_PINNED) {
    //      void* dst;
    //      FAIL_IF_CUDA_ERR(
    //          cudaMalloc(&dst, input0_size),
    //          "allocating GPU memory for INPUT0 data");
    //      input0_gpu.reset(dst);
    //      FAIL_IF_CUDA_ERR(
    //          cudaMemcpy(dst, &input0_data[0], input0_size,
    //          cudaMemcpyHostToDevice), "setting INPUT0 data in GPU memory");
    //      FAIL_IF_CUDA_ERR(
    //          cudaMalloc(&dst, input1_size),
    //          "allocating GPU memory for INPUT1 data");
    //      input1_gpu.reset(dst);
    //      FAIL_IF_CUDA_ERR(
    //          cudaMemcpy(dst, &input1_data[0], input1_size,
    //          cudaMemcpyHostToDevice), "setting INPUT1 data in GPU memory");
    //    } else {
    //      void* dst;
    //      FAIL_IF_CUDA_ERR(
    //          cudaHostAlloc(&dst, input0_size, cudaHostAllocPortable),
    //          "allocating pinned memory for INPUT0 data");
    //      input0_gpu.reset(dst);
    //      FAIL_IF_CUDA_ERR(
    //          cudaMemcpy(dst, &input0_data[0], input0_size,
    //          cudaMemcpyHostToHost), "setting INPUT0 data in pinned memory");
    //      FAIL_IF_CUDA_ERR(
    //          cudaHostAlloc(&dst, input1_size, cudaHostAllocPortable),
    //          "allocating pinned memory for INPUT1 data");
    //      input1_gpu.reset(dst);
    //      FAIL_IF_CUDA_ERR(
    //          cudaMemcpy(dst, &input1_data[0], input1_size,
    //          cudaMemcpyHostToHost), "setting INPUT1 data in pinned memory");
    //    }
    //  }
    //
    //  input0_base = use_cuda_memory ? input0_gpu.get() : &input0_data[0];
    //  input1_base = use_cuda_memory ? input1_gpu.get() : &input1_data[0];
    // #endif  // TRITON_ENABLE_GPU
    //
    //  FAIL_IF_ERR(
    //      TRITONSERVER_InferenceRequestAppendInputData(
    //          irequest, input0, input0_base, input0_size,
    //          requested_memory_type, 0 /* memory_type_id */),
    //      "assigning INPUT0 data");
    //  FAIL_IF_ERR(
    //      TRITONSERVER_InferenceRequestAppendInputData(
    //          irequest, input1, input1_base, input1_size,
    //          requested_memory_type, 0 /* memory_type_id */),
    //      "assigning INPUT1 data");
    //
    //  // Perform inference by calling TRITONSERVER_ServerInferAsync. This
    //  // call is asychronous and therefore returns immediately. The
    //  // completion of the inference and delivery of the response is done
    //  // by triton by calling the "response complete" callback functions
    //  // (InferResponseComplete in this case).
    //  {
    //    auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
    //    std::future<TRITONSERVER_InferenceResponse*> completed =
    //    p->get_future();
    //
    //    FAIL_IF_ERR(
    //        TRITONSERVER_InferenceRequestSetResponseCallback(
    //            irequest, allocator, nullptr /* response_allocator_userp */,
    //            InferResponseComplete, reinterpret_cast<void*>(p)),
    //        "setting response callback");
    //
    //    FAIL_IF_ERR(
    //        TRITONSERVER_ServerInferAsync(
    //            server.get(), irequest, nullptr /* trace */),
    //        "running inference");
    //
    //    // The InferResponseComplete function sets the std::promise so
    //    // that this thread will block until the response is returned.
    //    TRITONSERVER_InferenceResponse* completed_response = completed.get();
    //
    //    FAIL_IF_ERR(
    //        TRITONSERVER_InferenceResponseError(completed_response),
    //        "response status");
    //
    //    Check(
    //        completed_response, input0_data, input1_data, output0, output1,
    //        input0_size, datatype, is_int);
    //
    //    FAIL_IF_ERR(
    //        TRITONSERVER_InferenceResponseDelete(completed_response),
    //        "deleting inference response");
    //  }
    //
    //  // The TRITONSERVER_InferenceRequest object can be reused for
    //  // multiple (sequential) inference requests. For example, if we have
    //  // multiple requests where the inference request is the same except
    //  // for different input tensor data, then we can just change the
    //  // input data buffers. Below some input data is changed in place and
    //  // then another inference request is issued. For simplicity we only
    //  // do this when the input tensors are in non-pinned system memory.
    //  if (!enforce_memory_type ||
    //      (requested_memory_type == ts_utils::MemoryType::CPU)) {
    //    if (is_int) {
    //      int32_t* input0_base = reinterpret_cast<int32_t*>(&input0_data[0]);
    //      input0_base[0] = 27;
    //    } else {
    //      float* input0_base = reinterpret_cast<float*>(&input0_data[0]);
    //      input0_base[0] = 27.0;
    //    }
    //
    //    auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
    //    std::future<TRITONSERVER_InferenceResponse*> completed =
    //    p->get_future();
    //
    //    // Using a new promise so have to re-register the callback to set
    //    // the promise as the userp.
    //    FAIL_IF_ERR(
    //        TRITONSERVER_InferenceRequestSetResponseCallback(
    //            irequest, allocator, nullptr /* response_allocator_userp */,
    //            InferResponseComplete, reinterpret_cast<void*>(p)),
    //        "setting response callback");
    //
    //    FAIL_IF_ERR(
    //        TRITONSERVER_ServerInferAsync(
    //            server.get(), irequest, nullptr /* trace */),
    //        "running inference");
    //
    //    TRITONSERVER_InferenceResponse* completed_response = completed.get();
    //    FAIL_IF_ERR(
    //        TRITONSERVER_InferenceResponseError(completed_response),
    //        "response status");
    //
    //    Check(
    //        completed_response, input0_data, input1_data, output0, output1,
    //        input0_size, datatype, is_int);
    //
    //    FAIL_IF_ERR(
    //        TRITONSERVER_InferenceResponseDelete(completed_response),
    //        "deleting inference response");
    //  }
    //
    //  // There are other TRITONSERVER_InferenceRequest APIs that allow
    //  // other in-place modifications so that the object can be reused for
    //  // multiple (sequential) inference requests. For example, we can
    //  // assign a new data buffer for an input by first removing the
    //  // existing data with
    //  // TRITONSERVER_InferenceRequestRemoveAllInputData.
    //  {
    //    FAIL_IF_ERR(
    //        TRITONSERVER_InferenceRequestRemoveAllInputData(irequest, input0),
    //        "removing INPUT0 data");
    //    FAIL_IF_ERR(
    //        TRITONSERVER_InferenceRequestAppendInputData(
    //            irequest, input0, input1_base, input1_size,
    //            requested_memory_type, 0 /* memory_type_id */),
    //        "assigning INPUT1 data to INPUT0");
    //
    //    auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
    //    std::future<TRITONSERVER_InferenceResponse*> completed =
    //    p->get_future();
    //
    //    // Using a new promise so have to re-register the callback to set
    //    // the promise as the userp.
    //    FAIL_IF_ERR(
    //        TRITONSERVER_InferenceRequestSetResponseCallback(
    //            irequest, allocator, nullptr /* response_allocator_userp */,
    //            InferResponseComplete, reinterpret_cast<void*>(p)),
    //        "setting response callback");
    //
    //    FAIL_IF_ERR(
    //        TRITONSERVER_ServerInferAsync(
    //            server.get(), irequest, nullptr /* trace */),
    //        "running inference");
    //
    //    TRITONSERVER_InferenceResponse* completed_response = completed.get();
    //    FAIL_IF_ERR(
    //        TRITONSERVER_InferenceResponseError(completed_response),
    //        "response status");
    //
    //    // Both inputs are using input1_data...
    //    Check(
    //        completed_response, input1_data, input1_data, output0, output1,
    //        input0_size, datatype, is_int);
    //
    //    FAIL_IF_ERR(
    //        TRITONSERVER_InferenceResponseDelete(completed_response),
    //        "deleting inference response");
    //  }
    //
    //  FAIL_IF_ERR(
    //      TRITONSERVER_InferenceRequestDelete(irequest),
    //      "deleting inference request");
    //
    //  FAIL_IF_ERR(
    //      TRITONSERVER_ResponseAllocatorDelete(allocator),
    //      "deleting response allocator");
  }
  catch (const ts_utils::TritonException& ex) {
    std::cerr << "Error: " << ex.what();
    exit(1);
  }
  return 0;
}
