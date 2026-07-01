// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// vision_flexmlrt.cpp — MODIFIED VERSION for CPU preprocessing
//
// This version accepts CPU-preprocessed [1073, 4, 1280] input instead of raw
// pixel_values

#include <FlexMLClient.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// Debug logging gated by VLLM_LOGGING_LEVEL=DEBUG
inline bool is_vllm_debug() {
  static int debug_enabled = -1;
  if (debug_enabled == -1) {
    const char* level = std::getenv("VLLM_LOGGING_LEVEL");
    debug_enabled = (level && std::strcmp(level, "DEBUG") == 0) ? 1 : 0;
  }
  return debug_enabled == 1;
}

// Use stderr (not PySys_WriteStdout) so logging is safe while the GIL is
// released during model_->forward().
#define DEBUG_LOG(expr)               \
  do {                                \
    if (is_vllm_debug()) {            \
      std::ostringstream oss;         \
      oss << "[FlexMLRT] " << expr;   \
      std::cerr << oss.str() << '\n'; \
    }                                 \
  } while (0)

// Build ErtIoTypeNew tensor descriptor
static flexmlrt::client::ErtIoTypeNew makeIO(
    const std::string& name, int index, void* data, size_t size_bytes,
    const std::string& dtype, const std::vector<int64_t>& shape) {
  flexmlrt::client::ErtIoTypeNew io;
  io.name = name;
  io.idx = index;
  io.data = data;
  io.size = size_bytes;
  io.type = dtype;
  io.shape = shape;
  return io;
}

// VisionFlexMLRTModel with CPU preprocessing support
class VisionFlexMLRTModel {
 public:
  VisionFlexMLRTModel(const std::string& model_cache,
                      const std::string& device_name)
      : device_name_(device_name) {
    DEBUG_LOG(" VisionFlexMLRTModel constructor START");
    DEBUG_LOG("   model_cache: " << model_cache);
    DEBUG_LOG("   device_name: " << device_name);

    // Create options object (will be destroyed after model creation)
    flexmlrt::client::Options opts;
    opts.modelPath = model_cache;
    opts.deviceName = device_name;
    opts.subgraphName = "0";  // Specify subgraph name explicitly
    opts.executeMode = 2;     // From test_generic line 446

    DEBUG_LOG(" Creating FlexMLRT Model object...");
    try {
      model_ = std::make_unique<flexmlrt::client::Model>(opts);
      DEBUG_LOG(" FlexMLRT Model object created");
    } catch (const std::exception& e) {
      std::cerr << "[FlexMLRT ERROR] FlexMLRT Model creation threw exception: "
                << e.what() << std::endl;
      throw std::runtime_error(
          std::string("Failed to load FlexMLRT vision model: ") + e.what());
    }
    // opts goes out of scope here - memory automatically freed

    if (!model_->good()) {
      std::cerr << "[FlexMLRT ERROR] model->good() returned false" << std::endl;
      throw std::runtime_error(
          "FlexMLRT vision model creation failed - check model cache and "
          "device availability");
    }
    DEBUG_LOG(" model->good() returned true");
    DEBUG_LOG(" VisionFlexMLRTModel constructor END (opts memory released)");
  }

  // Generic forward pass: the caller supplies the NPU partition's input/output
  // tensor names and the output shape (read from the cache's own IO spec), so
  // this shim works for ANY model — Qwen2.5-VL, MiniCPM-V, Gemma-3, ...
  // The input array's own shape is used verbatim (any ndim is accepted).
  py::array_t<float> forward(py::array_t<float> input,
                             const std::string& input_name,
                             const std::string& output_name,
                             const std::vector<int64_t>& output_shape) {
    DEBUG_LOG(" forward() START");

    auto buf = input.request();
    std::vector<int64_t> in_shape(buf.shape.begin(), buf.shape.end());
    const size_t in_bytes = static_cast<size_t>(buf.size) * sizeof(float);
    DEBUG_LOG(" Input '" << input_name << "' ndim=" << buf.ndim
                         << " numel=" << buf.size);

    // Build input tensor from the array's actual shape.
    std::vector<flexmlrt::client::ErtIoTypeNew> ifms;
    ifms.push_back(
        makeIO(input_name, 0, buf.ptr, in_bytes, "float32", in_shape));

    // Build output tensor sized by the caller-provided output shape.
    int64_t out_numel = 1;
    for (int64_t d : output_shape) out_numel *= d;
    if (out_numel <= 0) {
      throw std::runtime_error("output_shape must have positive dimensions");
    }
    std::vector<float> output_buf(static_cast<size_t>(out_numel));
    std::vector<flexmlrt::client::ErtIoTypeNew> ofms;
    ofms.push_back(makeIO(output_name, 0, output_buf.data(),
                          output_buf.size() * sizeof(float), "float32",
                          output_shape));
    DEBUG_LOG(" Output '" << output_name << "' numel=" << out_numel);

    std::vector<flexmlrt::client::ErtIoTypeNew> wts;

    // Run NPU inference (release the GIL so the GPU LLM can run in parallel).
    DEBUG_LOG(" Calling model->forward()...");
    try {
      py::gil_scoped_release release;
      model_->forward(ifms, ofms, wts);
      DEBUG_LOG(" model->forward() returned successfully (GIL reacquired)");
    } catch (const std::exception& e) {
      std::cerr << "[FlexMLRT ERROR] model->forward() threw exception: "
                << e.what() << std::endl;
      throw std::runtime_error(std::string("FlexMLRT forward failed: ") +
                               e.what());
    }

    // Copy output into a numpy array of the requested shape.
    py::array_t<float> result(output_shape);
    auto result_buf = result.request();
    std::memcpy(result_buf.ptr, output_buf.data(),
                output_buf.size() * sizeof(float));

    output_buf.clear();
    output_buf.shrink_to_fit();
    ifms.clear();
    ofms.clear();
    DEBUG_LOG(" forward() END");
    return result;
  }

 private:
  std::unique_ptr<flexmlrt::client::Model> model_;
  std::string device_name_;
  // Removed unused members:
  // - std::unique_ptr<RaiLoader> rai_loader_; (never initialized or used)
  // - int output_dim_; (unused, output_dim() returns hardcoded 3584)
};

// pybind11 module
PYBIND11_MODULE(_vision_flexmlrt_cpu, m) {
  m.doc() = "FlexMLRT vision model with CPU preprocessing support";

  py::class_<VisionFlexMLRTModel>(m, "VisionFlexMLRTModel")
      .def(py::init<std::string, std::string>(), py::arg("model_cache"),
           py::arg("device_name") = "stx",
           "Load FlexMLRT vision model\n\n"
           "Args:\n"
           "    model_cache: Path to VAIP model cache (vaiml_par_0 directory)\n"
           "    device_name: XRT device name (default: 'stx')")
      .def("forward", &VisionFlexMLRTModel::forward, py::arg("input"),
           py::arg("input_name") = "/blocks/Gather_output_0",
           py::arg("output_name") = "/merger/merger/mlp/mlp.2/Gemm_output_0",
           py::arg("output_shape") = std::vector<int64_t>{1073, 3584},
           "Run vision encoding on the NPU (generic; IO names/shape supplied by "
           "the caller from the cache's own spec).\n\n"
           "Args:\n"
           "    input: float32 array matching the cache's input tensor\n"
           "    input_name: NPU-partition input tensor name\n"
           "    output_name: NPU-partition output tensor name\n"
           "    output_shape: output tensor shape\n\n"
           "Returns:\n"
           "    float32 array of output_shape\n\n"
           "Defaults reproduce the Qwen2.5-VL contract for backward compat.");
}
