#ifndef MLIR_ASSIGNER_MEMORY_MEMREF_HPP
#define MLIR_ASSIGNER_MEMORY_MEMREF_HPP

#include <mlir-assigner/helper/asserts.hpp>

#include <vector>
#include <functional>

namespace nil {
namespace blueprint {

template <typename VarType> struct memref {
  memref() = default;

  memref(const memref &) = default;

  memref(memref &&) = default;

  memref &operator=(const memref &) = default;

  memref &operator=(memref &&) = default;

  memref(std::vector<uint32_t> dims) : data(), dims(dims), strides() {
    strides.resize(dims.size());
    for (int i = dims.size() - 1; i >= 0; i--) {
      strides[i] = (i == dims.size() - 1) ? 1 : strides[i + 1] * dims[i + 1];
    }
    data.resize(std::accumulate(std::begin(dims), std::end(dims), 1,
                                std::multiplies<uint32_t>()));
  }

  memref(std::vector<uint32_t> dims, std::vector<VarType> data)
      : data(data), dims(dims), strides() {
    strides.resize(dims.size());
    for (int i = dims.size() - 1; i >= 0; i--) {
      strides[i] = (i == dims.size() - 1) ? 1 : strides[i + 1] * dims[i + 1];
    }
    assert(data.size() == std::accumulate(std::begin(dims), std::end(dims), 1,
                                          std::multiplies<uint32_t>()));
  }

  const VarType &get(const std::vector<uint32_t> &indices) const {
    assert(indices.size() == dims.size());
    uint32_t offset = 0;
    for (int i = 0; i < indices.size(); i++) {
      assert(indices[i] < dims[i]);
      offset += indices[i] * strides[i];
    }
    return data[offset];
  }
  VarType &get(const std::vector<uint32_t> &indices) const {
    assert(indices.size() == dims.size());
    uint32_t offset = 0;
    for (int i = 0; i < indices.size(); i++) {
      assert(indices[i] < dims[i]);
      offset += indices[i] * strides[i];
    }
    return data[offset];
  }

  void put(const std::vector<uint32_t> &indices, const VarType &value) {
    assert(indices.size() == dims.size());
    uint32_t offset = 0;
    for (int i = 0; i < indices.size(); i++) {
      assert(indices[i] < dims[i]);
      offset += indices[i] * strides[i];
    }
    data[offset] = value;
  }

private:
  std::vector<VarType> data;
  std::vector<uint32_t> dims;
  std::vector<uint32_t> strides;
  mlir::Type type;
};
} // namespace blueprint
} // namespace nil
#endif // MLIR_ASSIGNER_MEMORY_MEMREF_HPP
