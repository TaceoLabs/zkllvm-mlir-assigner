#ifndef MLIR_ASSIGNER_MEMORY_MEMREF_HPP
#define MLIR_ASSIGNER_MEMORY_MEMREF_HPP

#include <cstdint>
#include <mlir-assigner/helper/asserts.hpp>
#include <nil/blueprint/blueprint/plonk/assignment.hpp>
#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/type.hpp>
#include <llvm/ADT/ArrayRef.h>

#include <vector>
#include <functional>

namespace nil {
    namespace blueprint {

        template<typename VarType>
        struct memref {
            memref() = default;

            memref(const memref &) = default;

            memref(memref &&) = default;

            memref &operator=(const memref &) = default;

            memref &operator=(memref &&) = default;

            memref(std::vector<int64_t> dims, mlir::Type type) : data(), dims(dims), strides(), type(type) {
                strides.resize(dims.size());
                for (int i = dims.size() - 1; i >= 0; i--) {
                    strides[i] = (i == dims.size() - 1) ? 1 : strides[i + 1] * dims[i + 1];
                    ASSERT(dims[i] > 0);
                }
                // this also handles the case when dims is empty, since we still allocate
                // 1 here
                data.resize(std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<uint32_t>()));
            }
            memref(llvm::ArrayRef<int64_t> dims, mlir::Type type) : data(), dims(dims), strides(), type(type) {
                strides.resize(dims.size());
                for (int i = dims.size() - 1; i >= 0; i--) {
                    strides[i] = (i == dims.size() - 1) ? 1 : strides[i + 1] * dims[i + 1];
                    ASSERT(dims[i] > 0);
                }
                // this also handles the case when dims is empty, since we still allocate 1
                // here
                data.resize(std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<uint32_t>()));
            }

            memref(std::vector<int64_t> dims, std::vector<VarType> data, mlir::Type type) :
                data(data), dims(dims), strides() {
                strides.resize(dims.size());
                for (int i = dims.size() - 1; i >= 0; i--) {
                    strides[i] = (i == dims.size() - 1) ? 1 : strides[i + 1] * dims[i + 1];
                }
                assert(data.size() ==
                       std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<uint32_t>()));
            }

            const VarType &get(const std::vector<int64_t> &indices) const {
                assert(indices.size() == dims.size());
                uint32_t offset = 0;
                for (int i = 0; i < indices.size(); i++) {
                    assert(indices[i] < dims[i]);
                    offset += indices[i] * strides[i];
                }
                return data[offset];
            }
            // VarType &get(const std::vector<uint32_t> &indices) const {
            //   assert(indices.size() == dims.size());
            //   uint32_t offset = 0;
            //   for (int i = 0; i < indices.size(); i++) {
            //     assert(indices[i] < dims[i]);
            //     offset += indices[i] * strides[i];
            //   }
            //   return data[offset];
            // }

            void put(const std::vector<int64_t> &indices, const VarType &value) {
                assert(indices.size() == dims.size());
                uint32_t offset = 0;
                for (int i = 0; i < indices.size(); i++) {
                    assert(indices[i] < dims[i]);
                    offset += indices[i] * strides[i];
                }
                data[offset] = value;
            }

            void put_flat(const int64_t idx, const VarType &value) {
                assert(idx >= 0 && idx < data.size());
                data[idx] = value;
            }

            mlir::Type getType() const {
                return type;
            }
            int64_t size() const {
                return data.size();
            }

            template<typename BlueprintFieldType, typename ArithmetizationParams>
            void print(
                std::ostream& os,
                const assignment<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                    &assignment) {
                os << "memref<";
                for (int i = 0; i < dims.size(); i++) {
                    os << dims[i];
                    os << "x";
                }
                std::string type_str;
                llvm::raw_string_ostream ss(type_str);
                ss << type << ">[";
                os << type_str;
                if (type.isa<mlir::IntegerType>()) {
                  if (type.getIntOrFloatBitWidth() == 1) {
                    //bool
                    for (int i = 0; i < data.size(); i++) {
                        os << var_value(assignment, data[i]).data;
                        if (i != data.size() - 1)
                            os << ",";
                    }
                  } else {
                    //int
                  }
                } else if (type.isa<mlir::FloatType>()) {
                    for (int i = 0; i < data.size(); i++) {
                        auto value = var_value(assignment, data[i]).data;
                        components::FixedPoint<BlueprintFieldType, 1, 1> out(value, 16);
                        os << out.to_double();
                        if (i != data.size() - 1)
                            os << ",";
                    }
                }
                os << "]\n";
            }

            nil::blueprint::memref<VarType> reinterpret_as(llvm::ArrayRef<int64_t> new_dims, mlir::Type new_type,
                                                           nil::blueprint::logger &logger) {
                logger << new_dims;
                logger << new_type;
                // build a new memref
                nil::blueprint::memref<VarType> new_memref(new_dims, new_type);
                ASSERT(new_memref.size() == this->size());
                ASSERT(new_memref.getType() == this->type);
                // just copy over data
                new_memref.data = this->data;
                return new_memref;
            }

            std::vector<VarType> &getData() {
                ASSERT(dims.size() == 1 && "When obtaining data, must be one-dim");
                return data;
            }

            std::vector<int64_t> &getDims() {
                return dims;
            }

        private:
            std::vector<VarType> data;
            std::vector<int64_t> dims;
            std::vector<int64_t> strides;
            mlir::Type type;
        };

    }    // namespace blueprint
}    // namespace nil
#endif    // MLIR_ASSIGNER_MEMORY_MEMREF_HPP
