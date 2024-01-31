//---------------------------------------------------------------------------//
// Copyright (c) 2022 Mikhail Komarov <nemo@nil.foundation>
// Copyright (c) 2022 Nikita Kaskov <nbering@nil.foundation>
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//---------------------------------------------------------------------------//

#ifndef CRYPTO3_ASSIGNER_PUBLIC_INPUT_HPP
#define CRYPTO3_ASSIGNER_PUBLIC_INPUT_HPP

#include <boost/json/kind.hpp>
#include "mlir-assigner/helper/asserts.hpp"
#include "onnx/string_utils.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mlir-assigner/memory/stack_frame.hpp>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>
#include "mlir/Dialect/zkml/IR/ZkMlAttributes.h"

#include <nil/blueprint/components/algebra/fixedpoint/type.hpp>

#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <boost/json/src.hpp>

namespace nil {
    namespace blueprint {
        template<typename BlueprintFieldType, typename var, typename Assignment, std::uint8_t PreLimbs,
                 std::uint8_t PostLimbs>
        class InputReader {
        public:
            InputReader(stack_frame<var> &frame, Assignment &assignmnt, std::vector<memref<var>> &output_memrefs) :
                frame(frame), output_memrefs(output_memrefs), assignmnt(assignmnt), public_input_idx(0),
                private_input_idx(0) {
            }

            bool parse_fixedpoint(const boost::json::value &value, typename BlueprintFieldType::value_type &out) {
                double d;
                if (value.kind() == boost::json::kind::double_) {
                    d = value.as_double();
                } else if (value.kind() == boost::json::kind::int64) {
                    d = static_cast<double>(value.as_int64());
                } else {
                    UNREACHABLE("TODO add string support");
                }
                nil::blueprint::components::FixedPoint<BlueprintFieldType, PreLimbs, PostLimbs> fixed(d);
                out = fixed.get_value();
                return true;
            }

            bool parse_bool(const boost::json::value &value, typename BlueprintFieldType::value_type &out) {
                ASSERT(value.kind() == boost::json::kind::int64 && "bools must be 0 or 1");
                ASSERT((value.as_int64() >= 0 && value.as_int64() <= 1) && "bools must be 0 or 1");
                return parse_scalar(value, out);
            }

            bool parse_int(const boost::json::value &value, typename BlueprintFieldType::value_type &out) {
                switch (value.kind()) {
                    case boost::json::kind::int64:
                    case boost::json::kind::uint64:
                        return parse_scalar(value, out);
                    default:
                        std::cerr << "unsupported int type: " << value.as_string() << std::endl;
                        UNREACHABLE("int must be int64 or uint64");
                };
            }

            bool parse_scalar(const boost::json::value &value, typename BlueprintFieldType::value_type &out) {
                const std::size_t buflen = 256;
                char buf[buflen];
                std::size_t numlen = 0;
                switch (value.kind()) {
                    case boost::json::kind::int64:
                        out = value.as_int64();
                        return true;
                    case boost::json::kind::uint64:
                        out = value.as_uint64();
                        return true;
                    case boost::json::kind::string: {
                        numlen = value.as_string().size();
                        if (numlen > buflen - 1) {
                            std::cerr << "value " << value.as_string() << " exceeds buffer size (" << buflen - 1
                                      << ")\n";
                            UNREACHABLE("value size exceeds buffer size");
                        }
                        value.as_string().copy(buf, numlen);
                        buf[numlen] = '\0';
                        typename BlueprintFieldType::extended_integral_type number(buf);
                        if (number >= BlueprintFieldType::modulus) {
                            std::cerr << "Input does not fit into BlueprintFieldType" << std::endl;
                            return false;
                        }
                        out = number;
                        return true;
                    }
                    default:
                        return false;
                }
            }

            bool take_memref(mlir::BlockArgument arg, mlir::MemRefType memref_type, const boost::json::object &value,
                             bool is_private) {
                if (value.size() != 1 || !value.contains("memref") || !value.at("memref").is_object()) {
                    error = "invalid json object for input memref";
                    return false;
                }
                memref<var> m(memref_type.getShape(), memref_type.getElementType());

                const boost::json::object &mo = value.at("memref").as_object();
                if (!mo.contains("data") || !mo.at("data").is_array()) {
                    error = "memref does not contain data";
                    return false;
                }
                if (!mo.contains("dims") || !mo.at("dims").is_array()) {
                    error = "memref does not contain dims";
                    return false;
                }
                if (!mo.contains("type") || !mo.at("type").is_string()) {
                    error = "memref does not contain type";
                    return false;
                }
                auto dims = parse_dim_array(mo.at("dims").as_array());
                std::string type = mo.at("type").as_string().c_str();
                if (is_private) {
                    parse_memref_data_private(m, mo.at("data").as_array(), type);
                } else {
                    parse_memref_data_public(m, mo.at("data").as_array(), type);
                }

                auto res = frame.memrefs.insert({mlir::hash_value(arg), m});
                ASSERT(res.second);    // we do not want to override stuff here
                return true;
            }

            // reserves space for the output memref in the assignment table, directly after the other inputs
            // this will be filled later on with the actual values of the output
            bool reserve_output_memref(mlir::MemRefType memref_type, bool is_private) {
                memref<var> m(memref_type.getShape(), memref_type.getElementType());
                if (is_private) {
                    for (size_t i = 0; i < m.size(); ++i) {
                        assignmnt.private_storage(private_input_idx) = 0;
                        m.put_flat(i, var(Assignment::private_storage_index, private_input_idx++, false,
                                          var::column_type::public_input));
                    }
                } else {
                    for (size_t i = 0; i < m.size(); ++i) {
                        assignmnt.public_input(0, public_input_idx) = 0;
                        m.put_flat(i, var(0, public_input_idx++, false, var::column_type::public_input));
                    }
                }
                output_memrefs.push_back(std::move(m));
                return true;
            }
            // reserves space for the output memref in the assignment table, directly after the other inputs
            // fills the output memref with the values from the input file
            bool take_output_memref(mlir::MemRefType memref_type, const boost::json::object &value, bool is_private) {
                if (value.size() != 1 || !value.contains("memref") || !value.at("memref").is_object()) {
                    error = "invalid json object for output memref";
                    return false;
                }
                memref<var> m(memref_type.getShape(), memref_type.getElementType());

                const boost::json::object &mo = value.at("memref").as_object();
                if (!mo.contains("data") || !mo.at("data").is_array()) {
                    error = "output memref does not contain data";
                    return false;
                }
                if (!mo.contains("dims") || !mo.at("dims").is_array()) {
                    error = "output memref does not contain dims";
                    return false;
                }
                if (!mo.contains("type") || !mo.at("type").is_string()) {
                    error = "output memref does not contain type";
                    return false;
                }
                auto dims = parse_dim_array(mo.at("dims").as_array());
                std::string type = mo.at("type").as_string().c_str();
                if (is_private) {
                    parse_memref_data_private(m, mo.at("data").as_array(), type);
                } else {
                    parse_memref_data_public(m, mo.at("data").as_array(), type);
                }
                output_memrefs.push_back(std::move(m));

                return true;
            }

            /// parse a memref from the input file into the public input column
            bool parse_memref_data_public(memref<var> &data, const boost::json::array &tensor_arr, std::string &type) {
                if (type == "f32") {
                    for (size_t i = 0; i < tensor_arr.size(); ++i) {
                        if (!parse_fixedpoint(tensor_arr[i], assignmnt.public_input(0, public_input_idx))) {
                            llvm::errs() << "expect fixedpoints in tensor\n";
                            return false;
                        }
                        data.put_flat(i, var(0, public_input_idx++, false, var::column_type::public_input));
                    }
                } else if (type == "int") {
                    // TODO do we have to handle uint?
                    for (size_t i = 0; i < tensor_arr.size(); ++i) {
                        if (!parse_int(tensor_arr[i], assignmnt.public_input(0, public_input_idx))) {
                            llvm::errs() << "expect ints in tensor\n";
                            return false;
                        }
                        data.put_flat(i, var(0, public_input_idx++, false, var::column_type::public_input));
                    }
                } else if (type == "bool") {
                    for (size_t i = 0; i < tensor_arr.size(); ++i) {
                        if (!parse_bool(tensor_arr[i], assignmnt.public_input(0, public_input_idx))) {
                            llvm::errs() << "expect booleans in tensor\n";
                            return false;
                        }
                        data.put_flat(i, var(0, public_input_idx++, false, var::column_type::public_input));
                    }
                } else {
                    UNREACHABLE(std::string("unsupported memref type: ") + type);
                }
                return true;
            }

            /// parse a memref from the input file into the private input column
            bool parse_memref_data_private(memref<var> &data, const boost::json::array &tensor_arr, std::string &type) {
                if (type == "f32") {
                    for (size_t i = 0; i < tensor_arr.size(); ++i) {
                        if (!parse_fixedpoint(tensor_arr[i], assignmnt.private_storage(private_input_idx))) {
                            llvm::errs() << "expect fixedpoints in tensor\n";
                            return false;
                        }
                        data.put_flat(i, var(Assignment::private_storage_index, private_input_idx++, false,
                                             var::column_type::public_input));
                    }
                } else if (type == "int") {
                    // TODO do we have to handle uint?
                    for (size_t i = 0; i < tensor_arr.size(); ++i) {
                        if (!parse_int(tensor_arr[i], assignmnt.private_storage(private_input_idx))) {
                            llvm::errs() << "expect ints in tensor\n";
                            return false;
                        }
                        data.put_flat(i, var(Assignment::private_storage_index, private_input_idx++, false,
                                             var::column_type::public_input));
                    }
                } else if (type == "bool") {
                    for (size_t i = 0; i < tensor_arr.size(); ++i) {
                        if (!parse_bool(tensor_arr[i], assignmnt.private_storage(private_input_idx))) {
                            llvm::errs() << "expect booleans in tensor\n";
                            return false;
                        }
                        data.put_flat(i, var(Assignment::private_storage_index, private_input_idx++, false,
                                             var::column_type::public_input));
                    }
                } else {
                    UNREACHABLE(std::string("unsupported memref type: ") + type);
                }
                return true;
            }

            std::vector<int64_t> parse_dim_array(const boost::json::array &dim_arr) {
                std::vector<int64_t> res;
                res.reserve(dim_arr.size());
                for (size_t i = 0; i < dim_arr.size(); ++i) {
                    if (dim_arr[i].kind() != boost::json::kind::int64 || dim_arr[i].as_int64() <= 0) {
                        llvm::errs() << "expect unsigned ints for tensor dimensions >0\n";
                        return res;
                    }
                    // dimension
                    res.emplace_back(dim_arr[i].as_int64());
                }
                return res;
            }

            bool parse_indices(std::unordered_map<size_t, size_t> &map, const boost::json::array &json) {
                for (size_t i = 0; i < json.size(); ++i) {
                    const boost::json::object &current_value = json[i].as_object();
                    if (current_value.size() != 1 || !current_value.contains("memref") ||
                        !current_value.at("memref").is_object()) {
                        error = "invalid json object for input memref";
                        return false;
                    }

                    const boost::json::object &mo = current_value.at("memref").as_object();

                    if (!mo.contains("idx") || mo.at("idx").kind() != boost::json::kind::int64) {
                        error = "memref does not contain idx";
                        return false;
                    }
                    int64_t idx = mo.at("idx").as_int64();
                    if (idx < 0) {
                        error = "negative indices not supported";
                        return false;
                    }
                    auto res = map.insert(std::make_pair(i, static_cast<size_t>(idx)));
                    if (!res.second) {
                        error = "duplicate index in memrefs: " + std::to_string(idx);
                        return false;
                    }
                }
                return true;
            }

            bool map_input(size_t counter, mlir::BlockArgument arg, mlir::Type arg_type,
                           std::unordered_map<size_t, size_t> &idx_map, const boost::json::array &json,
                           bool is_private) {
                auto idx = idx_map.find(counter);
                if (idx == idx_map.end()) {
                    error = (is_private ? "No private input found with idx (" : "No public input found with idx (") +
                            std::to_string(counter) + ")";
                    return false;
                }
                if (json.size() <= idx->second) {
                    error = "idx (" + std::to_string(idx->second) + ") out-of-bounds in " +
                            (is_private ? "private input file" : "public input file");
                    return false;
                }
                const boost::json::object &current_value = json[idx->second].as_object();
                if (mlir::MemRefType memref_type = llvm::dyn_cast<mlir::MemRefType>(arg_type)) {
                    if (!take_memref(arg, memref_type, current_value, is_private))
                        return false;
                } else {
                    UNREACHABLE("only memref types are supported for now");
                    return false;
                }
                return true;
            }

            bool fill_input(mlir::func::FuncOp &function, const boost::json::array &public_input,
                            const boost::json::array &private_input) {

                std::unordered_map<size_t, size_t> public_indices;
                std::unordered_map<size_t, size_t> private_indices;
                if (!parse_indices(public_indices, public_input)) {
                    return false;
                }
                if (!parse_indices(private_indices, private_input)) {
                    return false;
                }

                mlir::FunctionType func_type = function.getFunctionType();
                mlir::Region &reg = function.getBody();
                auto args = reg.getArguments();
                size_t public_counter = 0;
                size_t private_counter = 0;
                for (size_t i = 0; i < func_type.getNumInputs(); ++i) {
                    auto arg = args[i];

                    mlir::Type arg_type = func_type.getInput(i);

                    bool is_private = false;
                    mlir::Attribute inputAttr = function.getArgAttr(i, "zkML.input");
                    if (inputAttr) {
                        assert(llvm::isa<mlir::zkml::ZkMlPrivateInputAttr>(inputAttr) &&
                               "Got unknown attribute for zkML.input on input");
                        is_private = true;
                    }
                    bool success =
                        is_private ? map_input(private_counter++, arg, arg_type, private_indices, private_input, true) :
                                     map_input(public_counter++, arg, arg_type, public_indices, public_input, false);
                    if (!success) {
                        return false;
                    }
                }

                // Check if there are remaining elements of public input
                if (func_type.getNumInputs() != public_input.size() + private_input.size()) {
                    std::stringstream ss;
                    ss << std::endl << "too many values in the input files" << std::endl;
                    ss << "Expected: " << func_type.getNumInputs() << " inputs, got ";
                    ss << public_input.size() + private_input.size() << std::endl;
                    error = ss.str();
                    return false;
                }
                return true;
            }

            bool reserve_outputs(mlir::func::FuncOp &function, boost::json::array &public_outputs,
                                 bool &output_is_already_present) {
                mlir::FunctionType func_type = function.getFunctionType();

                bool is_private = false;    // currently we don't have private output support in MLIR

                if (!public_outputs.empty()) {
                    // we have the outputs already
                    output_is_already_present = true;
                }
                auto results = func_type.getResults();
                for (unsigned i = 0; i < results.size(); ++i) {
                    mlir::Type return_type = results[i];
                    if (mlir::MemRefType memref_type = llvm::dyn_cast<mlir::MemRefType>(return_type)) {
                        if (!output_is_already_present) {
                            if (!reserve_output_memref(memref_type, is_private)) {
                                return false;
                            }
                        } else {
                            const boost::json::object &current_value = public_outputs[i].as_object();
                            if (!take_output_memref(memref_type, current_value, is_private))
                                return false;
                        }
                    } else {
                        UNREACHABLE("only memref types are supported for now");
                    }
                }
                return true;
            }
            size_t get_idx() const {
                return public_input_idx;
            }

            const std::string &get_error() const {
                return error;
            }

        private:
            stack_frame<var> &frame;
            std::vector<memref<var>> &output_memrefs;
            Assignment &assignmnt;
            size_t public_input_idx;
            size_t private_input_idx;
            std::string error;
        };
    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_PUBLIC_INPUT_HPP
