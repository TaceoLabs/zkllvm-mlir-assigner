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

#ifndef CRYPTO3_ASSIGNER_OUTPUT_WRITER_HPP
#define CRYPTO3_ASSIGNER_OUTPUT_WRITER_HPP

#include <boost/json/kind.hpp>
#include "mlir-assigner/helper/asserts.hpp"
#include "onnx/string_utils.h"
#include <cstdint>
#include <mlir-assigner/memory/stack_frame.hpp>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>

#include <nil/blueprint/components/algebra/fixedpoint/type.hpp>

#include <iostream>
#include <boost/json/src.hpp>

namespace nil {
    namespace blueprint {
        template<typename BlueprintFieldType,
                 typename var,
                 typename Assignment,
                 std::uint8_t PreLimbs,
                 std::uint8_t PostLimbs>
        class OutputWriter {
            using FixedPoint = nil::blueprint::components::FixedPoint<BlueprintFieldType, PreLimbs, PostLimbs>;

        public:
            OutputWriter(Assignment &assignmnt, std::vector<memref<var>> &output_memrefs) :
                output_memrefs(output_memrefs), assignmnt(assignmnt) {
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
                FixedPoint fixed(d);
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

            bool make_outputs_to_json(boost::json::array &public_output) {
                ASSERT(public_output.empty());
                for (unsigned i = 0; i < output_memrefs.size(); ++i) {
                    auto &memref = output_memrefs[i];
                    boost::json::object mo;
                    std::string type;
                    if (memref.getType().isF16()) {
                        type = "f16";
                    } else if (memref.getType().isF32()) {
                        type = "f32";
                    } else if (memref.getType().isF64()) {
                        type = "f64";
                    } else if (memref.getType().isInteger(1)) {
                        type = "bool";
                    } else if (memref.getType().isIntOrIndex()) {
                        type = "int";
                    } else {
                        UNREACHABLE(std::string("unsupported memref type: ") + type);
                    }
                    mo.emplace("type", type);
                    boost::json::array dims;
                    for (auto dim : memref.getDims()) {
                        dims.emplace_back(dim);
                    }
                    mo.emplace("dims", dims);
                    boost::json::array data;

                    if (type == "f16" || type == "f32" || type == "f64") {
                        for (int64_t j = 0; j < memref.size(); ++j) {
                            auto val = var_value(assignmnt, memref.get_flat(j));
                            FixedPoint fixed(val, FixedPoint::SCALE);
                            data.emplace_back(fixed.to_double());
                        }
                    } else if (type == "int") {
                        for (int64_t j = 0; j < memref.size(); ++j) {
                            int64_t val = resolve_number(memref.get_flat(j));
                            data.emplace_back(val);
                        }
                    } else if (type == "bool") {
                        for (int64_t j = 0; j < memref.size(); ++j) {
                            int64_t val = resolve_number(memref.get_flat(j));
                            ASSERT(val == 0 || val == 1);
                            data.emplace_back(val);
                        }
                    } else {
                        UNREACHABLE(std::string("unsupported memref type: ") + type);
                    }
                    mo.emplace("data", data);
                    mo.emplace("idx", i);

                    boost::json::object o;
                    o.emplace("memref", mo);
                    public_output.emplace_back(o);
                }
                return true;
            }

            const std::string &get_error() const {
                return error;
            }

        private:
            int64_t resolve_number(var scalar) {
                auto scalar_value = var_value(assignmnt, scalar);
                static constexpr auto limit_value_max =
                    typename BlueprintFieldType::integral_type(std::numeric_limits<int64_t>::max());
                static constexpr auto limit_value_min =
                    BlueprintFieldType::modulus - limit_value_max - typename BlueprintFieldType::integral_type(1);
                static constexpr typename BlueprintFieldType::integral_type half_p =
                    (BlueprintFieldType::modulus - typename BlueprintFieldType::integral_type(1)) /
                    typename BlueprintFieldType::integral_type(2);
                auto integral_value = static_cast<typename BlueprintFieldType::integral_type>(scalar_value.data);
                ASSERT_MSG(integral_value <= limit_value_max || integral_value >= limit_value_min,
                           "cannot fit into requested number");
                // check if negative
                if (integral_value > half_p) {
                    integral_value = BlueprintFieldType::modulus - integral_value;
                    return -static_cast<int64_t>(integral_value);
                } else {
                    return static_cast<int64_t>(integral_value);
                }
            }

        private:
            std::vector<memref<var>> &output_memrefs;
            Assignment &assignmnt;
            std::string error;
        };
    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_OUTPUT_WRITER_HPP
