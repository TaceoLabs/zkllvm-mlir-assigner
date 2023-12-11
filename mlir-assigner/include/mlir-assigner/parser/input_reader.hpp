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
        template<typename BlueprintFieldType, typename var, typename Assignment>
        class InputReader {
        public:
            InputReader(stack_frame<var> &frame, Assignment &assignmnt) :
                frame(frame), assignmnt(assignmnt), public_input_idx(0), private_input_idx(0) {
            }

            template<typename InputType>
            var put_into_assignment(InputType &input, bool is_private) {
                if (is_private) {
                    assignmnt.private_storage(private_input_idx) = input;
                    return var(Assignment::PRIVATE_STORAGE_INDEX, private_input_idx++, false,
                               var::column_type::public_input);
                } else {
                    assignmnt.public_input(0, public_input_idx) = input;
                    return var(0, public_input_idx++, false, var::column_type::public_input);
                }
            }

            bool parse_fixedpoint(const boost::json::value &value, typename BlueprintFieldType::value_type &out) {
                // for now only double, but later we most likely will need strings as well
                // we hardcode the scale with 2^16 for now. Let's see later down the line
                double d;
                if (value.kind() == boost::json::kind::double_) {
                    d = value.as_double();
                } else {
                    UNREACHABLE("TODO add string support");
                }
                nil::blueprint::components::FixedPoint<BlueprintFieldType, 1, 1> fixed(d);
                out = fixed.get_value();
                return true;
            }

            bool parse_bool(const boost::json::value &value, typename BlueprintFieldType::value_type &out) {
                ASSERT(value.kind() == boost::json::kind::int64 && "bools must be 0 or 1");
                ASSERT((value.as_int64() >= 0 && value.as_int64() <= 1) && "bools must be 0 or 1");
                return parse_scalar(value, out);
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

            std::vector<var> put_field_into_assignmnt(std::vector<typename BlueprintFieldType::value_type> input,
                                                      bool is_private) {

                std::vector<var> res;

                for (std::size_t i = 0; i < input.size(); i++) {
                    res.push_back(put_into_assignment(input[i], is_private));
                }

                return res;
            }

            std::vector<var> process_int(const boost::json::object &object, std::size_t bitness, bool is_private) {
                ASSERT(object.size() == 1 && object.contains("int"));
                std::vector<var> res = std::vector<var>(1);

                typename BlueprintFieldType::value_type out;

                switch (object.at("int").kind()) {
                    case boost::json::kind::int64:
                        if (bitness < 64 && object.at("int").as_int64() >> bitness > 0) {
                            std::cerr << "value " << object.at("int").as_int64() << " does not fit into " << bitness
                                      << " bits\n";
                            UNREACHABLE("one of the input values is too large");
                        }
                        out = object.at("int").as_int64();
                        break;
                    case boost::json::kind::uint64:
                        if (bitness < 64 && object.at("int").as_uint64() >> bitness > 0) {
                            std::cerr << "value " << object.at("int").as_uint64() << " does not fit into " << bitness
                                      << " bits\n";
                            UNREACHABLE("one of the input values is too large");
                        }
                        out = object.at("int").as_uint64();
                        break;
                    case boost::json::kind::double_: {
                        std::cerr << "error in json value " << boost::json::serialize(object) << "\n";
                        error =
                            "got double value for int argument. Probably the value is too big to "
                            "be represented as "
                            "integer. You can put it in \"\" to avoid JSON parser restrictions.";
                        UNREACHABLE(error);
                    }
                    case boost::json::kind::string: {
                        const std::size_t buflen = 256;
                        char buf[buflen];

                        std::size_t numlen = object.at("int").as_string().size();

                        if (numlen > buflen - 1) {
                            std::cerr << "value " << object.at("int").as_string() << " exceeds buffer size ("
                                      << buflen - 1 << ")\n";
                            UNREACHABLE("value size exceeds buffer size");
                        }

                        object.at("int").as_string().copy(buf, numlen);
                        buf[numlen] = '\0';
                        typename BlueprintFieldType::extended_integral_type number =
                            typename BlueprintFieldType::extended_integral_type(buf);
                        typename BlueprintFieldType::extended_integral_type one = 1;
                        ASSERT_MSG(bitness <= 128,
                                   "integers larger than 128 bits are not "
                                   "supported, try to use field types");
                        typename BlueprintFieldType::extended_integral_type max_size = one << bitness;
                        if (number >= max_size) {
                            std::cout << "value " << buf << " does not fit into " << bitness
                                      << " bits, try to use other type\n";
                            UNREACHABLE("input value is too big");
                        }
                        out = number;
                        break;
                    }
                    default:
                        UNREACHABLE("process_int handles only ints");
                        break;
                }

                res[0] = put_into_assignment(out, is_private);
                return res;
            }

            bool take_memref(mlir::BlockArgument arg, mlir::MemRefType memref_type, const boost::json::object &value,
                             bool is_private) {
                if (value.size() != 1 || !value.contains("memref") || !value.at("memref").is_object()) {
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
                parse_memref_data(m, mo.at("data").as_array(), type);

                // TODO: process data in memref
                // auto values = process_fixedpoint(
                //     llvm::cast<llvm::ZkFixedPointType>(fixedpoint_type), value);
                // if (values.size() != 1)
                //   return false;
                // frame.scalars[fixedpoint_arg] = values[0];

                auto res = frame.memrefs.insert({mlir::hash_value(arg), m});
                ASSERT(res.second);    // we do not want to override stuff here
                return true;
            }

            bool parse_memref_data(memref<var> &data, const boost::json::array &tensor_arr, std::string &type) {

                if (type == "f32") {
                    for (size_t i = 0; i < tensor_arr.size(); ++i) {
                        if (!parse_fixedpoint(tensor_arr[i], assignmnt.public_input(0, public_input_idx))) {
                            llvm::errs() << "expect fixedpoints in tensor\n";
                            return false;
                        }
                        data.put_flat(i, var(0, public_input_idx++, false, var::column_type::public_input));
                    }
                } else if (type == "bool") {
                    for (size_t i = 0; i < tensor_arr.size(); ++i) {
                        if (!parse_bool(tensor_arr[i], assignmnt.public_input(0, public_input_idx))) {
                            llvm::errs() << "expect fixedpoints in tensor\n";
                            return false;
                        }
                        data.put_flat(i, var(0, public_input_idx++, false, var::column_type::public_input));
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

            bool fill_public_input(mlir::func::FuncOp &function, const boost::json::array &public_input) {
                size_t ret_gap = 0;
                mlir::FunctionType func_type = function.getFunctionType();
                mlir::Region &reg = function.getBody();
                auto args = reg.getArguments();
                for (size_t i = 0; i < func_type.getNumInputs(); ++i) {
                    if (public_input.size() <= i - ret_gap || !public_input[i - ret_gap].is_object()) {
                        error = "not enough values in the input file.";
                        return false;
                    }

                    auto arg = args[i];

                    mlir::Type arg_type = func_type.getInput(i);

                    const boost::json::object &current_value = public_input[i - ret_gap].as_object();

                    bool is_private = false;    // currently we don't have private input support in MLIR

                    if (mlir::MemRefType memref_type = llvm::dyn_cast<mlir::MemRefType>(arg_type)) {
                        if (!take_memref(arg, memref_type, current_value, is_private))
                            return false;
                    } else {
                        UNREACHABLE("only memref types are supported for now");
                    }
                }

                // Check if there are remaining elements of public input
                if (func_type.getNumInputs() - ret_gap != public_input.size()) {
                    error = "too many values in the input file";
                    return false;
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
            Assignment &assignmnt;
            size_t public_input_idx;
            size_t private_input_idx;
            std::string error;
        };
    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_PUBLIC_INPUT_HPP
