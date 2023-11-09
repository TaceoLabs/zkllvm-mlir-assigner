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

#include <mlir-assigner/memory/stack_frame.hpp>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>

#include <iostream>
#include <boost/json/src.hpp>

#define DELTA_FIX_1616 (1ULL << 16)

namespace nil {
namespace blueprint {
template <typename BlueprintFieldType, typename var, typename Assignment>
class InputReader {
public:
  InputReader(stack_frame<var> &frame, Assignment &assignmnt)
      : frame(frame), assignmnt(assignmnt), public_input_idx(0),
        private_input_idx(0) {}

  template <typename InputType>
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

  bool parse_fixedpoint(const boost::json::value &value,
                        typename BlueprintFieldType::value_type &out) {
    // for now only double, but later we most likely will need strings as well
    // we hardcode the scale with 2^16 for now. Let's see later down the line
    double d;
    if (value.kind() == boost::json::kind::double_) {
      d = value.as_double();
    } else {
      UNREACHABLE("TODO add string support");
    }
    if (d < 0) {
      out = static_cast<int64_t>(-d * DELTA_FIX_1616);
      out = -out;
    } else {
      out = static_cast<int64_t>(d * DELTA_FIX_1616);
    }
    return true;
  }

  bool parse_scalar(const boost::json::value &value,
                    typename BlueprintFieldType::value_type &out) {
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
        std::cerr << "value " << value.as_string() << " exceeds buffer size ("
                  << buflen - 1 << ")\n";
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

  std::vector<var> put_field_into_assignmnt(
      std::vector<typename BlueprintFieldType::value_type> input,
      bool is_private) {

    std::vector<var> res;

    for (std::size_t i = 0; i < input.size(); i++) {
      res.push_back(put_into_assignment(input[i], is_private));
    }

    return res;
  }

  std::vector<var> process_int(const boost::json::object &object,
                               std::size_t bitness, bool is_private) {
    ASSERT(object.size() == 1 && object.contains("int"));
    std::vector<var> res = std::vector<var>(1);

    typename BlueprintFieldType::value_type out;

    switch (object.at("int").kind()) {
    case boost::json::kind::int64:
      if (bitness < 64 && object.at("int").as_int64() >> bitness > 0) {
        std::cerr << "value " << object.at("int").as_int64()
                  << " does not fit into " << bitness << " bits\n";
        UNREACHABLE("one of the input values is too large");
      }
      out = object.at("int").as_int64();
      break;
    case boost::json::kind::uint64:
      if (bitness < 64 && object.at("int").as_uint64() >> bitness > 0) {
        std::cerr << "value " << object.at("int").as_uint64()
                  << " does not fit into " << bitness << " bits\n";
        UNREACHABLE("one of the input values is too large");
      }
      out = object.at("int").as_uint64();
      break;
    case boost::json::kind::double_: {
      std::cerr << "error in json value " << boost::json::serialize(object)
                << "\n";
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
        std::cerr << "value " << object.at("int").as_string()
                  << " exceeds buffer size (" << buflen - 1 << ")\n";
        UNREACHABLE("value size exceeds buffer size");
      }

      object.at("int").as_string().copy(buf, numlen);
      buf[numlen] = '\0';
      typename BlueprintFieldType::extended_integral_type number =
          typename BlueprintFieldType::extended_integral_type(buf);
      typename BlueprintFieldType::extended_integral_type one = 1;
      ASSERT_MSG(bitness <= 128, "integers larger than 128 bits are not "
                                 "supported, try to use field types");
      typename BlueprintFieldType::extended_integral_type max_size = one
                                                                     << bitness;
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

  bool take_memref(mlir::BlockArgument arg, mlir::MemRefType memref_type,
                   const boost::json::object &value, bool is_private) {
    if (value.size() != 1 || !value.contains("memref") ||
        !value.at("memref").is_object()) {
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

    parse_memref_data(m, mo.at("data").as_array());

    // TODO: process data in memref
    // auto values = process_fixedpoint(
    //     llvm::cast<llvm::ZkFixedPointType>(fixedpoint_type), value);
    // if (values.size() != 1)
    //   return false;
    // frame.scalars[fixedpoint_arg] = values[0];
    llvm::outs() << "parsed input: " << arg << "\n";
    m.print(llvm::outs(), assignmnt);

    auto res = frame.memrefs.insert({mlir::hash_value(arg), m});
    ASSERT(res.second); // we do not want to override stuff here
    return true;
  }

  bool parse_memref_data(memref<var> &data,
                         const boost::json::array &tensor_arr) {

    for (size_t i = 0; i < tensor_arr.size(); ++i) {
      if (!parse_fixedpoint(tensor_arr[i],
                            assignmnt.public_input(0, public_input_idx))) {
        llvm::errs() << "expect fixedpoints in tensor\n";
        return false;
      }
      data.put_flat(
          i, var(0, public_input_idx++, false, var::column_type::public_input));
    }
    return true;
  }

  std::vector<int64_t> parse_dim_array(const boost::json::array &dim_arr) {
    std::vector<int64_t> res;
    res.reserve(dim_arr.size());
    for (size_t i = 0; i < dim_arr.size(); ++i) {
      if (dim_arr[i].kind() != boost::json::kind::int64 ||
          dim_arr[i].as_int64() <= 0) {
        llvm::errs() << "expect unsigned ints for tensor dimensions >0\n";
        return res;
      }
      // dimension
      res.emplace_back(dim_arr[i].as_int64());
    }
    return res;
  }

  //   bool try_om_tensor(var &om_tensor_ptr, const boost::json::object &value,
  //                      size_t element_offset) {
  //     if (value.size() != 2 || !value.contains("data") ||
  //         !value.contains("dim")) {
  //       return false;
  //     }
  //     if (!value.at("data").is_array() || !value.at("dim").is_array()) {
  //       return false;
  //     }

  //     var data_ptr;
  //     if (!parse_tensor_data(data_ptr, value.at("data").as_array(),
  //                            element_offset)) {
  //       return false;
  //     }
  //     var dim_ptr;
  //     var strides_ptr;
  //     if (!parse_dim_array(dim_ptr, strides_ptr, value.at("dim").as_array(),
  //                          element_offset)) {
  //       return false;
  //     }

  //     assignmnt.public_input(0, public_input_idx) =
  //         value.at("dim").as_array().size();
  //     var tensor_rank =
  //         var(0, public_input_idx++, false, var::column_type::public_input);

  //     // hardcoded to one for the moment (float)
  //     assignmnt.public_input(0, public_input_idx) = 1;
  //     var data_type =
  //         var(0, public_input_idx++, false, var::column_type::public_input);
  //     // build the struct:
  //     //    void *_allocatedPtr;    -> data
  //     //    void *_alignedPtr;      -> TACEO_TODO do we need two pointers?
  //     //    int64_t _offset;        -> never used
  //     //    int64_t *_shape;        -> shape array
  //     //    int64_t *_strides;      -> strides array
  //     //    int64_t _rank;          -> rank
  //     //    OM_DATA_TYPE _dataType; -> ONNX data type
  //     //    int64_t _owning;        -> not used by us
  //     ptr_type ptr = memory.add_cells(
  //         std::vector<unsigned>(onnx::om_tensor_size, element_offset));
  //     assignmnt.public_input(0, public_input_idx) = ptr;
  //     om_tensor_ptr =
  //         var(0, public_input_idx++, false, var::column_type::public_input);

  //     // TACEO_TODO Lets check if we need to store something at the empty
  //     places memory.store(ptr++, data_ptr);    // _allocatedPtr;
  //     memory.store(ptr++, data_ptr);    // _alignedPtr;
  //     ptr++;                            // _offset not used so leave it be;
  //     memory.store(ptr++, dim_ptr);     // _shape
  //     memory.store(ptr++, strides_ptr); // _strides
  //     memory.store(ptr++, tensor_rank); // _rank
  //     memory.store(ptr++, data_type);   // _dataType
  //     ptr++;                            // _owning

  //     return true;
  //   }

  //   bool try_om_tensor_list(llvm::Value *arg, llvm::Type *arg_type,
  //                           const boost::json::object &value) {
  //     if (!arg_type->isPointerTy()) {
  //       return false;
  //     }
  //     if (!value.contains("tensor_list") ||
  //     !value.at("tensor_list").is_array()) {
  //       return false;
  //     }
  //     // TACEO_TODO this is a little bit hacky as we abuse the fact that ptr
  //     type
  //     // is same size as fixed point. Maybe think of something better
  //     size_t fp_size = layout_resolver.get_type_size(arg_type);
  //     // build the struct:
  //     //   OMTensor **_omts; // OMTensor array
  //     //   int64_t _size;    // Number of elements in _omts.
  //     //   int64_t _owning;  // not used by us
  //     ptr_type om_tensor_list_ptr = memory.add_cells(
  //         std::vector<unsigned>(onnx::om_tensor_list_size, fp_size));
  //     assignmnt.public_input(0, public_input_idx) = om_tensor_list_ptr;
  //     frame.scalars[arg] =
  //         var(0, public_input_idx++, false, var::column_type::public_input);

  //     auto json_arr = value.at("tensor_list").as_array();
  //     // store pointer to tensor list (_omts)
  //     ptr_type _omts_ptr =
  //         memory.add_cells(std::vector<unsigned>(json_arr.size(), fp_size));
  //     assignmnt.public_input(0, public_input_idx) = _omts_ptr;
  //     memory.store(om_tensor_list_ptr++, var(0, public_input_idx++, false,
  //                                            var::column_type::public_input));
  //     // store _size
  //     assignmnt.public_input(0, public_input_idx) = json_arr.size();
  //     memory.store(om_tensor_list_ptr++, var(0, public_input_idx++, false,
  //                                            var::column_type::public_input));

  //     // parse the tensors
  //     for (auto t : json_arr) {
  //       if (t.kind() != boost::json::kind::object) {
  //         return false;
  //       }
  //       var current_tensor;
  //       if (!try_om_tensor(current_tensor, t.as_object(), fp_size)) {
  //         return false;
  //       }
  //       memory.store(_omts_ptr++, current_tensor);
  //     }
  //     // owning nothing to do for use
  //     om_tensor_list_ptr++;
  //     return true;
  //   }

  bool fill_public_input(mlir::func::FuncOp &function,
                         const boost::json::array &public_input) {
    size_t ret_gap = 0;
    mlir::FunctionType func_type = function.getFunctionType();
    mlir::Region &reg = function.getBody();
    auto args = reg.getArguments();
    for (size_t i = 0; i < func_type.getNumInputs(); ++i) {
      if (public_input.size() <= i - ret_gap ||
          !public_input[i - ret_gap].is_object()) {
        error = "not enough values in the input file.";
        return false;
      }

      auto arg = args[i];
      llvm::outs() << hash_value(arg) << "\n";

      mlir::Type arg_type = func_type.getInput(i);
      llvm::outs() << arg_type << "\n";

      const boost::json::object &current_value =
          public_input[i - ret_gap].as_object();

      bool is_private =
          false; // currently we don't have private input support in MLIR

      if (mlir::MemRefType memref_type =
              llvm::dyn_cast<mlir::MemRefType>(arg_type)) {
        if (!take_memref(arg, memref_type, current_value, is_private))
          return false;
      }

      //   if (llvm::isa<llvm::PointerType>(arg_type)) {
      //     if (current_arg->hasStructRetAttr()) {
      //       auto pointee =
      //       current_arg->getAttribute(llvm::Attribute::StructRet)
      //                          .getValueAsType();
      //       ptr_type ptr = memory.add_cells(
      //           layout_resolver.get_type_layout<BlueprintFieldType>(pointee));
      //       frame.scalars[current_arg] = put_into_assignment(ptr,
      //       is_private); ret_gap += 1; continue;
      //     }
      //     if (current_arg->hasAttribute(llvm::Attribute::ByVal)) {
      //       auto pointee = current_arg->getAttribute(llvm::Attribute::ByVal)
      //                          .getValueAsType();
      //       if (pointee->isStructTy()) {
      //         if (try_struct(current_arg,
      //         llvm::cast<llvm::StructType>(pointee),
      //                        current_value, is_private))
      //           continue;
      //       } else if (pointee->isArrayTy()) {
      //         if (try_array(current_arg,
      //         llvm::cast<llvm::ArrayType>(pointee),
      //                       current_value, is_private))
      //           continue;
      //       } else {
      //         UNREACHABLE("unsupported pointer type");
      //       }
      //     }
      //     if (!try_string(current_arg, arg_type, current_value, is_private)
      //     &&
      //         !try_om_tensor_list(current_arg, arg_type, current_value)) {
      //       std::cerr << "Unhandled pointer argument" << std::endl;
      //       return false;
      //     }
      //   } else if (llvm::isa<llvm::FixedVectorType>(arg_type)) {
      //     if (!take_vector(current_arg, arg_type, current_value, is_private))
      //       return false;
      //   } else if (llvm::isa<llvm::EllipticCurveType>(arg_type)) {
      //     if (!take_curve(current_arg, arg_type, current_value, is_private))
      //       return false;
      //   } else if (llvm::isa<llvm::GaloisFieldType>(arg_type)) {
      //     if (!take_field(current_arg, arg_type, current_value, is_private))
      //       return false;
      //   } else if (llvm::isa<llvm::IntegerType>(arg_type)) {
      //     if (!take_int(current_arg, current_value, is_private))
      //       return false;
      //   } else if (llvm::isa<llvm::ZkFixedPointType>(arg_type)) {
      //     if (!take_fixedpoint(current_arg, arg_type, current_value))
      //       return false;
      //   } else {
      //     UNREACHABLE("unsupported input type");
      //   }
    }

    // Check if there are remaining elements of public input
    if (func_type.getNumInputs() - ret_gap != public_input.size()) {
      error = "too many values in the input file";
      return false;
    }
    return true;
  }
  size_t get_idx() const { return public_input_idx; }

  const std::string &get_error() const { return error; }

private:
  stack_frame<var> &frame;
  Assignment &assignmnt;
  size_t public_input_idx;
  size_t private_input_idx;
  std::string error;
};
} // namespace blueprint
} // namespace nil

#endif // CRYPTO3_ASSIGNER_PUBLIC_INPUT_HPP
