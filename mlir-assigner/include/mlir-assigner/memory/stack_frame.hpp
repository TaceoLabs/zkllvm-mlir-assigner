#ifndef NIL_BLUEPRINT_STACK_FRAME_HPP
#define NIL_BLUEPRINT_STACK_FRAME_HPP

#include <mlir-assigner/memory/memref.hpp>
#include <llvm/ADT/Hashing.h>
#include <vector>
#include <map>

namespace nil {
namespace blueprint {

template <typename VarType> struct stack_frame {
  std::map<llvm::hash_code, int64_t> constant_values;
  std::map<llvm::hash_code, nil::blueprint::memref<VarType>> memrefs;
  std::map<llvm::hash_code, VarType> locals;
};
} // namespace blueprint
} // namespace nil

#endif // NIL_BLUEPRINT_STACK_FRAME_HPP
