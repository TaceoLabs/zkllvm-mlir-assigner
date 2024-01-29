#ifndef NIL_BLUEPRINT_STACK_FRAME_HPP
#define NIL_BLUEPRINT_STACK_FRAME_HPP

#include "mlir-assigner/helper/asserts.hpp"
#include <cstddef>
#include <cstdint>
#include <mlir-assigner/memory/memref.hpp>
#include <llvm/ADT/Hashing.h>
#include <vector>
#include <unordered_map>

namespace nil {
    namespace blueprint {

        template<typename VarType>
        struct stack_frame {
            std::unordered_map<size_t, int64_t> constant_values;
            std::unordered_map<size_t, memref<VarType>> memrefs;
            std::unordered_map<size_t, VarType> locals;

            stack_frame() = default;
            stack_frame(stack_frame &&s) = default;
            stack_frame &operator=(stack_frame &&s) = default;
            stack_frame(const stack_frame &s) = delete;
            stack_frame &operator=(const stack_frame &s) = delete;
        };

        template<typename VarType>
        class stack {
        private:
            std::vector<stack_frame<VarType>> frames;

        public:
            stack() = default;
            stack(const stack &s) = delete;
            stack(stack &&s) = delete;
            stack &operator=(stack &&s) = delete;
            stack &operator=(const stack &s) = delete;

            void push_frame() {
                frames.push_back(stack_frame<VarType>());
            }

            void pop_frame() {
                frames.pop_back();
            }

            stack_frame<VarType> &get_last_frame() {
                assert(frames.size() && "stack is empty, cannot get last element?");
                return frames.back();
            }

            template<typename MlirType>
            void push_constant(MlirType identifier, int64_t constant, bool allow_overwrite = true) {
                push_constant(mlir::hash_value(identifier), constant, allow_overwrite);
            }

            void push_constant(llvm::hash_code hash_code, int64_t constant, bool allow_overwrite = true) {
                assert(frames.size() && "stack is empty but we push?");
                if (allow_overwrite) {
                    frames.back().constant_values[size_t(hash_code)] = constant;
                } else {
                    auto [ele, inserted] = frames.back().constant_values.try_emplace(size_t(hash_code), constant);
                    assert(inserted && "Overwrite not supported for this constant");
                }
            }

            template<typename MlirType>
            void erase_constant(MlirType identifier) {
                erase_constant(mlir::hash_value(identifier));
            }

            void erase_constant(llvm::hash_code hash_code) {
                assert(frames.size() && "stack is empty but we push?");
                frames.back().constant_values.erase(size_t(hash_code));
            }

            template<typename MlirType>
            int64_t get_constant(MlirType identifier) {
                return get_constant(mlir::hash_value(identifier));
            }

            int64_t get_constant(llvm::hash_code hash_code) {
                // go in reverse order trough the stack
                for (auto iter = frames.rbegin(); iter != frames.rend(); ++iter) {
                    if (iter->constant_values.find(size_t(hash_code)) != iter->constant_values.end()) {
                        // yay we found it
                        return iter->constant_values[hash_code];
                    }
                }
                UNREACHABLE("empty");
            }

            template<typename MlirType>
            bool peek_constant(MlirType identifier) {
                return peek_constant(mlir::hash_value(identifier));
            }

            bool peek_constant(llvm::hash_code hash_code) {
                size_t hash = size_t(hash_code);
                for (auto iter = frames.rbegin(); iter != frames.rend(); ++iter) {
                    if (iter->constant_values.find(hash) != iter->constant_values.end()) {
                        // yay we found it
                        return true;
                    }
                }
                return false;
            }

            template<typename MlirType>
            void push_local(MlirType identifier, VarType &local, bool allow_overwrite = true) {
                push_local(mlir::hash_value(identifier), local, allow_overwrite);
            }

            void push_local(llvm::hash_code hash_code, VarType &local, bool allow_overwrite = true) {
                assert(frames.size() && "stack is empty but we push?");
                if (allow_overwrite) {
                    frames.back().locals[size_t(hash_code)] = local;
                } else {
                    auto [ele, inserted] = frames.back().locals.try_emplace(size_t(hash_code), local);
                    assert(inserted && "Overwrite not supported for this constant");
                }
            }

            template<typename MlirType>
            void erase_local(MlirType identifier) {
                erase_local(mlir::hash_value(identifier));
            }

            void erase_local(llvm::hash_code hash_code) {
                assert(frames.size() && "stack is empty but we push?");
                frames.back().locals.erase(size_t(hash_code));
            }

            template<typename MlirType>
            VarType &get_local(MlirType identifier) {
                return get_local(mlir::hash_value(identifier));
            }

            VarType &get_local(llvm::hash_code hash_code) {
                // go in reverse order trough the stack
                for (auto iter = frames.rbegin(); iter != frames.rend(); ++iter) {
                    if (iter->locals.find(size_t(hash_code)) != iter->locals.end()) {
                        // yay we found it
                        return iter->locals[hash_code];
                    }
                }
                UNREACHABLE("empty");
            }

            template<typename MlirType>
            void push_memref(MlirType identifier, memref<VarType> &memref, bool allow_overwrite = true) {
                push_memref(mlir::hash_value(identifier), memref, allow_overwrite);
            }

            void push_memref(llvm::hash_code hash_code, memref<VarType> &memref, bool allow_overwrite = true) {
                assert(frames.size() && "stack is empty but we push?");
                if (allow_overwrite) {
                    frames.back().memrefs[size_t(hash_code)] = memref;
                } else {
                    auto [ele, inserted] = frames.back().memrefs.try_emplace(size_t(hash_code), memref);
                    assert(inserted && "Overwrite not supported for this constant");
                }
            }

            template<typename MlirType>
            void erase_memref(MlirType identifier) {
                erase_memref(mlir::hash_value(identifier));
            }

            void erase_memref(llvm::hash_code hash_code) {
                assert(frames.size() && "stack is empty but we erase?");
                frames.back().memrefs.erase(size_t(hash_code));
            }

            template<typename MlirType>
            memref<VarType> &get_memref(MlirType identifier) {
                return get_memref(mlir::hash_value(identifier));
            }

            memref<VarType> &get_memref(llvm::hash_code hash_code) {
                // go in reverse order trough the stack
                for (auto iter = frames.rbegin(); iter != frames.rend(); ++iter) {
                    if (iter->memrefs.find(size_t(hash_code)) != iter->memrefs.end()) {
                        // yay we found it
                        return iter->memrefs[hash_code];
                    }
                }
                UNREACHABLE("did not find wanted memref in our stack");
            }

            void print(std::ostream &os) {
                // collect all values
                uint64_t amount_constant = 0;
                uint64_t amount_locals = 0;
                uint64_t amount_memrefs = 0;
                for (auto &frame : frames) {
                    amount_constant += frame.constant_values.size();
                    amount_locals += frame.locals.size();
                    amount_memrefs += frame.memrefs.size();
                }
                os << "=============\n";
                os << "#frames:" << frames.size() << "\n";
                os << "#constants: " << amount_constant << "\n";
                os << "#locals: " << amount_locals << "\n";
                os << "#memrefs: " << amount_memrefs << "\n";
                os << "=============\n";
            }
        };

    }    // namespace blueprint
}    // namespace nil

#endif    // NIL_BLUEPRINT_STACK_FRAME_HPP
