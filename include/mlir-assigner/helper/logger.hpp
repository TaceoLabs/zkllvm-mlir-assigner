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

#ifndef CRYPTO3_ASSIGNER_NIL_BLUEPRINT_LOGGER_HPP
#define CRYPTO3_ASSIGNER_NIL_BLUEPRINT_LOGGER_HPP

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include <string_view>
#include <llvm/IR/Instructions.h>

#include <spdlog/spdlog.h>

namespace nil {
namespace blueprint {
class logger {
public:

  enum class level {
    INFO,
    DEBUG,
    ERROR,
  };

  logger() {
    // spdlog::set_pattern("%L %v");

    spdlog::set_pattern("[%H:%M:%S %z] [%^-%L-%$] %v");
    spdlog::set_level(spdlog::level::info);
  }

  void set_level(level lvl) {
    this->lvl = lvl;
    switch (lvl) {
    case level::DEBUG:
      spdlog::set_level(spdlog::level::debug);
      break;
    case level::INFO:
      spdlog::set_level(spdlog::level::info);
      break;
    case level::ERROR:
      spdlog::set_level(spdlog::level::err);
      break;
    }
  }

  template <typename... Args>
  void debug(std::string_view fmt, const Args &...args) {
    spdlog::debug(fmt, args...);
  }

  template <typename... Args>
  void error(std::string_view fmt, const Args &...args) {
    spdlog::error(fmt, args...);
  }

  void log_value(const mlir::Value &Value) {
    if (lvl < level::DEBUG) {
      return;
    }
    std::string str;
    llvm::raw_string_ostream ss(str);
    ss << Value;
    spdlog::debug(str);
  }

  void log_affine_map(const mlir::AffineMap &AffineMap) {
    if (lvl < level::DEBUG) {
      return;
    }
    std::string str;
    llvm::raw_string_ostream ss(str);
    ss << AffineMap;
    spdlog::debug(str);
  }

  void log_attribute(const mlir::Attribute &Attr) {
    if (lvl < level::DEBUG) {
      return;
    }
    std::string str;
    llvm::raw_string_ostream ss(str);
    ss << Attr;
    spdlog::debug(str);
  }

  void log_instruction(const llvm::Instruction *inst) {
    if (lvl < level::DEBUG) {
      return;
    }
    if (inst->getFunction() != current_function) {
      current_function = inst->getFunction();
      spdlog::debug("{}:", current_function->getName().str());
    }
    if (inst->getParent() != current_block) {
      current_block = inst->getParent();
      spdlog::debug("\t{}: ", current_block->getNameOrAsOperand());
    }
    std::string str;
    llvm::raw_string_ostream ss(str);
    inst->print(ss);
    spdlog::debug("\t\t{}", str);
  }

  template<typename T>
  void operator<<(const T &Val) {
    if (lvl < level::DEBUG) {
      return;
    }
    std::string str;
    llvm::raw_string_ostream ss(str);
    ss << Val;
    spdlog::debug("{}", str);
  }

  template<typename T>
  void operator<<(const llvm::ArrayRef<T> &Val) {
    if (lvl < level::DEBUG) {
      return;
    }
    std::string str;
    llvm::raw_string_ostream ss(str);
    ss << "[";
    for (auto V : Val) {
      ss << V << ", ";
    }
    ss << "]";
    spdlog::debug("{}", str);
  }

private:
  const llvm::BasicBlock *current_block = nullptr;
  const llvm::Function *current_function = nullptr;
  level lvl = level::INFO;
};
} // namespace blueprint
} // namespace nil

#endif // CRYPTO3_ASSIGNER_NIL_BLUEPRINT_LOGGER_HPP
