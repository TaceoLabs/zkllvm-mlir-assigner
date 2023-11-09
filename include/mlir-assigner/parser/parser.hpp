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

#ifndef CRYPTO3_BLUEPRINT_COMPONENT_INSTRUCTION_PARSER_HPP
#define CRYPTO3_BLUEPRINT_COMPONENT_INSTRUCTION_PARSER_HPP

#include <stack>
#include <variant>

#include <nil/blueprint/blueprint/plonk/assignment.hpp>
#include <nil/blueprint/blueprint/plonk/circuit.hpp>

#include "llvm/Support/Path.h"
#include <llvm/Support/SourceMgr.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>

// #include <mlir-assigner/helper/input_reader.hpp>
#include <mlir-assigner/helper/logger.hpp>

// ONNX_MLIR stuff, fix include paths
#include "src/Compiler/CompilerUtils.hpp"

// passes
#include <mlir-assigner/parser/AssignMLIRPass.hpp>

#include <mlir-assigner/policy/policy_manager.hpp>

namespace nil {
namespace blueprint {

template <typename BlueprintFieldType, typename ArithmetizationParams,
          bool PrintCircuitOutput>
struct parser {

  parser(long stack_size, bool detailed_logging, const std::string &kind = "") {
    if (detailed_logging) {
      log.set_level(logger::level::DEBUG);
    }
    detail::PolicyManager::set_policy(kind);
    onnx_mlir::registerDialects(context);
  }

  using ArithmetizationType =
      crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType,
                                                  ArithmetizationParams>;
  using var = crypto3::zk::snark::plonk_variable<
      typename BlueprintFieldType::value_type>;

  circuit<ArithmetizationType> bp;
  assignment<ArithmetizationType> assignmnt;

private:
public:
  mlir::OwningOpRef<mlir::ModuleOp>
  parseMLIRFile(const std::string &inputFilename) {
    mlir::OwningOpRef<mlir::ModuleOp> module;
    // Handle '.mlir' input to the ONNX-MLIR frontend.
    // The mlir format indicates that one or more of the supported
    // representations are used in the file.
    std::string errorMessage;
    auto input = mlir::openInputFile(inputFilename, &errorMessage);
    if (!input) {
      llvm::errs() << errorMessage << "\n";
      exit(1);
    }

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;
    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
    module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
      llvm::errs() << "Error can't load file " << inputFilename << "\n";
      exit(1);
    }
    return module;
  }

  bool evaluate(const mlir::ModuleOp &module,
                const boost::json::array &public_input) {

    // Initialize undef and zero vars once
    undef_var = put_into_assignment(typename BlueprintFieldType::value_type());
    zero_var = put_into_assignment(typename BlueprintFieldType::value_type(0));

    mlir::PassManager pm(&context, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(zk_ml_toolchain::createAssignMLIRPass<BlueprintFieldType,
                                                     ArithmetizationParams>(
        bp, assignmnt, public_input));
    if (mlir::failed(pm.run(module))) {
      llvm::errs() << "Passmanager failed to run!\n";
      return false;
    }

    llvm::outs() << assignmnt.rows_amount() << " rows\n";

    return true;
  }

  template <typename InputType> var put_into_assignment(InputType input) {
    assignmnt.public_input(0, public_input_idx) = input;
    return var(0, public_input_idx++, false, var::column_type::public_input);
  }

private:
  mlir::MLIRContext context;
  //   const llvm::BasicBlock *predecessor = nullptr;
  //   std::stack<stack_frame<var>> call_stack;
  //   std::unordered_map<const llvm::Value *, var> globals;
  //   std::unordered_map<const llvm::BasicBlock *, var> labels;
  bool finished = false;
  size_t public_input_idx = 0;
  //   std::unique_ptr<LayoutResolver> layout_resolver;
  var undef_var;
  var zero_var;
  logger log;
};

} // namespace blueprint
} // namespace nil

#endif // CRYPTO3_BLUEPRINT_COMPONENT_INSTRUCTION_PARSER_HPP
