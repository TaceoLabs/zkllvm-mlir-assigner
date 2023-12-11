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

#include <boost/log/trivial.hpp>

#include <nil/blueprint/blueprint/plonk/assignment_proxy.hpp>
#include <nil/blueprint/blueprint/plonk/circuit_proxy.hpp>

#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>

// #include <mlir-assigner/helper/input_reader.hpp>
#include <mlir-assigner/helper/logger.hpp>

// ONNX_MLIR stuff, fix include paths
#include "evaluator.hpp"
#include "src/Compiler/CompilerUtils.hpp"
#include "mlir/Dialect/zkml/ZkMlDialect.h"

#include <mlir-assigner/parser/evaluator.hpp>

#include <mlir-assigner/policy/policy_manager.hpp>

namespace nil {
namespace blueprint {

template <typename BlueprintFieldType, typename ArithmetizationParams,
          bool PrintCircuitOutput>
struct parser {

  parser(long stack_size, boost::log::trivial::severity_level log_level,
         std::uint32_t max_num_provers, const std::string &kind = "")
      : max_num_provers(max_num_provers) {
    if (max_num_provers != 1) {
      throw std::runtime_error("Currently only one prover is supported, please "
                               "set max_num_provers to 1");
    }
    log.set_level(log_level);
    detail::PolicyManager::set_policy(kind);

    onnx_mlir::registerDialects(context);
    context.getOrLoadDialect<zkml::ZkMlDialect>();

    assignment_ptr = std::make_shared<assignment<ArithmetizationType>>();
    bp_ptr = std::make_shared<circuit<ArithmetizationType>>();
    assignments.emplace_back(assignment_ptr, 0);
    circuits.emplace_back(bp_ptr, 0);
  }

  using ArithmetizationType =
      crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType,
                                                  ArithmetizationParams>;
  using var = crypto3::zk::snark::plonk_variable<
      typename BlueprintFieldType::value_type>;

  std::vector<circuit_proxy<ArithmetizationType>> circuits;
  std::vector<assignment_proxy<ArithmetizationType>> assignments;

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

  bool evaluate(mlir::OwningOpRef<mlir::ModuleOp> module,
                const boost::json::array &public_input) {

    zk_ml_toolchain::evaluator<BlueprintFieldType, ArithmetizationParams>
        evaluator(circuits[0], assignments[0], public_input, PrintCircuitOutput,
                  log);
    evaluator.evaluate(std::move(module));
    // if (mlir::failed(pm.run(module))) {
    //   llvm::errs() << "Passmanager failed to run!\n";
    //   return false;
    // }

    llvm::outs() << assignments[0].rows_amount() << " rows\n";

    return true;
  }

private:
  mlir::MLIRContext context;
  bool finished = false;
  logger log;
  std::uint32_t max_num_provers;
  std::shared_ptr<circuit<ArithmetizationType>> bp_ptr;
  std::shared_ptr<assignment<ArithmetizationType>> assignment_ptr;
};

} // namespace blueprint
} // namespace nil

#endif // CRYPTO3_BLUEPRINT_COMPONENT_INSTRUCTION_PARSER_HPP
