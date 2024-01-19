#ifndef ZK_ML_TOOLCHAIN_ADD_DEBUG_PASS
#define ZK_ML_TOOLCHAIN_ADD_DEBUG_PASS

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"

namespace mlir {
    namespace zk_ml {
        class AddTracingOperationsPass : public mlir::PassWrapper<AddTracingOperationsPass, OperationPass<func::FuncOp>> {
        private:
            void runOnOperation() override;    

            StringRef getArgument() const final {
                return "add-debug-output-pass";
            }

            StringRef getDescription() const final {
                return "Adds debug operations after every onnx operation";
            }
        };
        std::unique_ptr<mlir::Pass> createAddTracingOperationsPass();
    }    // namespace zk_ml
}    // namespace mlir
#endif    // ZK_ML_TOOLCHAIN_ADD_DEBUG_PASS
