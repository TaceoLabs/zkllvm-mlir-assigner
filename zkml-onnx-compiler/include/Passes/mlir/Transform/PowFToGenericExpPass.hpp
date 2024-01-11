#ifndef ZK_ML_TOOLCHAIN_POWF_TO_GENERIC_EXP_PASS
#define ZK_ML_TOOLCHAIN_POWF_TO_GENERIC_EXP_PASS

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"

namespace mlir {
    namespace zk_ml {
        class PowFToGenericExpPass : public mlir::PassWrapper<PowFToGenericExpPass, OperationPass<mlir::func::FuncOp>> {
        private:
            void runOnOperation() override;    

            StringRef getArgument() const final {
                return "powf-to-generic-exp-pass";
            }

            StringRef getDescription() const final {
                return "Rewrites powf calls to generic form and transforms to simpler terms when exponent is {0.5, 2, 3}";
            }
        };
        std::unique_ptr<mlir::Pass> createPowFToGenericExpPass();
    }    // namespace zk_ml
}    // namespace mlir
#endif    // ZK_ML_TOOLCHAIN_POWF_TO_GENERIC_EXP_PASS
