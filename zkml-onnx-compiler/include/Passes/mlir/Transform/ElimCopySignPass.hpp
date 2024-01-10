#ifndef ZK_ML_TOOLCHAIN_ELIM_COPY_SIGN_PASS
#define ZK_ML_TOOLCHAIN_ELIM_COPY_SIGN_PASS

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
    namespace zk_ml {
        class ElimCopySignPass : public mlir::PassWrapper<ElimCopySignPass, mlir::OperationPass<>> {
        private:
            void runOnOperation() override;

            StringRef getArgument() const final {
                return "elim-copysign-pass";
            }
            StringRef getDescription() const final {
                return "Eliminates redundant copysign operations that follow an frem operation";
            }
        };
        std::unique_ptr<mlir::Pass> createElimCopySignPass();
#include "ElimCopySign.cpp.inc"

    }    // namespace zk_ml
}    //  namespace  mlir

#endif    // ZK_ML_TOOLCHAIN_ELIM_COPY_SIGN_PASS
