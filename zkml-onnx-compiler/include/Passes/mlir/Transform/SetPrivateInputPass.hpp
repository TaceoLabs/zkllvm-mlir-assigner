#ifndef ZK_ML_TOOLCHAIN_SET_PRIVATE_INPUT_PASS
#define ZK_ML_TOOLCHAIN_SET_PRIVATE_INPUT_PASS

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "src/Builder/FrontendDialectTransformer.hpp"
#include <cstddef>

namespace mlir {
    namespace zk_ml {
        class SetPrivateInputPass : public mlir::PassWrapper<SetPrivateInputPass, OperationPass<mlir::ModuleOp>> {
        public:
            SetPrivateInputPass() = delete;
            SetPrivateInputPass(std::vector<bool> &privateInputs) : privateInputs(privateInputs) {}

        private:
            std::vector<bool> privateInputs;

            void runOnOperation() override;

            StringRef getArgument() const final {
                return "set-private-input-pass";
            }

            StringRef getDescription() const final {
                return "Marks inputs to entry point function as private, depending on provided input";
            }
        };
        std::unique_ptr<mlir::Pass> createSetPrivateInputPass(std::vector<bool> &privateInputs);
    }    // namespace zk_ml
}    // namespace mlir
#endif    // ZK_ML_TOOLCHAIN_SET_PRIVATE_INPUT_PASS
