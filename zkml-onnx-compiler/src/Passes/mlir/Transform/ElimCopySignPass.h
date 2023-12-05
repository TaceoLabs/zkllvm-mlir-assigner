#ifndef ZK_ML_TOOLCHAIN_ELIM_COPY_SIGN_PASS
#define ZK_ML_TOOLCHAIN_ELIM_COPY_SIGN_PASS

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <deque>

#include "llvm/ADT/Hashing.h"
using namespace mlir;

namespace zk_ml_toolchain
{
    class ElimCopySignPass : public mlir::PassWrapper<ElimCopySignPass, mlir::OperationPass<>>
    {
        StringRef getArgument() const final;

        StringRef getDescription() const final;

        void runOnOperation() override;
    };

    std::unique_ptr<mlir::Pass> createElimCopySignPass();

#include "ElimCopySign.cpp.inc"

} // namespace zk_ml_toolchain

#endif // ZK_ML_TOOLCHAIN_ELIM_COPY_SIGN_PASS
