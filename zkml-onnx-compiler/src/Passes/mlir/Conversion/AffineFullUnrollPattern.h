#pragma once

#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace zk_ml {

    class AffineFullUnrollPassAsPatternRewrite: public PassWrapper<AffineFullUnrollPassAsPatternRewrite, OperationPass<func::FuncOp>> {
    private:
      void runOnOperation() override;  

      StringRef getArgument() const final { return "affine-full-unroll-with-pattern"; }

      StringRef getDescription() const final {
        return "Fully unroll all affine loops with pattern rewrite";
      }
    };
    std::unique_ptr<mlir::Pass> createFullUnrollPassPatternRewriter();

}// zk ml
