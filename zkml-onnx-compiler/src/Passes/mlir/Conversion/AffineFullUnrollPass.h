#pragma once
#include "mlir/Pass/Pass.h"

// This class unrolls all affine for loops using the C++ API. Usually, this
// is NOT the correct way to handle things, but for more minor transformations
// and analyses it is sufficient.
// 
// It can be seen as a tutorial pass. Check the AffineFullUnrollPattern
// for how to use the Rewrite Engine.

using namespace mlir;
namespace zk_ml {
  class AffineFullUnrollPass: public PassWrapper<AffineFullUnrollPass, OperationPass<func::FuncOp>> {
  private:
    void runOnOperation() override;  // implemented in AffineFullUnroll.cpp

    StringRef getArgument() const final { return "affine-full-unroll"; }

    StringRef getDescription() const final {
      return "Fully unroll all affine loops";
    }
  };
  std::unique_ptr<mlir::Pass> createFullUnrollPass();
}// zk ml
