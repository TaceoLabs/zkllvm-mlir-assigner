#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "AffineFullUnrollPass.h"

using mlir::AffineForOp;
using mlir::loopUnrollFull;

void zk_ml::AffineFullUnrollPass::runOnOperation() {
  getOperation().walk([&](AffineForOp op) {
    if (failed(loopUnrollFull(op))) {
      op.emitError("unrolling failed");
      signalPassFailure();
    }
  });
}
std::unique_ptr<Pass> zk_ml::createFullUnrollPass() {
  return std::make_unique<AffineFullUnrollPass>();
}
