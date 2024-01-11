#include <Passes/mlir/Transform/ElimCopySignPass.hpp>
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

void zk_ml::ElimCopySignPass::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ElimCopySign>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<Pass> mlir::zk_ml::createElimCopySignPass() {
    return std::make_unique<ElimCopySignPass>();
}
