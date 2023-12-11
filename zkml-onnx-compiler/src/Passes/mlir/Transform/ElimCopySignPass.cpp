
#include "ElimCopySignPass.h"

StringRef zk_ml_toolchain::ElimCopySignPass::getArgument() const {
    return "elim-copysign-pass";
}

StringRef zk_ml_toolchain::ElimCopySignPass::getDescription() const {
    return "Eliminates redundant copysign operations that follow an frem operation";
}

void zk_ml_toolchain::ElimCopySignPass::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ElimCopySign>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<Pass> zk_ml_toolchain::createElimCopySignPass() {
    return std::make_unique<ElimCopySignPass>();
}
