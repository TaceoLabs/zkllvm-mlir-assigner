#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "AffineFullUnrollPattern.h"

using mlir::AffineForOp;
using mlir::loopUnrollFull;

namespace {
    struct AffineFullUnrollPattern : public OpRewritePattern<AffineForOp> {

        AffineFullUnrollPattern(MLIRContext *context) : OpRewritePattern<AffineForOp>(context, /*benefit=*/1) {
        }

        LogicalResult matchAndRewrite(AffineForOp op, PatternRewriter &rewriter) const override {
            return loopUnrollFull(op);
        }
    };
}    // namespace

void zk_ml::AffineFullUnrollPassAsPatternRewrite::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<AffineFullUnrollPattern>(&getContext());
    // One could use GreedyRewriteConfig here to slightly tweak the behavior of
    // the pattern application.
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}
std::unique_ptr<Pass> zk_ml::createFullUnrollPassPatternRewriter() {
    return std::make_unique<AffineFullUnrollPassAsPatternRewrite>();
}
