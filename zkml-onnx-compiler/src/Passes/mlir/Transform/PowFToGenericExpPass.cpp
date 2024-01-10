#include <Passes/mlir/Transform/PowFToGenericExpPass.hpp>

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using mlir::math::PowFOp;
namespace {
    struct PowFToGenericRewritePattern : public OpRewritePattern<PowFOp> {

        PowFToGenericRewritePattern(MLIRContext *context) : OpRewritePattern<PowFOp>(context, /*benefit=*/1) {
        }

        LogicalResult matchAndRewrite(PowFOp Op, PatternRewriter &Rewriter) const override {
            Location Loc = NameLoc::get(StringAttr::get(Op->getContext(), "math.powf"), Op->getLoc());
            // check if exp is a constant
            auto Base = Op->getOperand(0);
            auto Exp = Op->getOperand(1);
            Operation *DefiningOp = Exp.getDefiningOp();
            if (arith::ConstantFloatOp ConstOp = llvm::dyn_cast<arith::ConstantFloatOp>(*DefiningOp)) {
                APFloat Constant = ConstOp.value();
                double d = Constant.convertToDouble();
                if (d == 0.5 || d == 1.0 || d == 2.0 || d == 3.0)
                    llvm_unreachable("I do not think this can happen but we want to assert to catch IF it happens");
            }
            // just ordinary rewrite
            // a^b becomes exp(ln(a)*b)
            Value LnA = Rewriter.create<math::LogOp>(Loc, Base);
            Value NewExp = Rewriter.create<arith::MulFOp>(Loc, LnA, Exp);
            Value Result = Rewriter.create<math::ExpOp>(Loc, NewExp);
            Rewriter.replaceOp(Op, Result);
            return success();
        }
    };
}    // namespace

void mlir::zk_ml::PowFToGenericExpPass::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<PowFToGenericRewritePattern>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}
std::unique_ptr<Pass> mlir::zk_ml::createPowFToGenericExpPass() {
    return std::make_unique<PowFToGenericExpPass>();
}
