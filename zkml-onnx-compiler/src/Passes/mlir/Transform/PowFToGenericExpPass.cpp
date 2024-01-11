#include <Passes/mlir/Transform/PowFToGenericExpPass.hpp>

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using mlir::math::PowFOp;
using mlir::math::Exp2Op;
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
            Rewriter.replaceOp(Op, Rewriter.create<math::ExpOp>(Loc, NewExp));
            return success();
        }
    };

    struct Exp2ToGenericRewritePattern : public OpRewritePattern<Exp2Op> {

        Exp2ToGenericRewritePattern(MLIRContext *context) : OpRewritePattern<Exp2Op>(context, /*benefit=*/1) {
        }

        LogicalResult matchAndRewrite(Exp2Op Op, PatternRewriter &Rewriter) const override {
            Location Loc = NameLoc::get(StringAttr::get(Op->getContext(), "math.exp2"), Op->getLoc());
            auto Exp = Op->getOperand(0);
            assert(Exp.getType().isa<FloatType>() && "Exponent must be float for exp2");
            FloatType FTy = Exp.getType().cast<FloatType>();
            // 2^b becomes exp(ln(2)*b)
            // so we create constant for ln(2) and go from there
            //0.6931471805599453094172321214581765680755001343602552541206800094
            // Value Ln2 = Rewriter.create<arith::ConstantFloatOp>(Loc, APFloat(FTy.getFloatSemantics(), "0.69314718055994530941"), FTy);
            // Value NewExp = Rewriter.create<arith::MulFOp>(Loc, Ln2, Exp);
            // Rewriter.replaceOp(Op, Rewriter.create<math::ExpOp>(Loc, NewExp));
            Value Base = Rewriter.create<arith::ConstantFloatOp>(Loc, APFloat(FTy.getFloatSemantics(), "2.0"), FTy);
            Value LnA = Rewriter.create<math::LogOp>(Loc, Base);
            Value NewExp = Rewriter.create<arith::MulFOp>(Loc, LnA, Exp);
            Rewriter.replaceOp(Op, Rewriter.create<math::ExpOp>(Loc, NewExp));
            llvm::errs() << "base: " << Base << "\n"; 
            return success();
        }
    };
}    // namespace

void mlir::zk_ml::PowFToGenericExpPass::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<PowFToGenericRewritePattern>(&getContext());
    patterns.add<Exp2ToGenericRewritePattern>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}
std::unique_ptr<Pass> mlir::zk_ml::createPowFToGenericExpPass() {
    return std::make_unique<PowFToGenericExpPass>();
}
