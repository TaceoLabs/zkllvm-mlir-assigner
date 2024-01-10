/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* Generated from:
    ./src/Passes/ElimCopySign.td:11
*/
struct ElimCopySign : public ::mlir::RewritePattern {
  ElimCopySign(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("math.copysign", 2, context, {"arith.remf"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::FastMathFlagsAttr type_cpysgn;
    ::mlir::arith::FastMathFlagsAttr type_remf;
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range b0(op0->getOperands());
    ::mlir::arith::RemFOp a;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::math::CopySignOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::RemFOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::RemFOp type";
        });
      }
      a = castedOp1;
      b0 = castedOp1.getODSOperands(0);
      c = castedOp1.getODSOperands(1);
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::arith::FastMathFlagsAttr>("fastmath");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = ::mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), ::mlir::arith::FastMathFlags::none);
        type_remf = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }
    b = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::FastMathFlagsAttr>("fastmath");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), ::mlir::arith::FastMathFlags::none);
      type_cpysgn = tblgen_attr;
    }
    if (!(((*a.getODSResults(0).begin()).hasOneUse()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities '' failed to satisfy constraint: 'has one use'";
      });
    }
    if (!(*b.begin() == *b0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'b' and 'b0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::RemFOp tblgen_RemFOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*b.begin()));
      tblgen_values.push_back((*c.begin()));
      if (auto tmpAttr = type_remf) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("fastmath"), tmpAttr);
      }
      tblgen_RemFOp_0 = rewriter.create<::mlir::arith::RemFOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_RemFOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};
