#ifndef ZK_ML_TOOLCHAIN_ELIM_COPY_SIGN_TD
#define ZK_ML_TOOLCHAIN_ELIM_COPY_SIGN_TD

include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Math/IR/MathOps.td"
include "mlir/IR/PatternBase.td"


def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;

def ElimCopySign : Pat<
  (Math_CopySignOp (Arith_RemFOp:$a $b, $c, $type_remf), $b, $type_cpysgn),
  (Arith_RemFOp $b, $c, $type_remf),
  [(HasOneUse:$a)]
>;

#endif // ZK_ML_TOOLCHAIN_ELIM_COPY_SIGN_TD
