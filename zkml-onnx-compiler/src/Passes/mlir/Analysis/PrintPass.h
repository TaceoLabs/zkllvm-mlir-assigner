#ifndef ZK_ML_TOOLCHAIN_PRINT_PASS
#define ZK_ML_TOOLCHAIN_PRINT_PASS

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "llvm/ADT/Hashing.h"
using namespace mlir;

namespace zk_ml_toolchain
{
  class PrintPass : public mlir::PassWrapper<PrintPass, mlir::OperationPass<>>
  {
    StringRef getArgument() const final;

    StringRef getDescription() const final;

    void runOnOperation() override;

    void printVector(std::vector<llvm::hash_code> &typeIds);

    void printOperation(Operation *op, std::vector<llvm::hash_code> &typeIds);

    void printRegion(Region &region, std::vector<llvm::hash_code> &typeIds);

    void printBlock(Block &block, std::vector<llvm::hash_code> &typeIds);

    int indent;

    struct IdentRAII
    {
      int &indent;
      IdentRAII(int &indent);
      ~IdentRAII();
    };

    void resetIndent();

    IdentRAII pushIndent();

    llvm::raw_ostream &printIndent();
  };

  std::unique_ptr<Pass> createPrintPass();

} // namespace zk_ml_toolchain

#endif // ZK_ML_TOOLCHAIN_PRINT_PASS