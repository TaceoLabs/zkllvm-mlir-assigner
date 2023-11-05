#ifndef ZK_ML_TOOLCHAIN_COUNT_PASS
#define ZK_ML_TOOLCHAIN_COUNT_PASS

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Support/MathExtras.h"
#include <unordered_map>
#include <map>
#include <unistd.h>
using namespace mlir;

#define AFFINE_FOR "affine.for"
#define AFFINE_IF "affine.if"
#define ARITH_CONST "arith.constant"

#define DEBUG_FLAG false
#define DEBUG(X)    \
    if (DEBUG_FLAG) \
    llvm::outs() << X << "\n"

namespace zk_ml_toolchain
{

    // TODO link to mlir-hlo so that we do not have to copy-paste

    int64_t evalAffineExpr(AffineExpr expr, ArrayRef<int64_t> dims, ArrayRef<int64_t> symbols);
    bool evalIntegerSet(IntegerSet set, ArrayRef<int64_t> dims, ArrayRef<int64_t> symbols);
    bool evalIntegerSet(IntegerSet set, ArrayRef<int64_t> operands);
    SmallVector<int64_t> evalAffineMap(AffineMap map, ArrayRef<int64_t> dims, ArrayRef<int64_t> symbols);
    llvm::SmallVector<int64_t> evalAffineMap(AffineMap map, ArrayRef<int64_t> operands);

    // END COPY

    class CountPass
        : public mlir::PassWrapper<CountPass, mlir::OperationPass<>>
    {
    private:
        unsigned indent = 0;
        std::unordered_map<std::string, unsigned> counter;
        std::map<llvm::hash_code, int64_t> values;

        StringRef getArgument() const final;
        StringRef getDescription() const final;
        void runOnOperation() override;

        template <class T>
        T castFromAttr(Attribute attr);

        int64_t getMaxFromVector(llvm::SmallVector<int64_t> v);
        int64_t getMinFromVector(llvm::SmallVector<int64_t> v);

        void printIndent(unsigned offset = 0);

        void doAffineFor(Operation *op, int64_t from, int64_t to, int64_t step);

        template <class T>
        void printSmallvector(llvm::SmallVector<T> &v);

        int64_t evaluateForParameter(AffineMap &affineMap, llvm::SmallVector<Value> &operands, bool from);

        void countDepth(Operation *op);

        void countRegion(Region &region);

        void countBlock(Block &block);
    };

    std::unique_ptr<Pass> createCountPass();
} // namespace zk_ml_toolchain

#endif // ZK_ML_TOOLCHAIN_COUNT_PASS