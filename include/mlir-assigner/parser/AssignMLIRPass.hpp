#ifndef ZK_ML_TOOLCHAIN_ASSIGN_MLIR_PASS
#define ZK_ML_TOOLCHAIN_ASSIGN_MLIR_PASS

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Support/MathExtras.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include <mlir-assigner/memory/memref.hpp>

#include <unordered_map>
#include <map>
#include <unistd.h>
using namespace mlir;

#define AFFINE_FOR "affine.for"
#define AFFINE_IF "affine.if"
#define ARITH_CONST "arith.constant"

#define DEBUG_FLAG false
#define DEBUG(X)                                                               \
  if (DEBUG_FLAG)                                                              \
  llvm::outs() << X << "\n"

namespace zk_ml_toolchain {

namespace detail {

int64_t evalAffineExpr(AffineExpr expr, ArrayRef<int64_t> dims,
                       ArrayRef<int64_t> symbols) {
  int64_t lhs = 0, rhs = 0;
  if (auto bin = expr.dyn_cast<AffineBinaryOpExpr>()) {
    lhs = evalAffineExpr(bin.getLHS(), dims, symbols);
    rhs = evalAffineExpr(bin.getRHS(), dims, symbols);
  }
  switch (expr.getKind()) {
  case AffineExprKind::Add:
    return lhs + rhs;
  case AffineExprKind::Mul:
    return lhs * rhs;
  case AffineExprKind::Mod:
    return mod(lhs, rhs);
  case AffineExprKind::FloorDiv:
    return floorDiv(lhs, rhs);
  case AffineExprKind::CeilDiv:
    return ceilDiv(lhs, rhs);
  case AffineExprKind::Constant:
    return expr.cast<AffineConstantExpr>().getValue();
  case AffineExprKind::DimId:
    return dims[expr.cast<AffineDimExpr>().getPosition()];
  case AffineExprKind::SymbolId:
    return symbols[expr.cast<AffineSymbolExpr>().getPosition()];
  default:
    llvm_unreachable("must be one of AffineExprKind");
  }
}

bool evalIntegerSet(IntegerSet set, ArrayRef<int64_t> dims,
                    ArrayRef<int64_t> symbols) {
  // according to mlir/lib/IR/IntegerSetDetail.h constraints are either
  // an equality (affine_expr == 0) or an inequality (affine_expr >= 0).
  // Nevertheless, according to https://mlir.llvm.org/docs/Dialects/Affine/
  // a Constraint in an IntegerSet may be one of:
  //    affine_expr <= affine_expr
  //    affine_expr == affine_expr
  //    affine_expr >= affine_expr
  // we have to stick to code anyway but somehow strange
  ArrayRef<AffineExpr> constraints = set.getConstraints();
  for (unsigned i = 0; i < constraints.size(); ++i) {
    int64_t constraint = evalAffineExpr(constraints[i], dims, symbols);
    if (set.isEq(i)) {
      llvm::outs() << "we have a equality????\n";
      exit(-1);
    } else {
      if (constraint < 0) {
        return false;
      }
    }
  }
  return true;
}
bool evalIntegerSet(IntegerSet set, ArrayRef<int64_t> operands) {
  return evalIntegerSet(set, operands.take_front(set.getNumDims()),
                        operands.drop_front(set.getNumDims()));
}
SmallVector<int64_t> evalAffineMap(AffineMap map, ArrayRef<int64_t> dims,
                                   ArrayRef<int64_t> symbols) {
  SmallVector<int64_t> result;
  for (auto expr : map.getResults()) {
    result.push_back(evalAffineExpr(expr, dims, symbols));
  }
  return result;
}

llvm::SmallVector<int64_t> evalAffineMap(AffineMap map,
                                         ArrayRef<int64_t> operands) {
  return evalAffineMap(map, operands.take_front(map.getNumDims()),
                       operands.drop_front(map.getNumDims()));
}

int64_t getMaxFromVector(llvm::SmallVector<int64_t> v) {
  assert(!v.empty());
  int64_t currentMax = v[0];
  for (unsigned i = 1; i < v.size(); ++i) {
    if (currentMax < v[i])
      currentMax = v[i];
  }
  return currentMax;
}
int64_t getMinFromVector(llvm::SmallVector<int64_t> v) {
  assert(!v.empty());
  int64_t currentMin = v[0];
  for (unsigned i = 1; i < v.size(); ++i) {
    if (currentMin > v[i])
      currentMin = v[i];
  }
  return currentMin;
}

template <class T> T castFromAttr(Attribute attr) {
  T result = llvm::dyn_cast<T>(attr);
  assert(result);
  return result;
}

template <class T> void printSmallvector(llvm::SmallVector<T> &v) {
  if (DEBUG_FLAG) {
    llvm::outs() << "v[";
    for (auto c : v)
      llvm::outs() << c << ",";
    llvm::outs() << "]\n";
  }
}
} // namespace detail

using namespace detail;

template <typename VarType>
class AssignMLIRPass
    : public mlir::PassWrapper<AssignMLIRPass<VarType>, mlir::OperationPass<>> {

private:
  unsigned indent = 0;
  std::unordered_map<std::string, unsigned> counter;
  std::map<llvm::hash_code, int64_t> values;
  std::unordered_map<mlir::Value *, nil::blueprint::memref<VarType>> memrefs;
  std::unordered_map<mlir::Value *, VarType> locals;

  //   std::unordered_map<> memrefs;

  virtual StringRef getArgument() const final { return "assign-mlir"; }
  virtual StringRef getDescription() const final {
    return "Assigns the MLIR to a Blueprint trace.";
  }
  void runOnOperation() override {
    Operation *op = this->getOperation();
    handleOperation(op);
    for (auto elem : this->counter) {
      llvm::outs() << elem.first << ": " << elem.second << "\n";
    }
  }

  void doAffineFor(Operation *op, int64_t from, int64_t to, int64_t step) {
    assert(from < to);
    assert(step);
    assert(op->getRegions().size() == 1);
    assert(op->getRegions()[0].hasOneBlock());
    assert(op->getRegions()[0].getArguments().size() == 1);
    DEBUG("for (" << from << "->" << to << " step " << step << ")");
    indent++;
    llvm::hash_code counterHash =
        hash_value(op->getRegions()[0].getArguments()[0]);
    DEBUG("inserting hash: " << counterHash << ":" << from);
    this->values.insert(std::make_pair(counterHash, from));
    while (from < to) {
      for (Region &region : op->getRegions())
        handleRegion(region);
      from += step;
      DEBUG("updating hash: " << counterHash << ":" << from);
      this->values.insert(std::make_pair(counterHash, from));
      DEBUG(from << "->" << to);
      DEBUG("for done! go next iteration..");
    }
    this->values.erase(counterHash);
    DEBUG("deleting: " << counterHash);
    indent--;
  }

  int64_t evaluateForParameter(AffineMap &affineMap,
                               llvm::SmallVector<Value> &operands, bool from) {
    if (affineMap.isConstant()) {
      return affineMap.getResult(0).cast<AffineConstantExpr>().getValue();
    } else {
      assert(affineMap.getNumInputs() == operands.size());
      llvm::SmallVector<int64_t> inVector(affineMap.getNumInputs());
      for (unsigned i = 0; i < affineMap.getNumInputs(); ++i) {
        llvm::hash_code hash = hash_value(operands[i]);
        DEBUG("looking for: " << hash);
        if (values.find(hash) == values.end()) {
          DEBUG(affineMap);
          DEBUG("CANNOT FIND " << hash_value(operands[i]));
          DEBUG("CANNOT FIND " << operands[i]);
          exit(0);
        } else {
          assert(values.find(hash) != values.end());
          assert(values.count(hash));
          inVector[i] = this->values[hash];
        }
      }
      llvm::SmallVector<int64_t> eval = evalAffineMap(affineMap, inVector);
      return from ? getMaxFromVector(eval) : getMinFromVector(eval);
    }
  }

  void handleAffineOperation(Operation *op) {
    // Print the operation itself and some of its properties
    // Print the operation attributes
    std::string opName = op->getName().getIdentifier().str();
    // printIndent();
    //  DEBUG("visiting " << opName);
    if (AffineForOp operation = llvm::dyn_cast<AffineForOp>(op)) {
      DEBUG("visiting affine for!");
      assert(op->getAttrs().size() == 3);
      AffineMap fromMap = operation.getLowerBoundMap();
      int64_t step = operation.getStep();
      AffineMap toMap = operation.getUpperBoundMap();
      assert(fromMap.getNumInputs() + toMap.getNumInputs() ==
             op->getNumOperands());

      auto operandsFrom = operation.getLowerBoundOperands();
      auto operandsTo = operation.getUpperBoundOperands();
      auto operandsFromV =
          llvm::SmallVector<Value>(operandsFrom.begin(), operandsFrom.end());
      auto operandsToV =
          llvm::SmallVector<Value>(operandsTo.begin(), operandsTo.end());
      int64_t from = evaluateForParameter(fromMap, operandsFromV, true);
      int64_t to = evaluateForParameter(toMap, operandsToV, false);
      doAffineFor(op, from, to, step);
    } else if (AffineLoadOp operation = llvm::dyn_cast<AffineLoadOp>(op)) {
      llvm::outs() << operation << "\n";
      llvm::outs() << operation.getMemref() << "\n";
      for (auto a : operation.getIndices()) {
        llvm::outs() << a << "\n";
      }
      llvm::outs() << operation.getResult() << "\n";
    } else if (AffineStoreOp operation = llvm::dyn_cast<AffineStoreOp>(op)) {
      llvm::outs() << operation << "\n";
    } else if (AffineYieldOp operation = llvm::dyn_cast<AffineYieldOp>(op)) {
      // Affine Yields are Noops for us
    } else if (opName == AFFINE_IF) {
      DEBUG("visiting affine if!");
      assert(op->getAttrs().size() == 1);
      IntegerSet condition =
          castFromAttr<IntegerSetAttr>(op->getAttrs()[0].getValue()).getValue();
      //  IntegerSet condition = op->getAttrs()[0].getValue();
      //  assert(op->getNumOperands() == condition.getNumInputs());
      llvm::SmallVector<int64_t> operands(op->getNumOperands());
      DEBUG(op->getAttrs()[0].getValue());
      int i = 0;
      for (auto operand : op->getOperands()) {
        llvm::hash_code hash = hash_value(operand);
        assert(values.find(hash) != values.end());
        assert(values.count(hash));
        int64_t test = this->values[hash];
        operands[i++] = test;
      }
      if (evalIntegerSet(condition, operands)) {
        handleRegion(op->getRegion(0));
      } else {
        handleRegion(op->getRegion(1));
      }
    } else if (opName == "affine.apply" || opName == "affine.min") {
      DEBUG("got affine.apply");
      assert(op->getResults().size() == 1);
      assert(op->getAttrs().size() == 1);
      AffineMap applyMap =
          castFromAttr<AffineMapAttr>(op->getAttrs()[0].getValue())
              .getAffineMap();
      llvm::SmallVector<Value> operands(op->getOperands().begin(),
                                        op->getOperands().end());
      int64_t result = evaluateForParameter(applyMap, operands, false);
      values.insert(std::make_pair(hash_value(op->getResults()[0]), result));
    } else if (opName == "affine.max") {
      DEBUG("got affine.apply");
      assert(op->getResults().size() == 1);
      assert(op->getAttrs().size() == 1);
      AffineMap applyMap =
          castFromAttr<AffineMapAttr>(op->getAttrs()[0].getValue())
              .getAffineMap();
      llvm::SmallVector<Value> operands(op->getOperands().begin(),
                                        op->getOperands().end());
      int64_t result = evaluateForParameter(applyMap, operands, true);
      values.insert(std::make_pair(hash_value(op->getResults()[0]), result));
    } else {
      UNREACHABLE(std::string("unhandled affine operation: ") + opName);
    }
  }

  void handleMemRefOperation(Operation *op) {

    if (memref::AllocOp operation = llvm::dyn_cast<memref::AllocOp>(op)) {
      llvm::outs() << "allocating memref\n";
      MemRefType type = operation.getType();
      llvm::outs() << type.getElementType() << "\n";
      for (auto dim : type.getShape()) {
        llvm::outs() << dim << "\n";
      }
      auto uses = operation->getResult(0).getUsers();
      for (auto use : uses) {
        llvm::outs() << "use: " << use << "\n";
      }
      auto m = nil::blueprint::memref<VarType>(type.getShape(),
                                               type.getElementType());
      // memrefs.insert({operation.getMemref(), m});
    } else {
      UNREACHABLE("unhandled memref operation");
    }
  }

  void handleOperation(Operation *op) {
    Dialect *dial = op->getDialect();
    if (!dial) {
      llvm::outs() << "Encountered an unregistered Dialect\n";
      exit(-1);
    }

    if (llvm::isa<AffineDialect>(dial)) {
      handleAffineOperation(op);
      return;
    }

    if (llvm::isa<mlir::memref::MemRefDialect>(dial)) {
      handleMemRefOperation(op);
      return;
    }

    std::string opName = op->getName().getIdentifier().str();
    if (opName == ARITH_CONST) {
      assert(op->getNumResults() == 1);
      assert(op->getAttrs().size() == 1);
      Attribute contantValue = op->getAttrs()[0].getValue();
      if (contantValue.isa<IntegerAttr>()) {
        int64_t value = llvm::dyn_cast<IntegerAttr>(contantValue).getInt();
        values.insert(std::make_pair(hash_value(op->getResult(0)), value));
      } else {
        DEBUG("ignoring non int constant");
      }
    } else {
      auto operationIter = this->counter.find(opName);
      if (operationIter != this->counter.end()) {
        (*operationIter).second++;
        // std::cout << "increasing " << opName << std::endl;
      } else {
        this->counter.insert(std::make_pair(opName, 1));
        // std::cout << "inserting " << opName << std::endl;
      }

      // Recurse into each of the regions attached to the operation.
      for (Region &region : op->getRegions())
        handleRegion(region);
    }
  }

  void handleRegion(Region &region) {
    for (Block &block : region.getBlocks())
      handleBlock(block);
  }

  void handleBlock(Block &block) {
    for (Operation &op : block.getOperations())
      handleOperation(&op);
  }
};

template <typename VarType> std::unique_ptr<Pass> createAssignMLIRPass() {
  return std::make_unique<AssignMLIRPass<VarType>>();
}
} // namespace zk_ml_toolchain

#endif // ZK_ML_TOOLCHAIN_ASSIGN_MLIR_PASS
