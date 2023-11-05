#include <mlir-assigner/parser/CountPass.hpp>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

int64_t zk_ml_toolchain::evalAffineExpr(AffineExpr expr, ArrayRef<int64_t> dims,
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

bool zk_ml_toolchain::evalIntegerSet(IntegerSet set, ArrayRef<int64_t> dims,
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
bool zk_ml_toolchain::evalIntegerSet(IntegerSet set,
                                     ArrayRef<int64_t> operands) {
  return evalIntegerSet(set, operands.take_front(set.getNumDims()),
                        operands.drop_front(set.getNumDims()));
}
SmallVector<int64_t> zk_ml_toolchain::evalAffineMap(AffineMap map,
                                                    ArrayRef<int64_t> dims,
                                                    ArrayRef<int64_t> symbols) {
  SmallVector<int64_t> result;
  for (auto expr : map.getResults()) {
    result.push_back(evalAffineExpr(expr, dims, symbols));
  }
  return result;
}

llvm::SmallVector<int64_t>
zk_ml_toolchain::evalAffineMap(AffineMap map, ArrayRef<int64_t> operands) {
  return evalAffineMap(map, operands.take_front(map.getNumDims()),
                       operands.drop_front(map.getNumDims()));
}

// END COPY

StringRef zk_ml_toolchain::CountPass::getArgument() const {
  return "count-pass";
}
StringRef zk_ml_toolchain::CountPass::getDescription() const {
  return "Does some counting - lets see what";
}
void zk_ml_toolchain::CountPass::runOnOperation() {
  Operation *op = getOperation();
  countDepth(op);
  for (auto elem : this->counter) {
    llvm::outs() << elem.first << ": " << elem.second << "\n";
  }
}

template <class T> T zk_ml_toolchain::CountPass::castFromAttr(Attribute attr) {
  T result = llvm::dyn_cast<T>(attr);
  assert(result);
  return result;
}

int64_t
zk_ml_toolchain::CountPass::getMaxFromVector(llvm::SmallVector<int64_t> v) {
  assert(!v.empty());
  int64_t currentMax = v[0];
  for (unsigned i = 1; i < v.size(); ++i) {
    if (currentMax < v[i])
      currentMax = v[i];
  }
  return currentMax;
}
int64_t
zk_ml_toolchain::CountPass::getMinFromVector(llvm::SmallVector<int64_t> v) {
  assert(!v.empty());
  int64_t currentMin = v[0];
  for (unsigned i = 1; i < v.size(); ++i) {
    if (currentMin > v[i])
      currentMin = v[i];
  }
  return currentMin;
}

void zk_ml_toolchain::CountPass::printIndent(unsigned offset) {
  if (DEBUG_FLAG) {
    assert(indent >= offset);
    for (unsigned i = 0; i < indent - offset; ++i)
      llvm::outs() << "  ";
  }
}

void zk_ml_toolchain::CountPass::doAffineFor(Operation *op, int64_t from,
                                             int64_t to, int64_t step) {
  assert(from < to);
  assert(step);
  assert(op->getRegions().size() == 1);
  assert(op->getRegions()[0].hasOneBlock());
  assert(op->getRegions()[0].getArguments().size() == 1);
  printIndent();
  DEBUG("for (" << from << "->" << to << " step " << step << ")");
  indent++;
  llvm::hash_code counterHash =
      hash_value(op->getRegions()[0].getArguments()[0]);
  DEBUG("inserting hash: " << counterHash << ":" << from);
  this->values.insert(std::make_pair(counterHash, from));
  while (from < to) {
    for (Region &region : op->getRegions())
      countRegion(region);
    from += step;
    DEBUG("updating hash: " << counterHash << ":" << from);
    this->values.insert(std::make_pair(counterHash, from));
    printIndent(1);
    DEBUG(from << "->" << to);
    DEBUG("for done! go next iteration..");
  }
  this->values.erase(counterHash);
  DEBUG("deleting: " << counterHash);
  indent--;
}

template <class T>
void zk_ml_toolchain::CountPass::printSmallvector(llvm::SmallVector<T> &v) {
  if (DEBUG_FLAG) {
    llvm::outs() << "v[";
    for (auto c : v)
      llvm::outs() << c << ",";
    llvm::outs() << "]\n";
  }
}

int64_t zk_ml_toolchain::CountPass::evaluateForParameter(
    AffineMap &affineMap, llvm::SmallVector<Value> &operands, bool from) {
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

void zk_ml_toolchain::CountPass::countDepth(Operation *op) {

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
      countRegion(op->getRegion(0));
    } else {
      countRegion(op->getRegion(1));
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
  } else if (opName == ARITH_CONST) {
    assert(op->getNumResults() == 1);
    assert(op->getAttrs().size() == 1);
    Attribute contantValue = op->getAttrs()[0].getValue();
    if (contantValue.isa<IntegerAttr>()) {
      int64_t value = llvm::dyn_cast<IntegerAttr>(contantValue).getInt();
      values.insert(std::make_pair(hash_value(op->getResult(0)), value));
    } else {
      DEBUG("ignoring non int constant");
    }
  } else if (memref::AllocOp operation = llvm::dyn_cast<memref::AllocOp>(op)) {
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
      countRegion(region);
  }
}

void zk_ml_toolchain::CountPass::countRegion(Region &region) {
  for (Block &block : region.getBlocks())
    countBlock(block);
}

void zk_ml_toolchain::CountPass::countBlock(Block &block) {
  for (Operation &op : block.getOperations())
    countDepth(&op);
}

std::unique_ptr<Pass> zk_ml_toolchain::createCountPass() {
  return std::make_unique<CountPass>();
}
