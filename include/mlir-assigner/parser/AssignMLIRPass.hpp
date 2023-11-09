#ifndef ZK_ML_TOOLCHAIN_ASSIGN_MLIR_PASS
#define ZK_ML_TOOLCHAIN_ASSIGN_MLIR_PASS

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Support/MathExtras.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinDialect.h>

#include "src/Dialect/Krnl/KrnlOps.hpp"

#include <nil/blueprint/blueprint/plonk/assignment.hpp>
#include <nil/blueprint/blueprint/plonk/circuit.hpp>

#include <nil/blueprint/components/algebra/fixedpoint/type.hpp>

#include <mlir-assigner/components/fixedpoint/addition.hpp>
#include <mlir-assigner/components/fixedpoint/subtraction.hpp>
#include <mlir-assigner/components/fixedpoint/mul_rescale.hpp>
#include <mlir-assigner/components/fixedpoint/division.hpp>
#include <mlir-assigner/components/fixedpoint/exp.hpp>
#include <mlir-assigner/components/comparison/fixed_comparison.hpp>
#include <mlir-assigner/components/comparison/select.hpp>

#include <mlir-assigner/memory/memref.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>
#include <mlir-assigner/parser/input_reader.hpp>

#include <unordered_map>
#include <map>
#include <unistd.h>
using namespace mlir;

#define DEBUG_FLAG true
#define DEBUG(X)                                                               \
  if (DEBUG_FLAG)                                                              \
  llvm::outs() << X << "\n"

namespace zk_ml_toolchain {

namespace detail {

int64_t evalAffineExpr(AffineExpr expr, llvm::ArrayRef<int64_t> dims,
                       llvm::ArrayRef<int64_t> symbols) {
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

bool evalIntegerSet(IntegerSet set, llvm::ArrayRef<int64_t> dims,
                    llvm::ArrayRef<int64_t> symbols) {
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
bool evalIntegerSet(IntegerSet set, llvm::ArrayRef<int64_t> operands) {
  return evalIntegerSet(set, operands.take_front(set.getNumDims()),
                        operands.drop_front(set.getNumDims()));
}
SmallVector<int64_t> evalAffineMap(AffineMap map, llvm::ArrayRef<int64_t> dims,
                                   llvm::ArrayRef<int64_t> symbols) {
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

template <typename BlueprintFieldType, typename ArithmetizationParams>
class AssignMLIRPass
    : public mlir::PassWrapper<
          AssignMLIRPass<BlueprintFieldType, ArithmetizationParams>,
          mlir::OperationPass<>> {
public:
  using ArithmetizationType =
      nil::crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType,
                                                       ArithmetizationParams>;
  using VarType = nil::crypto3::zk::snark::plonk_variable<
      typename BlueprintFieldType::value_type>;

  AssignMLIRPass(nil::blueprint::circuit<ArithmetizationType> &circuit,
                 nil::blueprint::assignment<ArithmetizationType> &assignment,
                 const boost::json::array &public_input)
      : bp(circuit), assignmnt(assignment), public_input(public_input) {}

  // copy constructor
  AssignMLIRPass(const AssignMLIRPass &pass)
      : bp(pass.bp), assignmnt(pass.assignmnt),
        public_input(pass.public_input) {
    // since we misappropreate the pass runner for our assignment, assume it the
    // CC is not called we cannot delete it since it is required by the
    // PassWrapper
    UNREACHABLE("copy constructor should not be called atm");
  }
  AssignMLIRPass(AssignMLIRPass &&pass) = default;
  AssignMLIRPass &operator=(const AssignMLIRPass &pass) = delete;

private:
  virtual StringRef getArgument() const final { return "assign-mlir"; }
  virtual StringRef getDescription() const final {
    return "Assigns the MLIR to a Blueprint trace.";
  }
  void runOnOperation() override {
    Operation *op = this->getOperation();
    handleOperation(op);
  }

  void doAffineFor(AffineForOp &op, int64_t from, int64_t to, int64_t step) {
    ASSERT(from < to);
    ASSERT(step);
    // atm handle only simple loops with one region,block and argument
    ASSERT(op.getRegion().hasOneBlock());
    ASSERT(op.getRegion().getArguments().size() == 1);
    DEBUG("for (" << from << "->" << to << " step " << step << ")");
    llvm::hash_code counterHash = mlir::hash_value(op.getInductionVar());
    DEBUG("inserting hash: " << counterHash << ":" << from);
    auto res = frames.back().constant_values.insert({counterHash, from});
    ASSERT(res.second); // we do not want overrides here, since we delete it
                        // after loop this should never happen
    while (from < to) {
      handleRegion(op.getLoopBody());
      from += step;
      DEBUG("updating hash: " << counterHash << ":" << from);
      frames.back().constant_values[counterHash] = from;
      DEBUG(from << "->" << to);
      DEBUG("for done! go next iteration..");
    }
    frames.back().constant_values.erase(counterHash);
    DEBUG("deleting: " << counterHash);
  }

  int64_t evaluateForParameter(AffineMap &affineMap,
                               llvm::SmallVector<Value> &operands, bool from) {
    if (affineMap.isConstant()) {
      return affineMap.getResult(0).cast<AffineConstantExpr>().getValue();
    } else {
      assert(affineMap.getNumInputs() == operands.size());
      llvm::SmallVector<int64_t> inVector(affineMap.getNumInputs());
      for (unsigned i = 0; i < affineMap.getNumInputs(); ++i) {
        llvm::hash_code hash = mlir::hash_value(operands[i]);
        DEBUG("looking for: " << hash);
        if (frames.back().constant_values.find(hash) ==
            frames.back().constant_values.end()) {
          DEBUG(affineMap);
          DEBUG("CANNOT FIND " << mlir::hash_value(operands[i]));
          DEBUG("CANNOT FIND " << operands[i]);
          exit(0);
        } else {
          assert(frames.back().constant_values.find(hash) !=
                 frames.back().constant_values.end());
          assert(frames.back().constant_values.count(hash));
          inVector[i] = frames.back().constant_values[hash];
        }
      }
      llvm::SmallVector<int64_t> eval = evalAffineMap(affineMap, inVector);
      return from ? getMaxFromVector(eval) : getMinFromVector(eval);
    }
  }

  void handleArithOperation(Operation *op) {
    std::uint32_t start_row = assignmnt.allocated_rows();
    if (arith::AddFOp operation = llvm::dyn_cast<arith::AddFOp>(op)) {
      handle_fixedpoint_addition_component(operation, frames.back(), bp,
                                           assignmnt, start_row);
    } else if (arith::SubFOp operation = llvm::dyn_cast<arith::SubFOp>(op)) {
      handle_fixedpoint_subtraction_component(operation, frames.back(), bp,
                                              assignmnt, start_row);
    } else if (arith::MulFOp operation = llvm::dyn_cast<arith::MulFOp>(op)) {
      handle_fixedpoint_mul_rescale_component(operation, frames.back(), bp,
                                              assignmnt, start_row);
    } else if (arith::DivFOp operation = llvm::dyn_cast<arith::DivFOp>(op)) {
      handle_fixedpoint_division_component(operation, frames.back(), bp,
                                           assignmnt, start_row);
    } else if (arith::CmpFOp operation = llvm::dyn_cast<arith::CmpFOp>(op)) {
      handle_fixedpoint_comparison_component(operation, frames.back(), bp,
                                             assignmnt, start_row);
    } else if (arith::SelectOp operation =
                   llvm::dyn_cast<arith::SelectOp>(op)) {
      handle_select_component(operation, frames.back(), bp, assignmnt,
                              start_row);
    } else if (arith::ConstantOp operation =
                   llvm::dyn_cast<arith::ConstantOp>(op)) {
      TypedAttr constantValue = operation.getValueAttr();
      if (constantValue.isa<IntegerAttr>()) {
        int64_t value = llvm::dyn_cast<IntegerAttr>(constantValue).getInt();
        // this insert is ok, since this should never change, so we don't
        // override it if it is already there

        // TACEO_TODO: better separation of constant values that come from the
        // loop bounds an normal ones, ATM just do both
        frames.back().constant_values.insert(
            std::make_pair(mlir::hash_value(operation.getResult()), value));

        typename BlueprintFieldType::value_type field_constant = value;
        auto val = put_into_assignment(field_constant);
        frames.back().locals.insert(
            std::make_pair(mlir::hash_value(operation.getResult()), val));
      } else if (constantValue.isa<FloatAttr>()) {
        double d = llvm::dyn_cast<FloatAttr>(constantValue).getValueAsDouble();
        nil::blueprint::components::FixedPoint<BlueprintFieldType, 1, 1> fixed(
            d);
        auto value = put_into_assignment(fixed.get_value());
        // this insert is ok, since this should never change, so we
        // don't override it if it is already there
        frames.back().locals.insert(
            std::make_pair(mlir::hash_value(operation.getResult()), value));
      } else {
        llvm::outs() << constantValue.getType() << "\n";
        UNREACHABLE("unhandled constant");
      }
    } else {
      std::string opName = op->getName().getIdentifier().str();
      UNREACHABLE(std::string("unhandled arith operation: ") + opName);
    }
  }

  void handleMathOperation(Operation *op) {
    std::uint32_t start_row = assignmnt.allocated_rows();
    if (math::ExpOp operation = llvm::dyn_cast<math::ExpOp>(op)) {
      handle_fixedpoint_exp_component(operation, frames.back(), bp, assignmnt,
                                      start_row);
    } else if (math::SqrtOp operation = llvm::dyn_cast<math::SqrtOp>(op)) {
      UNREACHABLE("TODO: sqrt");
    } else {
      std::string opName = op->getName().getIdentifier().str();
      UNREACHABLE(std::string("unhandled math operation: ") + opName);
    }
  }

  void handleAffineOperation(Operation *op) {
    // Print the operation itself and some of its properties
    // Print the operation attributes
    std::string opName = op->getName().getIdentifier().str();
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
      doAffineFor(operation, from, to, step);
    } else if (AffineLoadOp operation = llvm::dyn_cast<AffineLoadOp>(op)) {
      // affine.load
      // llvm::outs() << "affine.load:" <<
      // mlir::hash_value(operation.getMemref())
      //              << "\n";
      auto memref =
          frames.back().memrefs.find(mlir::hash_value(operation.getMemref()));
      ASSERT(memref != frames.back().memrefs.end());

      // grab the indices and build index vector
      auto indices = operation.getIndices();
      std::vector<int64_t> indicesV;
      indicesV.reserve(indices.size());
      for (auto a : indices) {
        // look for indices in constant_values
        // llvm::outs() << a << "," << mlir::hash_value(a) << "\n";
        auto res = frames.back().constant_values.find(mlir::hash_value(a));
        ASSERT(res != frames.back().constant_values.end());
        // llvm::outs() << res->second << "\n";
        indicesV.push_back(res->second);
      }
      auto value = memref->second.get(indicesV);
      frames.back().locals[mlir::hash_value(operation.getResult())] = value;
    } else if (AffineStoreOp operation = llvm::dyn_cast<AffineStoreOp>(op)) {
      // affine.store
      auto memref =
          frames.back().memrefs.find(mlir::hash_value(operation.getMemref()));
      ASSERT(memref != frames.back().memrefs.end());

      // grab the indices and build index vector
      auto indices = operation.getIndices();
      std::vector<int64_t> indicesV;
      indicesV.reserve(indices.size());
      for (auto a : indices) {
        // look for indices in constant_values
        // llvm::outs() << a << "," << mlir::hash_value(a) << "\n";
        auto res = frames.back().constant_values.find(mlir::hash_value(a));
        ASSERT(res != frames.back().constant_values.end());
        // llvm::outs() << res->second << "\n";
        indicesV.push_back(res->second);
      }
      // grab the element from the locals array
      auto value =
          frames.back().locals.find(mlir::hash_value(operation.getValue()));
      ASSERT(value != frames.back().locals.end());
      // put the element from the memref using index vector
      memref->second.put(indicesV, value->second);

    } else if (AffineYieldOp operation = llvm::dyn_cast<AffineYieldOp>(op)) {
      // Affine Yields are Noops for us
    } else if (opName == "affine.if") {
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
        llvm::hash_code hash = mlir::hash_value(operand);
        assert(frames.back().constant_values.find(hash) !=
               frames.back().constant_values.end());
        assert(frames.back().constant_values.count(hash));
        int64_t test = frames.back().constant_values[hash];
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
      frames.back().constant_values[mlir::hash_value(op->getResults()[0])] =
          result;
    } else if (opName == "affine.max") {
      DEBUG("got affine.max");
      assert(op->getResults().size() == 1);
      assert(op->getAttrs().size() == 1);
      AffineMap applyMap =
          castFromAttr<AffineMapAttr>(op->getAttrs()[0].getValue())
              .getAffineMap();
      llvm::SmallVector<Value> operands(op->getOperands().begin(),
                                        op->getOperands().end());
      int64_t result = evaluateForParameter(applyMap, operands, true);
      frames.back().constant_values[mlir::hash_value(op->getResults()[0])] =
          result;
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
      auto res = operation->getResult(0);
      auto res2 = operation.getMemref();
      llvm::outs() << res << "\n";
      llvm::outs() << res2 << "\n";
      auto m = nil::blueprint::memref<VarType>(type.getShape(),
                                               type.getElementType());
      auto insert_res = frames.back().memrefs.insert(
          {mlir::hash_value(operation.getMemref()), m});
      ASSERT(insert_res.second); // Reallocating over an existing memref
                                 // should not happen ATM
    } else if (memref::AllocaOp operation =
                   llvm::dyn_cast<memref::AllocaOp>(op)) {
      // TACEO_TODO: handle cleanup of these stack memrefs
      // TACEO_TODO: deduplicate with above
      llvm::outs() << "allocating (stack) memref\n";
      MemRefType type = operation.getType();
      llvm::outs() << type.getElementType() << "\n";
      for (auto dim : type.getShape()) {
        llvm::outs() << dim << "\n";
      }
      auto uses = operation->getResult(0).getUsers();
      for (auto use : uses) {
        llvm::outs() << "use: " << use << "\n";
      }
      auto res = operation->getResult(0);
      auto res2 = operation.getMemref();
      llvm::outs() << res << "\n";
      llvm::outs() << res2 << "\n";
      auto m = nil::blueprint::memref<VarType>(type.getShape(),
                                               type.getElementType());
      auto insert_res = frames.back().memrefs.insert(
          {mlir::hash_value(operation.getMemref()), m});
      ASSERT(insert_res.second); // Reallocating over an existing memref
                                 // should not happen ATM
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

    if (llvm::isa<mlir::arith::ArithDialect>(dial)) {
      handleArithOperation(op);
      return;
    }

    if (llvm::isa<mlir::math::MathDialect>(dial)) {
      handleMathOperation(op);
      return;
    }

    if (llvm::isa<AffineDialect>(dial)) {
      handleAffineOperation(op);
      return;
    }

    if (llvm::isa<mlir::memref::MemRefDialect>(dial)) {
      handleMemRefOperation(op);
      return;
    }

    if (mlir::ModuleOp operation = llvm::dyn_cast<mlir::ModuleOp>(op)) {
      // this is the toplevel operation of the IR
      // TODO: handle attributes if needed
      handleRegion(operation.getBodyRegion());
      return;
    }

    if (func::FuncOp operation = llvm::dyn_cast<func::FuncOp>(op)) {
      auto res = functions.insert({operation.getSymName().str(), operation});
      ASSERT(res.second); // Redefining an existing function should not
                          // happen ATM
      return;
    }

    if (func::ReturnOp operation = llvm::dyn_cast<func::ReturnOp>(op)) {
      auto ops = operation.getOperands();
      ASSERT(ops.size() == 1); // only handle single return value atm
      // llvm::outs() << "returning: ";
      // llvm::outs() << ops[0] << "\n";
      // the ops[0] is something that we can hash_value to grab the result from
      // maps
      auto retval = frames.back().memrefs.find(mlir::hash_value(ops[0]));
      ASSERT(retval != frames.back().memrefs.end());
      llvm::outs() << "returning: ";
      retval->second.print(llvm::outs(), assignmnt);
      return;
    }

    if (KrnlEntryPointOp operation = llvm::dyn_cast<KrnlEntryPointOp>(op)) {
      int32_t numInputs = -1;
      int32_t numOutputs = -1;
      std::string func = "";

      for (auto a : operation->getAttrs()) {
        if (a.getName() == operation.getEntryPointFuncAttrName()) {
          func = a.getValue().cast<SymbolRefAttr>().getLeafReference().str();
        } else if (a.getName() == operation.getNumInputsAttrName()) {
          numInputs = a.getValue().cast<IntegerAttr>().getInt();
        } else if (a.getName() == operation.getNumOutputsAttrName()) {
          numOutputs = a.getValue().cast<IntegerAttr>().getInt();
        } else if (a.getName() == operation.getSignatureAttrName()) {
          // do nothing for signature atm
          // TODO: check against input types & shapes
        } else {
          UNREACHABLE("unhandled attribute: " + a.getName().str());
        }
      }
      // this operation defines the entry point of the program
      // grab the entry point function from the functions map
      auto funcOp = functions.find(func);
      ASSERT(funcOp != functions.end());

      // only can handle single outputs atm
      ASSERT(numOutputs == 1);

      // prepare the arguments for the function
      frames.push_back(nil::blueprint::stack_frame<VarType>());

      nil::blueprint::InputReader<
          BlueprintFieldType, VarType,
          nil::blueprint::assignment<ArithmetizationType>>
          input_reader(frames.back(), assignmnt);
      bool ok = input_reader.fill_public_input(funcOp->second, public_input);
      if (!ok) {
        std::cerr << "Public input does not match the circuit signature";
        const std::string &error = input_reader.get_error();
        if (!error.empty()) {
          std::cout << ": " << error;
        }
        std::cout << std::endl;
        exit(-1);
      }
      public_input_idx = input_reader.get_idx();

      // Initialize undef and zero vars once
      undef_var =
          put_into_assignment(typename BlueprintFieldType::value_type());
      zero_var =
          put_into_assignment(typename BlueprintFieldType::value_type(0));

      // go execute the function
      handleRegion(funcOp->second.getRegion());

      // TODO: what to do when done...
      // maybe print output?
      return;
    }

    std::string opName = op->getName().getIdentifier().str();
    UNREACHABLE(std::string("unhandled operation: ") + opName);
  }

  void handleRegion(Region &region) {
    for (Block &block : region.getBlocks())
      handleBlock(block);
  }

  void handleBlock(Block &block) {
    for (Operation &op : block.getOperations())
      handleOperation(&op);
  }

private:
  template <typename InputType> VarType put_into_assignment(InputType input) {
    assignmnt.public_input(0, public_input_idx) = input;
    return VarType(0, public_input_idx++, false,
                   VarType::column_type::public_input);
  }

  std::vector<nil::blueprint::stack_frame<VarType>> frames;
  std::map<std::string, func::FuncOp> functions;
  nil::blueprint::circuit<ArithmetizationType> &bp;
  nil::blueprint::assignment<ArithmetizationType> &assignmnt;
  const boost::json::array &public_input;
  size_t public_input_idx = 0;
  VarType undef_var;
  VarType zero_var;
};

template <typename BlueprintFieldType, typename ArithmetizationParams>
std::unique_ptr<Pass> createAssignMLIRPass(
    nil::blueprint::circuit<nil::crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &circuit,
    nil::blueprint::assignment<nil::crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &assignment,
    const boost::json::array &public_input) {
  return std::make_unique<
      AssignMLIRPass<BlueprintFieldType, ArithmetizationParams>>(
      circuit, assignment, public_input);
}
} // namespace zk_ml_toolchain

#endif // ZK_ML_TOOLCHAIN_ASSIGN_MLIR_PASS
