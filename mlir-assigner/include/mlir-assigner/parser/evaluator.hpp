#ifndef CRYPTO3_BLUEPRINT_COMPONENT_INSTRUCTION_MLIR_EVALUATOR_HPP

#include "nil/blueprint/blueprint/plonk/assignment.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
// #define TEST_WITHOUT_LOOKUP_TABLES

#include "mlir-assigner/helper/asserts.hpp"
#include "mlir-assigner/helper/logger.hpp"
#include "mlir/Dialect/zkml/ZkMlDialect.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Dialect/zkml/ZkMlOps.h"

#include <cstddef>
#include <cstdlib>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinDialect.h>

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Support/KrnlSupport.hpp"

#include <nil/blueprint/blueprint/plonk/assignment_proxy.hpp>
#include <nil/blueprint/blueprint/plonk/circuit_proxy.hpp>

#include <nil/blueprint/components/algebra/fixedpoint/type.hpp>

#include <mlir-assigner/components/comparison/fixed_comparison.hpp>
#include <mlir-assigner/components/comparison/argminmax.hpp>
#include <mlir-assigner/components/comparison/select.hpp>
#include <mlir-assigner/components/fixedpoint/abs.hpp>
#include <mlir-assigner/components/fixedpoint/ceil.hpp>
#include <mlir-assigner/components/fixedpoint/division.hpp>
#include <mlir-assigner/components/fixedpoint/exp.hpp>
#include <mlir-assigner/components/fixedpoint/sqrt.hpp>
#include <mlir-assigner/components/fixedpoint/log.hpp>
#include <mlir-assigner/components/fixedpoint/floor.hpp>
#include <mlir-assigner/components/fixedpoint/mul_rescale.hpp>
#include <mlir-assigner/components/fixedpoint/neg.hpp>
#include <mlir-assigner/components/fixedpoint/remainder.hpp>
#include <mlir-assigner/components/fixedpoint/dot_product.hpp>
#include <mlir-assigner/components/fixedpoint/cmp_set.hpp>
#include <mlir-assigner/components/fixedpoint/gather.hpp>
#include <mlir-assigner/components/fixedpoint/erf.hpp>
#include <mlir-assigner/components/fixedpoint/trigonometric.hpp>
#include <mlir-assigner/components/fixedpoint/conversion.hpp>
#include <mlir-assigner/components/boolean/logic_ops.hpp>
#include <mlir-assigner/components/fields/basic_arith.hpp>
#include <mlir-assigner/components/integer/mul_div.hpp>

#include <mlir-assigner/memory/memref.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>
#include <mlir-assigner/parser/input_reader.hpp>
#include <mlir-assigner/parser/output_writer.hpp>

#include <unordered_map>
#include <map>
#include <unistd.h>
using namespace mlir;

namespace zk_ml_toolchain {

    namespace detail {

        int64_t evalAffineExpr(AffineExpr expr, llvm::ArrayRef<int64_t> dims, llvm::ArrayRef<int64_t> symbols) {
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

        bool evalIntegerSet(IntegerSet set, llvm::ArrayRef<int64_t> dims, llvm::ArrayRef<int64_t> symbols,
                            nil::blueprint::logger &logger) {
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
                assert(!set.isEq(i) && "equality in evalIntegerSet??");
                if (constraint < 0) {
                    return false;
                }
            }
            return true;
        }
        bool evalIntegerSet(IntegerSet set, llvm::ArrayRef<int64_t> operands, nil::blueprint::logger &logger) {
            return evalIntegerSet(set, operands.take_front(set.getNumDims()), operands.drop_front(set.getNumDims()),
                                  logger);
        }
        SmallVector<int64_t> evalAffineMap(AffineMap map, llvm::ArrayRef<int64_t> dims,
                                           llvm::ArrayRef<int64_t> symbols) {
            SmallVector<int64_t> result;
            for (auto expr : map.getResults()) {
                result.push_back(evalAffineExpr(expr, dims, symbols));
            }
            return result;
        }

        llvm::SmallVector<int64_t> evalAffineMap(AffineMap map, ArrayRef<int64_t> operands) {
            return evalAffineMap(map, operands.take_front(map.getNumDims()), operands.drop_front(map.getNumDims()));
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

        template<class T>
        T castFromAttr(Attribute attr) {
            T result = llvm::dyn_cast<T>(attr);
            assert(result);
            return result;
        }

    }    // namespace detail

    using namespace detail;

    template<typename BlueprintFieldType, typename ArithmetizationParams, std::uint8_t PreLimbs, std::uint8_t PostLimbs>
    class evaluator {
    public:
        using ArithmetizationType =
            nil::crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>;
        using VarType = nil::crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>;
        using FixedPoint = nil::blueprint::components::FixedPoint<BlueprintFieldType, PreLimbs, PostLimbs>;
        using ComponentParameters = nil::blueprint::common_component_parameters<VarType>;
        using generation_mode = nil::blueprint::generation_mode;

        enum class ClipStrategy { PANIC, CLIP, ZERO };

        evaluator(nil::blueprint::circuit_proxy<ArithmetizationType> &circuit,
                  nil::blueprint::assignment_proxy<ArithmetizationType> &assignment,
                  const boost::json::array &public_input, const boost::json::array &private_input,
                  boost::json::array &public_output, generation_mode gen_mode,
                  nil::blueprint::print_format print_circuit_format, std::string &clip,
                  nil::blueprint::logger &logger) :
            gen_mode(gen_mode),
            print_circuit_format(print_circuit_format), logger(logger), bp(circuit), assignmnt(assignment),
            public_input(public_input), private_input(private_input), public_output(public_output) {
            lower_bound = FixedPoint(1, FixedPoint::SCALE).to_double();
            if ("clip" == clip) {
                clip_strategy = ClipStrategy::CLIP;
            } else if ("zero" == clip) {
                clip_strategy = ClipStrategy::ZERO;
            } else if ("panic" == clip) {
                clip_strategy = ClipStrategy::PANIC;
            } else {
                UNREACHABLE("unsupported clipping strategy: " + clip);
            }
        }

        evaluator(const evaluator &pass) = delete;
        evaluator(evaluator &&pass) = delete;
        evaluator &operator=(const evaluator &pass) = delete;

        void handleKrnlEntryOperation(KrnlEntryPointOp &EntryPoint, std::string &func) {
            for (auto a : EntryPoint->getAttrs()) {
                if (a.getName() == EntryPoint.getEntryPointFuncAttrName()) {
                    func = a.getValue().cast<SymbolRefAttr>().getLeafReference().str();
                }
            }
        }

        std::string gatherFuncDecls(Region &BodyRegion) {
            assert(BodyRegion.getBlocks().size() == 1 && "must have single block in main region");
            std::string MainDecl = "";
            Block &MainBlock = BodyRegion.getBlocks().front();
            bool EntryFound = false;
            for (auto &Op : MainBlock.getOperations()) {
                if (func::FuncOp FuncOp = llvm::dyn_cast<func::FuncOp>(Op)) {
                    auto res = functions.insert({FuncOp.getSymName().str(), FuncOp});
                    assert(res.second);    // Redefining an existing function should not
                                           // happen ATM
                } else if (auto EntryPointOp = llvm::dyn_cast<KrnlEntryPointOp>(Op)) {
                    assert(!EntryFound && "multiple entries found");
                    EntryFound = true;
                    handleKrnlEntryOperation(EntryPointOp, MainDecl);
                } else {
                    std::string opName = Op.getName().getIdentifier().str();
                    UNREACHABLE(std::string("only func.func and krnl.entryPoint allowed but got: ") + opName);
                }
            }
            assert(EntryFound && "no entry point found");
            return MainDecl;
        }

        void evaluate(mlir::OwningOpRef<mlir::ModuleOp> module) {
            std::string MainDecl = gatherFuncDecls(module->getBodyRegion());
            auto funcOp = functions.find(MainDecl);
            assert(funcOp != functions.end());
            // prepare everything for run

            // Setup a stack frame to use
            stack.push_frame();
            nil::blueprint::stack_frame<VarType> &main_frame = stack.get_last_frame();
            {
                nil::blueprint::InputReader<BlueprintFieldType, VarType,
                                            nil::blueprint::assignment<ArithmetizationType>, PreLimbs, PostLimbs>
                    input_reader(main_frame, assignmnt, output_memrefs);
                bool ok = input_reader.fill_input(funcOp->second, public_input, private_input,
                                                  uint8_t(gen_mode & generation_mode::ASSIGNMENTS));
                if (!ok) {
                    std::cerr << "Provided input files do not match the circuit signature";
                    const std::string &error = input_reader.get_error();
                    if (!error.empty()) {
                        std::cerr << ": " << error;
                    }
                    std::cerr << std::endl;
                    exit(-1);
                }

                if (uint8_t(gen_mode & generation_mode::ASSIGNMENTS) == 0 && !public_output.empty()) {
                    std::cerr << "Public output provided, but no assignments requested" << std::endl;
                    exit(-1);
                }
                // reserve space for the output constraints
                ok = input_reader.reserve_outputs(funcOp->second, public_output, output_is_already_assigned);
                if (!ok) {
                    std::cerr << "Could not reserve space for output constraints in public input column";
                    const std::string &error = input_reader.get_error();
                    if (!error.empty()) {
                        std::cerr << ": " << error;
                    }
                    std::cerr << std::endl;
                    exit(-1);
                }
            }
            // Initialize undef and zero vars once
            undef_var = put_into_assignment(typename BlueprintFieldType::value_type());
            zero_var = put_into_assignment(typename BlueprintFieldType::value_type(0));
            true_var = std::nullopt;
            // go execute the function
            handleRegion(funcOp->second.getRegion());
        }

    private:
        void doAffineFor(affine::AffineForOp &op, int64_t from, int64_t to, int64_t step) {
            assert(from < to);
            assert(step);
            // atm handle only simple loops with one region,block and argument
            assert(op.getRegion().hasOneBlock());
            assert(op.getRegion().getArguments().size() == 1);
            logger.trace("for (%d -> %d step %d)", from, to, step);
            llvm::hash_code counterHash = mlir::hash_value(op.getInductionVar());
            logger.trace("inserting hash: %x:%d", std::size_t(counterHash), from);
            // auto res = frames.back().constant_values.insert({counterHash, from});
            // we do not want overrides here, since we delete it
            // after loop this should never happen
            stack.push_constant(op.getInductionVar(), from, false);
            while (from < to) {
                handleRegion(op.getLoopBody());
                from += step;
                logger.trace("updating hash: %x:%d", std::size_t(counterHash), from);
                stack.push_constant(counterHash, from);
                logger.trace("%d -> %d", from, to);
                logger.trace("for done! go next iteration..");
            }
            stack.erase_constant(counterHash);
            logger.trace("deleting: %x", std::size_t(counterHash));
        }

        int64_t evaluateForParameter(AffineMap &affineMap, llvm::SmallVector<Value> &operands, bool from) {
            if (affineMap.isConstant()) {
                return affineMap.getResult(0).cast<AffineConstantExpr>().getValue();
            } else {
                assert(affineMap.getNumInputs() == operands.size());
                llvm::SmallVector<int64_t> inVector(affineMap.getNumInputs());
                for (unsigned i = 0; i < affineMap.getNumInputs(); ++i) {
                    inVector[i] = stack.get_constant(operands[i]);
                }
                llvm::SmallVector<int64_t> eval = evalAffineMap(affineMap, inVector);
                return from ? getMaxFromVector(eval) : getMinFromVector(eval);
            }
        }

        double toFixpoint(VarType toConvert) {
            auto val = var_value(assignmnt, toConvert).data;
            FixedPoint out(val, FixedPoint::SCALE);
            return out.to_double();
            // auto Lhs = stack.get_local(operation.getLhs());
            // auto Rhs = stack.get_local(operation.getRhs());
            // auto Result = stack.get_local(operation.getResult());
            // std::cout << toFixpoint(Lhs) << " + " << toFixpoint(Rhs) << " = " << toFixpoint(Result) << "\n";
        }

#define BITSWITCHER(func, b)                                                                           \
    switch (b) {                                                                                       \
        case 8:                                                                                        \
            func<1>(operation, stack, bp, assignmnt, compParams);                                      \
            break;                                                                                     \
        case 16:                                                                                       \
            func<2>(operation, stack, bp, assignmnt, compParams);                                      \
            break;                                                                                     \
        case 32:                                                                                       \
            func<4>(operation, stack, bp, assignmnt, compParams);                                      \
            break;                                                                                     \
        case 64:                                                                                       \
            func<8>(operation, stack, bp, assignmnt, compParams);                                      \
            break;                                                                                     \
        default:                                                                                       \
            UNREACHABLE(std::string("unsupported int bit size for bitwise op: ") + std::to_string(b)); \
    }

        void handleArithOperation(Operation *op, const ComponentParameters &compParams) {
            if (arith::AddFOp operation = llvm::dyn_cast<arith::AddFOp>(op)) {
                nil::blueprint::handle_add(operation, stack, bp, assignmnt, compParams);
            } else if (arith::SubFOp operation = llvm::dyn_cast<arith::SubFOp>(op)) {
                nil::blueprint::handle_sub(operation, stack, bp, assignmnt, compParams);
            } else if (arith::MulFOp operation = llvm::dyn_cast<arith::MulFOp>(op)) {
                nil::blueprint::handle_fmul<PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (arith::DivFOp operation = llvm::dyn_cast<arith::DivFOp>(op)) {
                nil::blueprint::handle_fdiv<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (arith::RemFOp operation = llvm::dyn_cast<arith::RemFOp>(op)) {
                nil::blueprint::handle_frem<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (arith::CmpFOp operation = llvm::dyn_cast<arith::CmpFOp>(op)) {
                nil::blueprint::handle_fcmp<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (arith::SelectOp operation = llvm::dyn_cast<arith::SelectOp>(op)) {
                ASSERT(operation.getNumOperands() == 3 && "Select must have three operands");
                ASSERT(operation->getOperand(1).getType() == operation->getOperand(2).getType() &&
                       "Select must operate on same type");
                // check if we work on indices
                Type operandType = operation->getOperand(1).getType();
                auto i1Hash = mlir::hash_value(operation->getOperand(0));
                if (operandType.isa<IndexType>()) {
                    if (stack.get_constant(i1Hash)) {
                        auto truthy = stack.get_constant(operation->getOperand(1));
                        stack.push_constant(operation->getResult(0), truthy);
                    } else {
                        auto falsy = stack.get_constant(operation->getOperand(2));
                        stack.push_constant(operation->getResult(0), falsy);
                    }
                } else if (operandType.isa<FloatType>() || operandType.isa<IntegerType>()) {
                    if (stack.peek_constant(i1Hash)) {
                        // we can just pick as selector was produces by index
                        if (stack.get_constant(i1Hash)) {
                            auto truthy = stack.get_local(operation->getOperand(1));
                            stack.push_local(operation->getResult(0), truthy);
                        } else {
                            auto falsy = stack.get_local(operation->getOperand(2));
                            stack.push_local(operation->getResult(0), falsy);
                        }
                    } else {
                        nil::blueprint::handle_select(operation, stack, bp, assignmnt, compParams);
                    }
                } else {
                    std::string typeStr;
                    llvm::raw_string_ostream ss(typeStr);
                    ss << operandType;
                    UNREACHABLE(std::string("unhandled select operand: ") + typeStr);
                }
            } else if (arith::NegFOp operation = llvm::dyn_cast<arith::NegFOp>(op)) {
                nil::blueprint::handle_neg(operation, stack, bp, assignmnt, compParams);
            } else if (arith::AndIOp operation = llvm::dyn_cast<arith::AndIOp>(op)) {
                // check if logical and or bitwise and
                mlir::Type LhsType = operation.getLhs().getType();
                mlir::Type RhsType = operation.getRhs().getType();
                assert(LhsType == RhsType && "must be same type for AndIOp");
                uint8_t bits = LhsType.getIntOrFloatBitWidth();
                if (1 == bits) {
                    nil::blueprint::handle_logic_and(operation, stack, bp, assignmnt, compParams);
                } else {
                    BITSWITCHER(nil::blueprint::handle_bitwise_and, bits);
                }
            } else if (arith::OrIOp operation = llvm::dyn_cast<arith::OrIOp>(op)) {
                ASSERT(operation.getNumOperands() == 2 && "Or must have two operands");
                ASSERT(operation->getOperand(0).getType() == operation->getOperand(1).getType() &&
                       "Or must operate on same type");
                // check if we work on indices
                // TODO this seems like a hack, maybe we can do something better
                auto lhsHash = mlir::hash_value(operation.getLhs());
                if (stack.peek_constant(lhsHash)) {
                    auto lhs = stack.get_constant(lhsHash);
                    auto rhs = stack.get_constant(operation.getRhs());
                    auto result = lhs | rhs;
                    stack.push_constant(operation.getResult(), result);
                } else {
                    // check if logical and or bitwise and
                    mlir::Type LhsType = operation.getLhs().getType();
                    mlir::Type RhsType = operation.getRhs().getType();
                    assert(LhsType == RhsType && "must be same type for OrIOp");
                    unsigned bits = LhsType.getIntOrFloatBitWidth();
                    if (1 == bits) {
                        nil::blueprint::handle_logic_or(operation, stack, bp, assignmnt, compParams);
                    } else {
                        BITSWITCHER(nil::blueprint::handle_bitwise_or, bits);
                    }
                }
            } else if (arith::ExtFOp operation = llvm::dyn_cast<arith::ExtFOp>(op)) {
                // nothing for us to do here. just copy in stack
                auto opHash = mlir::hash_value(operation->getOperand(0));
                if (stack.peek_local(opHash)) {
                    stack.push_local(operation.getResult(), stack.get_local(opHash));
                } else if (stack.peek_memref(opHash)) {
                    stack.push_memref(operation.getResult(), stack.get_memref(opHash));
                } else {
                    UNREACHABLE("cannot find operand for extf");
                }
            } else if (arith::XOrIOp operation = llvm::dyn_cast<arith::XOrIOp>(op)) {
                // check if logical and or bitwise and
                mlir::Type LhsType = operation.getLhs().getType();
                mlir::Type RhsType = operation.getRhs().getType();
                assert(LhsType == RhsType && "must be same type for XOrIOp");
                unsigned bits = LhsType.getIntOrFloatBitWidth();
                if (1 == bits) {
                    nil::blueprint::handle_logic_xor(operation, stack, bp, assignmnt, compParams);
                } else {
                    BITSWITCHER(nil::blueprint::handle_bitwise_xor, bits);
                }
            } else if (arith::AddIOp operation = llvm::dyn_cast<arith::AddIOp>(op)) {
                if (operation.getLhs().getType().isa<IndexType>()) {
                    assert(operation.getRhs().getType().isa<IndexType>());
                    auto lhs = stack.get_constant(operation.getLhs());
                    auto rhs = stack.get_constant(operation.getRhs());
                    stack.push_constant(operation.getResult(), lhs + rhs);
                } else {
                    nil::blueprint::handle_add(operation, stack, bp, assignmnt, compParams);
                }
            } else if (arith::SubIOp operation = llvm::dyn_cast<arith::SubIOp>(op)) {
                if (operation.getLhs().getType().isa<IndexType>()) {
                    assert(operation.getRhs().getType().isa<IndexType>());
                    auto lhs = stack.get_constant(operation.getLhs());
                    auto rhs = stack.get_constant(operation.getRhs());
                    stack.push_constant(operation.getResult(), lhs - rhs);
                } else {
                    nil::blueprint::handle_sub(operation, stack, bp, assignmnt, compParams);
                }
            } else if (arith::MulIOp operation = llvm::dyn_cast<arith::MulIOp>(op)) {
                if (operation.getLhs().getType().isa<IndexType>()) {
                    assert(operation.getRhs().getType().isa<IndexType>());
                    auto lhs = stack.get_constant(operation.getLhs());
                    auto rhs = stack.get_constant(operation.getRhs());
                    stack.push_constant(operation.getResult(), lhs * rhs);
                } else {
                    nil::blueprint::handle_integer_mul(operation, stack, bp, assignmnt, compParams);
                }
            } else if (arith::CmpIOp operation = llvm::dyn_cast<arith::CmpIOp>(op)) {
                if (operation.getLhs().getType().isa<IndexType>()) {
                    assert(operation.getRhs().getType().isa<IndexType>());

                    auto lhs = stack.get_constant(operation.getLhs());
                    auto rhs = stack.get_constant(operation.getRhs());
                    int64_t cmpResult;
                    switch (operation.getPredicate()) {
                        case arith::CmpIPredicate::eq:
                            cmpResult = static_cast<int64_t>(lhs == rhs);
                            break;
                        case arith::CmpIPredicate::ne:
                            cmpResult = static_cast<int64_t>(lhs != rhs);
                            break;
                        case arith::CmpIPredicate::slt:
                            cmpResult = static_cast<int64_t>(lhs < rhs);
                            break;
                        case arith::CmpIPredicate::sle:
                            cmpResult = static_cast<int64_t>(lhs <= rhs);
                            break;
                        case arith::CmpIPredicate::sgt:
                            cmpResult = static_cast<int64_t>(lhs > rhs);
                            break;
                        case arith::CmpIPredicate::sge:
                            cmpResult = static_cast<int64_t>(lhs >= rhs);
                            break;
                        case arith::CmpIPredicate::ult:
                            cmpResult = static_cast<int64_t>(static_cast<uint64_t>(lhs) < static_cast<uint64_t>(rhs));
                            break;
                        case arith::CmpIPredicate::ule:
                            cmpResult = static_cast<int64_t>(static_cast<uint64_t>(lhs) <= static_cast<uint64_t>(rhs));
                            break;
                        case arith::CmpIPredicate::ugt:
                            cmpResult = static_cast<int64_t>(static_cast<uint64_t>(lhs) > static_cast<uint64_t>(rhs));
                            break;
                        case arith::CmpIPredicate::uge:
                            cmpResult = static_cast<int64_t>(static_cast<uint64_t>(lhs) >= static_cast<uint64_t>(rhs));
                            break;
                    }
                    stack.push_constant(operation.getResult(), cmpResult);
                } else {
                    handle_icmp(operation, stack, bp, assignmnt, compParams);
                }
            } else if (arith::ConstantOp operation = llvm::dyn_cast<arith::ConstantOp>(op)) {
                TypedAttr constantValue = operation.getValueAttr();
                if (operation->getResult(0).getType().isa<IndexType>()) {
                    stack.push_constant(operation.getResult(), llvm::dyn_cast<IntegerAttr>(constantValue).getInt());
                } else if (constantValue.isa<IntegerAttr>()) {
                    // this insert is ok, since this should never change, so we don't
                    // override it if it is already there
                    int64_t value;
                    if (constantValue.isa<BoolAttr>()) {
                        value = llvm::dyn_cast<BoolAttr>(constantValue).getValue() ? 1 : 0;
                    } else {
                        value = llvm::dyn_cast<IntegerAttr>(constantValue).getInt();
                    }
                    stack.push_constant(operation.getResult(), value);
                    typename BlueprintFieldType::value_type field_constant = value;
                    auto val = put_into_assignment(field_constant);
                    stack.push_local(operation.getResult(), val);
                } else if (constantValue.isa<FloatAttr>()) {
                    double d = llvm::dyn_cast<FloatAttr>(constantValue).getValueAsDouble();
                    VarType var = put_float_into_assignment(d);
                    stack.push_local(operation.getResult(), var);
                } else {
                    logger << constantValue;
                    UNREACHABLE("unhandled constant");
                }
            } else if (arith::IndexCastOp operation = llvm::dyn_cast<arith::IndexCastOp>(op)) {
                assert(operation->getNumOperands() == 1 && "IndexCast must have exactly one operand");
                auto opHash = mlir::hash_value(operation->getOperand(0));
                Type casteeType = operation->getOperand(0).getType();
                if (casteeType.isa<IntegerType>()) {
                    UNREACHABLE("Illegal cast. Cannot cast from int to index (dynamic shaping of circuit)");
                } else if (casteeType.isa<IndexType>()) {
                    auto index = stack.get_constant(opHash);
                    typename BlueprintFieldType::value_type field_constant = index;
                    auto val = put_into_assignment(field_constant);
                    stack.push_local(operation.getResult(), val);
                } else {
                    UNREACHABLE("unsupported Index Cast");
                }
            } else if (arith::SIToFPOp operation = llvm::dyn_cast<arith::SIToFPOp>(op)) {
                nil::blueprint::handle_to_fixedpoint<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (arith::UIToFPOp operation = llvm::dyn_cast<arith::UIToFPOp>(op)) {
                nil::blueprint::handle_to_fixedpoint<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (arith::FPToSIOp operation = llvm::dyn_cast<arith::FPToSIOp>(op)) {
                nil::blueprint::handle_to_int<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (arith::FPToUIOp operation = llvm::dyn_cast<arith::FPToUIOp>(op)) {
                nil::blueprint::handle_to_int<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (llvm::isa<arith::ExtUIOp>(op) || llvm::isa<arith::ExtSIOp>(op) ||
                       llvm::isa<arith::TruncIOp>(op)) {
                VarType &toExtend = stack.get_local(op->getOperand(0));
                stack.push_local(op->getResult(0), toExtend);
            } else {
                std::string opName = op->getName().getIdentifier().str();
                UNREACHABLE(std::string("unhandled arith operation: ") + opName);
            }
        }
#undef BITSWITCHER

        void handleMathOperation(Operation *op, const ComponentParameters &compParams) {
            if (math::ExpOp operation = llvm::dyn_cast<math::ExpOp>(op)) {
                nil::blueprint::handle_exp<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (math::LogOp operation = llvm::dyn_cast<math::LogOp>(op)) {
                nil::blueprint::handle_log<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (math::PowFOp operation = llvm::dyn_cast<math::PowFOp>(op)) {
                UNREACHABLE("powf not supported. Did you compile the model with standard MLIR?");
            } else if (math::IPowIOp operation = llvm::dyn_cast<math::IPowIOp>(op)) {
                // we only allow that on index for now
                assert(operation.getLhs().getType().isa<IndexType>() && "powi only allowed for indices");
                assert(operation.getRhs().getType().isa<IndexType>() && "powi only allowed for indices");
                int64_t base = stack.get_constant(operation.getLhs());
                int64_t pow = stack.get_constant(operation.getRhs());
                assert(base == 2 && "For now we only support powi to power of 2");
                stack.push_constant(operation.getResult(), 1 << pow);
            } else if (math::AbsFOp operation = llvm::dyn_cast<math::AbsFOp>(op)) {
                nil::blueprint::handle_abs<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (math::CeilOp operation = llvm::dyn_cast<math::CeilOp>(op)) {
                nil::blueprint::handle_ceil<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (math::FloorOp operation = llvm::dyn_cast<math::FloorOp>(op)) {
                nil::blueprint::handle_floor<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (math::CopySignOp operation = llvm::dyn_cast<math::CopySignOp>(op)) {
                // do nothing for now since it only comes up during mod, and there
                // the component handles this correctly; do we need this later on?
                VarType &src = stack.get_local(operation.getLhs());
                stack.push_local(operation.getResult(), src);
            } else if (math::SqrtOp operation = llvm::dyn_cast<math::SqrtOp>(op)) {
                nil::blueprint::handle_sqrt<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (math::SinOp operation = llvm::dyn_cast<math::SinOp>(op)) {
                nil::blueprint::handle_sin<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (math::CosOp operation = llvm::dyn_cast<math::CosOp>(op)) {
                nil::blueprint::handle_cos<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (math::TanhOp operation = llvm::dyn_cast<math::TanhOp>(op)) {
                nil::blueprint::handle_tanh<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (math::ErfOp operation = llvm::dyn_cast<math::ErfOp>(op)) {
                nil::blueprint::handle_erf<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else {
                std::string opName = op->getName().getIdentifier().str();
                UNREACHABLE(std::string("unhandled math operation: ") + opName);
            }
        }

        void handleAffineOperation(Operation *op) {
            // Print the operation itself and some of its properties
            // Print the operation attributes
            std::string opName = op->getName().getIdentifier().str();
            logger.debug("visiting %s", opName);
            if (affine::AffineForOp operation = llvm::dyn_cast<affine::AffineForOp>(op)) {
                logger.debug("visiting affine for!");
                assert(op->getAttrs().size() == 3);
                AffineMap fromMap = operation.getLowerBoundMap();
                int64_t step = operation.getStep();
                AffineMap toMap = operation.getUpperBoundMap();
                assert(fromMap.getNumInputs() + toMap.getNumInputs() == op->getNumOperands());

                auto operandsFrom = operation.getLowerBoundOperands();
                auto operandsTo = operation.getUpperBoundOperands();
                auto operandsFromV = llvm::SmallVector<Value>(operandsFrom.begin(), operandsFrom.end());
                auto operandsToV = llvm::SmallVector<Value>(operandsTo.begin(), operandsTo.end());
                int64_t from = evaluateForParameter(fromMap, operandsFromV, true);
                int64_t to = evaluateForParameter(toMap, operandsToV, false);
                doAffineFor(operation, from, to, step);
            } else if (affine::AffineLoadOp operation = llvm::dyn_cast<affine::AffineLoadOp>(op)) {
                nil::blueprint::memref<VarType> &memref = stack.get_memref(operation.getMemref());
                // grab the indices and build index vector
                auto indices = operation.getIndices();
                std::vector<int64_t> mapDims;
                mapDims.reserve(indices.size());
                for (auto idx : indices) {
                    // look for indices in constant_values
                    mapDims.push_back(stack.get_constant(idx));
                }
                auto affineMap =
                    castFromAttr<AffineMapAttr>(operation->getAttr(affine::AffineLoadOp::getMapAttrStrName()))
                        .getAffineMap();
                auto value = memref.get(evalAffineMap(affineMap, mapDims));
                stack.push_local(operation.getResult(), value);
            } else if (affine::AffineStoreOp operation = llvm::dyn_cast<affine::AffineStoreOp>(op)) {
                // affine.store
                nil::blueprint::memref<VarType> &memref = stack.get_memref(operation.getMemref());
                // grab the indices and build index vector
                auto indices = operation.getIndices();
                std::vector<int64_t> mapDims;
                mapDims.reserve(indices.size());
                for (auto idx : indices) {
                    mapDims.push_back(stack.get_constant(idx));
                }
                auto affineMap =
                    castFromAttr<AffineMapAttr>(operation->getAttr(affine::AffineStoreOp::getMapAttrStrName()))
                        .getAffineMap();
                VarType &value = stack.get_local(operation.getValue());
                memref.put(evalAffineMap(affineMap, mapDims), value);

            } else if (affine::AffineYieldOp operation = llvm::dyn_cast<affine::AffineYieldOp>(op)) {
                // Affine Yields are Noops for us
            } else if (opName == "affine.apply" || opName == "affine.min") {
                logger.debug("got affine.apply");
                assert(op->getResults().size() == 1);
                assert(op->getAttrs().size() == 1);
                AffineMap applyMap = castFromAttr<AffineMapAttr>(op->getAttrs()[0].getValue()).getAffineMap();
                llvm::SmallVector<Value> operands(op->getOperands().begin(), op->getOperands().end());
                int64_t result = evaluateForParameter(applyMap, operands, false);
                stack.push_constant(op->getResults()[0], result);
            } else if (opName == "affine.max") {
                logger.debug("got affine.max");
                assert(op->getResults().size() == 1);
                assert(op->getAttrs().size() == 1);
                AffineMap applyMap = castFromAttr<AffineMapAttr>(op->getAttrs()[0].getValue()).getAffineMap();
                llvm::SmallVector<Value> operands(op->getOperands().begin(), op->getOperands().end());
                int64_t result = evaluateForParameter(applyMap, operands, true);
                stack.push_constant(mlir::hash_value(op->getResults()[0]), result);
            } else {
                UNREACHABLE(std::string("unhandled affine operation: ") + opName);
            }
        }

        void handleKrnlOperation(Operation *op, const ComponentParameters &compParams) {
            // Print the operation itself and some of its properties
            // Print the operation attributes
            std::string opName = op->getName().getIdentifier().str();
            logger.debug("visiting %s", opName);
            if (KrnlGlobalOp operation = llvm::dyn_cast<KrnlGlobalOp>(op)) {
                logger.debug("global op");
                logger << operation;
                logger << operation.getOutput();
                logger << operation.getShape();
                logger << operation.getName();
                logger << operation.getValue();

                // The element type of the array.
                const mlir::Type type = operation.getOutput().getType();
                const mlir::MemRefType memRefTy = type.cast<mlir::MemRefType>();
                const mlir::Type constantElementType = memRefTy.getElementType();
                const auto shape = memRefTy.getShape();
                logger << memRefTy;

                // build a memref and fill it with value
                nil::blueprint::memref<VarType> m(shape, constantElementType);

                // Create the global at the entry of the module.
                assert(operation.getValue().has_value() && "Krnl Global must always have a value");
                auto value = operation.getValue().value();
                if (DenseElementsAttr attr = llvm::dyn_cast<DenseElementsAttr>(value)) {
                    mlir::Type attrType = attr.getElementType();
                    if (attrType.isa<mlir::IntegerType>()) {
                        mlir::IntegerType intType = attrType.cast<mlir::IntegerType>();
                        if (1 == intType.getIntOrFloatBitWidth()) {
                            // check if we already have a true var created
                            if (!true_var.has_value()) {
                                true_var =
                                    std::make_optional(put_into_assignment(typename BlueprintFieldType::value_type(1)));
                            }
                            auto bools = attr.tryGetValues<bool>();
                            assert(!mlir::failed(bools) && "must work as we checked above");
                            size_t idx = 0;
                            for (auto a : bools.value()) {
                                m.put_flat(idx++, a ? true_var.value() : zero_var);
                            }
                        } else {
                            auto ints = attr.tryGetValues<APInt>();
                            assert(!mlir::failed(ints) && "must work as we checked above");
                            size_t idx = 0;
                            for (auto a : ints.value()) {
                                auto var = put_into_assignment(a.getSExtValue());
                                m.put_flat(idx++, var);
                            }
                        }
                    } else if (attrType.isa<mlir::FloatType>()) {
                        auto floats = attr.tryGetValues<APFloat>();
                        assert(!mlir::failed(floats) && "must work as we checked above");
                        size_t idx = 0;
                        for (auto a : floats.value()) {
                            double d;
                            if (&a.getSemantics() == &llvm::APFloat::IEEEdouble()) {
                                d = a.convertToDouble();
                            } else if (&a.getSemantics() == &llvm::APFloat::IEEEsingle()) {
                                d = a.convertToFloat();
                            } else {
                                UNREACHABLE("unsupported float semantics");
                            }
                            VarType var = put_float_into_assignment(d);
                            m.put_flat(idx++, var);
                        }
                    } else {
                        UNREACHABLE("Unsupported attribute type");
                    }
                } else {
                    UNREACHABLE("Expected a DenseElementsAttr");
                }
                stack.push_memref(operation.getOutput(), m);
                return;
            } else if (KrnlMemcpyOp operation = llvm::dyn_cast<KrnlMemcpyOp>(op)) {
                // get dst and src memref
                nil::blueprint::memref<VarType> &DstMemref = stack.get_memref(operation.getDest());
                nil::blueprint::memref<VarType> &SrcMemref = stack.get_memref(operation.getSrc());
                // get num elements and offset
                auto NumElements = stack.get_constant(operation.getNumElems());
                auto DstOffset = stack.get_constant(operation.getDestOffset());
                auto SrcOffset = stack.get_constant(operation.getSrcOffset());
                DstMemref.copyFrom(SrcMemref, NumElements, DstOffset, SrcOffset);
            } else if (KrnlAsinOp operation = llvm::dyn_cast<KrnlAsinOp>(op)) {
                nil::blueprint::handle_asin<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (KrnlAsinhOp operation = llvm::dyn_cast<KrnlAsinhOp>(op)) {
                nil::blueprint::handle_asinh<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (KrnlAcosOp operation = llvm::dyn_cast<KrnlAcosOp>(op)) {
                nil::blueprint::handle_acos<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (KrnlAcoshOp operation = llvm::dyn_cast<KrnlAcoshOp>(op)) {
                nil::blueprint::handle_acosh<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (KrnlTanOp operation = llvm::dyn_cast<KrnlTanOp>(op)) {
                nil::blueprint::handle_tan<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (KrnlAtanOp operation = llvm::dyn_cast<KrnlAtanOp>(op)) {
                nil::blueprint::handle_atan<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (KrnlAtanhOp operation = llvm::dyn_cast<KrnlAtanhOp>(op)) {
                nil::blueprint::handle_atanh<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else {
                std::string opName = op->getName().getIdentifier().str();
                UNREACHABLE(std::string("unhandled krnl operation: ") + opName);
            }
        }

        void handleZkMlOperation(Operation *op, const ComponentParameters &compParams) {
            if (zkml::DotProductOp operation = llvm::dyn_cast<zkml::DotProductOp>(op)) {
                nil::blueprint::handle_dot_product<PreLimbs, PostLimbs>(operation, zero_var, stack, bp, assignmnt,
                                                                        compParams);
            } else if (zkml::ArgMinOp operation = llvm::dyn_cast<zkml::ArgMinOp>(op)) {
                auto nextIndexVar = put_into_assignment(stack.get_constant(operation.getNextIndex()));
                nil::blueprint::handle_argmin<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, nextIndexVar,
                                                                   compParams);
            } else if (zkml::ArgMaxOp operation = llvm::dyn_cast<zkml::ArgMaxOp>(op)) {
                auto nextIndexVar = put_into_assignment(stack.get_constant(operation.getNextIndex()));
                nil::blueprint::handle_argmax<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, nextIndexVar,
                                                                   compParams);
            } else if (zkml::GatherOp operation = llvm::dyn_cast<zkml::GatherOp>(op)) {
                auto dataIndex = stack.get_constant(operation.getDataIndex());
                auto dataIndexVar = put_into_assignment(dataIndex);
                nil::blueprint::handle_gather(operation, stack, bp, assignmnt, dataIndexVar, compParams);
            } else if (zkml::CmpSetOp operation = llvm::dyn_cast<zkml::CmpSetOp>(op)) {
                auto dataIndex = stack.get_constant(operation.getIndex());
                auto dataIndexVar = put_into_assignment(dataIndex);
                handle_cmp_set(operation, stack, bp, assignmnt, dataIndexVar, compParams);
            } else if (zkml::ExpNoClipOp operation = llvm::dyn_cast<zkml::ExpNoClipOp>(op)) {
                nil::blueprint::handle_exp_no_clip<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (zkml::SinhOp operation = llvm::dyn_cast<zkml::SinhOp>(op)) {
                nil::blueprint::handle_sinh<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (zkml::CoshOp operation = llvm::dyn_cast<zkml::CoshOp>(op)) {
                nil::blueprint::handle_cosh<PreLimbs, PostLimbs>(operation, stack, bp, assignmnt, compParams);
            } else if (zkml::OnnxAmountOp operation = llvm::dyn_cast<zkml::OnnxAmountOp>(op)) {
                amount_ops = operation.getAmount();
            } else if (zkml::TraceOp operation = llvm::dyn_cast<zkml::TraceOp>(op)) {
                std::cout << "> " << operation.getTrace().str() << " (" << (++progress) << "/" << amount_ops << ")\n";
                stack.print(std::cout);
            } else {
                std::string opName = op->getName().getIdentifier().str();
                UNREACHABLE(std::string("unhandled zkML operation: ") + opName);
            }
        }

        void handleMemRefOperation(Operation *op) {

            if (memref::AllocOp operation = llvm::dyn_cast<memref::AllocOp>(op)) {
                logger.debug("allocating memref");
                logger << operation;
                MemRefType type = operation.getType();
                // check for dynamic size
                std::vector<int64_t> dims;
                auto operands = operation.getOperands();
                unsigned dynamicCounter = 0;
                for (auto dim : type.getShape()) {
                    if (dim == mlir::ShapedType::kDynamic) {
                        assert(dynamicCounter < operands.size() && "not enough operands for dynamic memref");
                        dims.emplace_back(stack.get_constant(operands[dynamicCounter++]));
                    } else {
                        dims.emplace_back(dim);
                    }
                }
                auto m = nil::blueprint::memref<VarType>(dims, type.getElementType());
                stack.push_memref(operation.getMemref(), m, false);
            } else if (memref::AllocaOp operation = llvm::dyn_cast<memref::AllocaOp>(op)) {
                logger.debug("allocating (stack) memref");
                MemRefType type = operation.getType();
                logger << type.getElementType();
                logger << type.getShape();
                auto res = operation->getResult(0);
                auto res2 = operation.getMemref();
                logger << res;
                logger << res2;
                auto m = nil::blueprint::memref<VarType>(type.getShape(), type.getElementType());
                stack.push_memref(operation.getMemref(), m, false);
            } else if (memref::LoadOp operation = llvm::dyn_cast<memref::LoadOp>(op)) {
                nil::blueprint::memref<VarType> &memref = stack.get_memref(operation.getMemref());
                // grab the indices and build index vector
                auto indices = operation.getIndices();
                std::vector<int64_t> indicesV;
                indicesV.reserve(indices.size());
                for (auto idx : indices) {
                    // look for indices in constant_values
                    indicesV.push_back(stack.get_constant(idx));
                }
                auto value = memref.get(indicesV);
                stack.push_local(operation.getResult(), value);
            } else if (memref::StoreOp operation = llvm::dyn_cast<memref::StoreOp>(op)) {
                auto memRefHash = mlir::hash_value(operation.getMemref());
                logger.debug("looking for MemRef %x", size_t(memRefHash));
                nil::blueprint::memref<VarType> &memref = stack.get_memref(operation.getMemref());
                // grab the indices and build index vector
                auto indices = operation.getIndices();
                std::vector<int64_t> indicesV;
                indicesV.reserve(indices.size());
                for (auto idx : indices) {
                    indicesV.push_back(stack.get_constant(idx));
                }
                // grab the element from the locals array
                VarType &value = stack.get_local(operation.getValue());
                // put the element from the memref using index vector
                memref.put(indicesV, value);
            } else if (memref::DeallocOp operation = llvm::dyn_cast<memref::DeallocOp>(op)) {
                stack.erase_memref(operation.getMemref());
            } else if (memref::ReinterpretCastOp operation = llvm::dyn_cast<memref::ReinterpretCastOp>(op)) {
                auto source = operation.getSource();
                auto result = operation.getResult();
                auto result_type = operation.getType();
                logger.debug("reinterpret cast");
                logger << source;
                logger << result;
                logger << result_type;

                nil::blueprint::memref<VarType> &old_memref = stack.get_memref(source);
                auto new_memref =
                    old_memref.reinterpret_as(result_type.getShape(), result_type.getElementType(), logger);
                stack.push_memref(operation.getResult(), new_memref, false);
            } else {
                std::string opName = op->getName().getIdentifier().str();
                UNREACHABLE(std::string("unhandled memref operation: ") + opName);
            }
        }

        void handleOperation(Operation *op) {
            logger.debug("visiting operation: %s", op->getName().getIdentifier().str());
            logger.debug("current start row: %d", assignmnt.allocated_rows());
            Dialect *dial = op->getDialect();
            if (!dial) {
                logger.error("Encountered an unregistered Dialect");
                exit(-1);
            }
            ComponentParameters compParams = {assignmnt.allocated_rows(), gen_mode};
            if (llvm::isa<mlir::arith::ArithDialect>(dial)) {
                handleArithOperation(op, compParams);
                return;
            }

            if (llvm::isa<mlir::math::MathDialect>(dial)) {
                handleMathOperation(op, compParams);
                return;
            }

            if (llvm::isa<affine::AffineDialect>(dial)) {
                handleAffineOperation(op);
                return;
            }

            if (llvm::isa<mlir::memref::MemRefDialect>(dial)) {
                handleMemRefOperation(op);
                return;
            }

            if (llvm::isa<zkml::ZkMlDialect>(dial)) {
                handleZkMlOperation(op, compParams);
                return;
            }

            if (llvm::isa<mlir::KrnlDialect>(dial)) {
                handleKrnlOperation(op, compParams);
                return;
            }

            if (func::ReturnOp operation = llvm::dyn_cast<func::ReturnOp>(op)) {

                // are we returning from the entry_point?
                if (function_call_depth == 0) {
                    auto ops = operation.getOperands();
                    if (print_circuit_format != nil::blueprint::print_format::no_print) {
                        // TODO: support different print formats
                        std::cout << "Result:\n";
                        for (unsigned i = 0; i < ops.size(); ++i) {
                            nil::blueprint::memref<VarType> &retval = stack.get_memref(ops[i]);
                            retval.template print<PreLimbs, PostLimbs>(std::cout, assignmnt);
                        }
                    }

                    // constrain the output based on the public input
                    for (unsigned i = 0; i < ops.size(); ++i) {
                        nil::blueprint::memref<VarType> &retval = stack.get_memref(ops[i]);
                        nil::blueprint::memref<VarType> &output = output_memrefs[i];

                        ASSERT(retval.getDims() == output.getDims() && "output shape must match retval shape");
                        ASSERT(retval.getType() == output.getType() && "output type must match retval type");
                        for (unsigned j = 0; j < retval.size(); ++j) {
                            auto ret_var = retval.get_flat(j);
                            auto output_var = output.get_flat(j);
                            bp.add_copy_constraint({ret_var, output_var});
                            if (!output_is_already_assigned) {
                                if (output_var.index ==
                                    nil::blueprint::assignment<nil::crypto3::zk::snark::plonk_constraint_system<
                                        BlueprintFieldType, ArithmetizationParams>>::private_storage_index) {
                                    assignmnt.private_storage(output_var.rotation) = var_value(assignmnt, ret_var);
                                } else if (output_var.type ==
                                           nil::crypto3::zk::snark::plonk_variable<
                                               typename BlueprintFieldType::value_type>::column_type::public_input) {
                                    assignmnt.public_input(output_var.index, output_var.rotation) =
                                        var_value(assignmnt, ret_var);
                                } else {
                                    UNREACHABLE("Outputs must be either private or public");
                                }
                            } else {
                                typename BlueprintFieldType::value_type left = var_value(assignmnt, ret_var);
                                typename BlueprintFieldType::value_type right = var_value(assignmnt, output_var);
                                ASSERT(left == right);
                            }
                        }
                    }
                    if (!output_is_already_assigned && uint8_t(gen_mode & generation_mode::ASSIGNMENTS)) {
                        // if the output was only just calculated, we write it out into the public output file
                        nil::blueprint::OutputWriter<BlueprintFieldType, VarType,
                                                     nil::blueprint::assignment<ArithmetizationType>, PreLimbs,
                                                     PostLimbs>
                            output_writer(assignmnt, output_memrefs);
                        bool ok = output_writer.make_outputs_to_json(public_output);
                        if (!ok) {
                            std::cerr << "Could not create output JSON object";
                            const std::string &error = output_writer.get_error();
                            if (!error.empty()) {
                                std::cerr << ": " << error;
                            }
                            std::cerr << std::endl;
                            exit(-1);
                        }
                    }
                    output_is_already_assigned = true;
                } else {
                    function_call_depth -= 1;
                }
                return;
            }

            if (mlir::UnrealizedConversionCastOp operation = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
                // we do not like this but when onnx-mlir lowers from onnx.Cast to unsigned it uses this to cast
                // from signless integers (e.g. i64) to unsigned integer(e.g. ui64)
                // SO if we transform from one signless integer to an unsigned integer with the SAME bit length
                // we indulge, otherwise we panic
                mlir::Type SrcType = operation->getOperand(0).getType();
                mlir::Type DstType = operation->getResult(0).getType();
                assert(SrcType.isSignlessInteger() && "src must be signless integertype for conversion cast");
                assert(DstType.isUnsignedInteger(SrcType.getIntOrFloatBitWidth()) &&
                       "dst must be unsigned integer with same bit width as src");
                VarType &Src = stack.get_local(operation->getOperand(0));
                stack.push_local(operation->getResult(0), Src);
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
            stack.push_frame();
            for (Operation &op : block.getOperations())
                handleOperation(&op);
            stack.pop_frame();
        }

    private:
        template<typename InputType>
        VarType put_into_assignment(InputType input) {
            // TODO: optimize this further:
            //  The assignment.constant function increments the currently used row, even though it does not need
            //  to... Cannot change this without modifying the blueprint library though
            assignmnt.constant(0, constant_idx) = input;
            return VarType(0, constant_idx++, false, VarType::column_type::constant);
        }

        VarType put_float_into_assignment(double d) {
            if (d > 0 && d < lower_bound && std::isfinite(d)) {
                // check clip strategy
                if (ClipStrategy::PANIC == clip_strategy) {
                    std::cerr << "model has value lower than lowest bound \"" << lower_bound
                              << "\" and clip strategy is panic!" << std::endl;
                    std::cerr << "Try either other clip strategy or increase fixed-bits if this is not desired."
                              << std::endl;
                    UNREACHABLE("clipping");
                } else if (ClipStrategy::CLIP == clip_strategy) {
                    // lowest possible element
                    std::cerr << "warning: small element (" << d << ") encountered. Setting to smallest fixed-point."
                              << std::endl;
                    std::cerr << "Try either other clip strategy or increase fixed-bits if this is not desired."
                              << std::endl;
                    return put_into_assignment(typename BlueprintFieldType::value_type(1));
                } else if (ClipStrategy::ZERO == clip_strategy) {
                    std::cerr << "warning: small element (" << d << ") encountered. Setting to zero." << std::endl;
                    std::cerr << "Try either other clip strategy or increase fixed-bits if this is not desired."
                              << std::endl;
                    return put_into_assignment(typename BlueprintFieldType::value_type(0));
                } else {
                    UNREACHABLE("unsupported clip strategy but should be caught in constructor?");
                }
            } else {
                FixedPoint fixed(d);
                return put_into_assignment(fixed.get_value());
            }
        }

        generation_mode gen_mode;
        nil::blueprint::print_format print_circuit_format;
        nil::blueprint::logger &logger;

        nil::blueprint::stack<VarType> stack;
        std::map<std::string, func::FuncOp> functions;
        size_t function_call_depth = 0;
        std::vector<nil::blueprint::memref<VarType>> output_memrefs;
        bool output_is_already_assigned = false;
        nil::blueprint::circuit_proxy<ArithmetizationType> &bp;
        nil::blueprint::assignment_proxy<ArithmetizationType> &assignmnt;
        const boost::json::array &public_input;
        const boost::json::array &private_input;
        boost::json::array &public_output;
        size_t constant_idx = 0;
        unsigned amount_ops = 0;
        unsigned progress = 0;
        VarType undef_var;
        VarType zero_var;
        std::optional<VarType> true_var;
        double lower_bound;
        ClipStrategy clip_strategy;
    };
}    // namespace zk_ml_toolchain

#endif    // CRYPTO3_BLUEPRINT_COMPONENT_INSTRUCTION_MLIR_EVALUATOR_HPP
