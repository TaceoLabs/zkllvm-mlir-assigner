
#ifndef CRYPTO3_ASSIGNER_FIXEDPOINT_SUBTRACTION_HPP
#define CRYPTO3_ASSIGNER_FIXEDPOINT_SUBTRACTION_HPP

#include <mlir/Dialect/Arith/IR/Arith.h>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <mlir-assigner/components/fields/subtraction.hpp>

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>

namespace nil {
namespace blueprint {

template <typename BlueprintFieldType, typename ArithmetizationParams>
void handle_fixedpoint_subtraction_component(
    mlir::arith::SubFOp &operation,
    stack_frame<crypto3::zk::snark::plonk_variable<
        typename BlueprintFieldType::value_type>> &frame,
    circuit_proxy<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &bp,
    assignment_proxy<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &assignment,
    std::uint32_t start_row) {

  auto lhs = frame.locals.find(mlir::hash_value(operation.getLhs()));
  ASSERT(lhs != frame.locals.end());
  auto rhs = frame.locals.find(mlir::hash_value(operation.getRhs()));
  ASSERT(rhs != frame.locals.end());

  auto x = lhs->second;
  auto y = rhs->second;

  // TACEO_TODO: check types

  auto result = detail::handle_native_field_subtraction_component(
      x, y, bp, assignment, start_row);
  frame.locals[mlir::hash_value(operation.getResult())] = result.output;
}
} // namespace blueprint
} // namespace nil

#endif // CRYPTO3_ASSIGNER_FIXEDPOINT_SUBTRACTION_HPP
