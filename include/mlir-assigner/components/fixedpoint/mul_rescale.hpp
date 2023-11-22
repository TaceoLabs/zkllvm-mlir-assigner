#ifndef CRYPTO3_ASSIGNER_FIXEDPOINT_MULTIPLICATION_RESCALE_HPP
#define CRYPTO3_ASSIGNER_FIXEDPOINT_MULTIPLICATION_RESCALE_HPP

#include <mlir/Dialect/Arith/IR/Arith.h>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/component.hpp>
#include <nil/blueprint/basic_non_native_policy.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/mul_rescale.hpp>

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>
#include <mlir-assigner/policy/policy_manager.hpp>

namespace nil {
namespace blueprint {
namespace detail {

template <typename BlueprintFieldType, typename ArithmetizationParams>
typename components::fix_mul_rescale<
    crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType,
                                                ArithmetizationParams>,
    BlueprintFieldType,
    basic_non_native_policy<BlueprintFieldType>>::result_type
handle_fixedpoint_mul_rescale_component(
    crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>
        x,
    crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>
        y,
    circuit<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &bp,
    assignment<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &assignment,
    std::uint32_t start_row) {

  using component_type = components::fix_mul_rescale<
      crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType,
                                                  ArithmetizationParams>,
      BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;
  const auto p = PolicyManager::get_parameters(
      ManifestReader<component_type, ArithmetizationParams, 1, 1>::get_witness(
          0));
  component_type component_instance(
      p.witness,
      ManifestReader<component_type, ArithmetizationParams, 1,
                     1>::get_constants(),
      ManifestReader<component_type, ArithmetizationParams, 1,
                     1>::get_public_inputs(),
      1);

  // TACEO_TODO in the previous line I hardcoded 1 for now!!! CHANGE THAT
  // TACEO_TODO make an assert that both have the same scale?
  // TACEO_TODO we probably have to extract the field element from the type here

  components::generate_circuit(component_instance, bp, assignment, {x, y},
                               start_row);
  return components::generate_assignments(component_instance, assignment,
                                          {x, y}, start_row);
}

} // namespace detail
template <typename BlueprintFieldType, typename ArithmetizationParams>
void handle_fixedpoint_mul_rescale_component(
    mlir::arith::MulFOp &operation,
    stack_frame<crypto3::zk::snark::plonk_variable<
        typename BlueprintFieldType::value_type>> &frame,
    circuit<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &bp,
    assignment<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &assignment,
    std::uint32_t start_row) {

  auto lhs = frame.locals.find(mlir::hash_value(operation.getLhs()));
  ASSERT(lhs != frame.locals.end());
  auto rhs = frame.locals.find(mlir::hash_value(operation.getRhs()));
  ASSERT(rhs != frame.locals.end());

  auto x = lhs->second;
  auto y = rhs->second;

  // TACEO_TODO: check types

  auto result = detail::handle_fixedpoint_mul_rescale_component(
      x, y, bp, assignment, start_row);
  frame.locals[mlir::hash_value(operation.getResult())] = result.output;
}
} // namespace blueprint
} // namespace nil

#endif // CRYPTO3_ASSIGNER_FIXEDPOINT_MULTIPLICATION_RESCALE_HPP