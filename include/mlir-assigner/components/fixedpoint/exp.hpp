#ifndef CRYPTO3_ASSIGNER_FIXEDPOINT_EXP_HPP
#define CRYPTO3_ASSIGNER_FIXEDPOINT_EXP_HPP

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/component.hpp>
#include <nil/blueprint/basic_non_native_policy.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/exp.hpp>

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>
#include <mlir-assigner/policy/policy_manager.hpp>

namespace nil {
namespace blueprint {
namespace detail {

template <typename BlueprintFieldType, typename ArithmetizationParams>
typename components::fix_exp<
    crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType,
                                                ArithmetizationParams>,
    BlueprintFieldType,
    basic_non_native_policy<BlueprintFieldType>>::result_type
handle_fixedpoint_exp_component(
    crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>
        x,
    circuit<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &bp,
    assignment<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &assignment,
    std::uint32_t start_row) {

  using var = crypto3::zk::snark::plonk_variable<
      typename BlueprintFieldType::value_type>;

  using component_type = components::fix_exp<
      crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType,
                                                  ArithmetizationParams>,
      BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;
  const auto p = PolicyManager::get_parameters(
      ManifestReader<component_type, ArithmetizationParams, 1>::get_witness(0));
  component_type component_instance(
      p.witness,
      ManifestReader<component_type, ArithmetizationParams, 1>::get_constants(),
      ManifestReader<component_type, ArithmetizationParams,
                     1>::get_public_inputs(),
      1);

  // TACEO_TODO in the previous line I hardcoded 1 for now!!! CHANGE THAT
  // TACEO_TODO make an assert that both have the same scale?
  // TACEO_TODO we probably have to extract the field element from the type here

  components::generate_circuit(component_instance, bp, assignment, {x},
                               start_row);
  return components::generate_assignments(component_instance, assignment, {x},
                                          start_row);
}

} // namespace detail
template <typename BlueprintFieldType, typename ArithmetizationParams>
void handle_fixedpoint_exp_component(
    const llvm::Instruction *inst,
    stack_frame<crypto3::zk::snark::plonk_variable<
        typename BlueprintFieldType::value_type>> &frame,
    circuit<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &bp,
    assignment<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &assignment,
    std::uint32_t start_row) {
  llvm::Value *operand = inst->getOperand(0);
  llvm::Type *op_type = operand->getType();
  ASSERT(llvm::isa<llvm::ZkFixedPointType>(op_type));
  frame.scalars[inst] =
      detail::handle_fixedpoint_exp_component<BlueprintFieldType,
                                              ArithmetizationParams>(
          operand, frame.scalars, bp, assignment, start_row)
          .output;

  // TACEO_TODO check Scale size here in LLVM???
  // ASSERT(llvm::cast<llvm::GaloisFieldType>(op0_type)->getFieldKind() ==
  //        llvm::cast<llvm::GaloisFieldType>(op1_type)->getFieldKind());
}
} // namespace blueprint
} // namespace nil

#endif // CRYPTO3_ASSIGNER_FIXEDPOINT_EXP_HPP
