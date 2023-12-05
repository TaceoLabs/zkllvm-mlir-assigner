#ifndef CRYPTO3_ASSIGNER_FIXEDPOINT_DOT_PRODUCT_HPP
#define CRYPTO3_ASSIGNER_FIXEDPOINT_DOT_PRODUCT_HPP

#include "mlir/Dialect/zkml/IR/DotProduct.h"
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/component.hpp>
#include <nil/blueprint/basic_non_native_policy.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/dot_rescale_2_gates.hpp>

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>

#include <mlir-assigner/components/fields/addition.hpp>

namespace nil {
namespace blueprint {
namespace detail {

template <typename BlueprintFieldType, typename ArithmetizationParams>
typename components::fix_dot_rescale_2_gates<
    crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType,
                                                ArithmetizationParams>,
    BlueprintFieldType,
    basic_non_native_policy<BlueprintFieldType>>::result_type
handle_fixedpoint_dot_product_component(
    memref<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>>
        x,
    memref<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>>
        y,
    crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type> &zero_var,
    circuit<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &bp,
    assignment<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &assignment,
    std::uint32_t start_row) {

  using var = crypto3::zk::snark::plonk_variable<
      typename BlueprintFieldType::value_type>;

  using component_type = components::fix_dot_rescale_2_gates<
      crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType,
                                                  ArithmetizationParams>,
      BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;
  using manifest_reader =
      ManifestReader<component_type, ArithmetizationParams, 1, 1>;
  auto dims = x.getDims();
  ASSERT(dims.size() == 1 && "must be one-dim for dot product");
  const auto p = PolicyManager::get_parameters(manifest_reader::get_witness(0, dims.front(), 1));
  component_type component_instance(p.witness, manifest_reader::get_constants(),
                                    manifest_reader::get_public_inputs(), dims.front(), 1);

  if constexpr (nil::blueprint::use_custom_lookup_tables<component_type>()) {
    auto lookup_tables = component_instance.component_custom_lookup_tables();
    for (auto &t : lookup_tables) {
      bp.register_lookup_table(
          std::shared_ptr<nil::crypto3::zk::snark::detail::
                              lookup_table_definition<BlueprintFieldType>>(t));
    }
  };

  if constexpr (nil::blueprint::use_lookups<component_type>()) {
    auto lookup_tables = component_instance.component_lookup_tables();
    for (auto &[k, v] : lookup_tables) {
      bp.reserve_table(k);
    }
  };

  using DotProductInputType = const typename components::plonk_fixedpoint_dot_rescale_2_gates<
                        BlueprintFieldType, ArithmetizationParams>::input_type;


  // TACEO_TODO in the previous line I hardcoded 1 for now!!! CHANGE THAT
  // TACEO_TODO make an assert that both have the same scale?
  // TACEO_TODO we probably have to extract the field element from the type here

  DotProductInputType input = {x.getData(), y.getData(), zero_var};

  components::generate_circuit(component_instance, bp, assignment, input,
                               start_row);
  return components::generate_assignments(component_instance, assignment, input,
                                           start_row);
}

} // namespace detail

template <typename BlueprintFieldType, typename ArithmetizationParams>
void handle_fixedpoint_dot_product_component(
    mlir::zkml::DotProductOp &operation,
    crypto3::zk::snark::plonk_variable<
            typename BlueprintFieldType::value_type> &zero_var,
    stack_frame<crypto3::zk::snark::plonk_variable<
        typename BlueprintFieldType::value_type>> &frame,
    circuit<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &bp,
    assignment<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &assignment) {

  auto lhs = frame.memrefs.find(mlir::hash_value(operation.getLhs()));
  ASSERT(lhs != frame.memrefs.end());
  auto rhs = frame.memrefs.find(mlir::hash_value(operation.getRhs()));
  ASSERT(rhs != frame.memrefs.end());

  auto x = lhs->second;
  auto y = rhs->second;

  // TACEO_TODO: check types

  auto result = detail::handle_fixedpoint_dot_product_component(
      x, y, zero_var, bp, assignment, assignment.allocated_rows());
  frame.locals[mlir::hash_value(operation.getResult())] = result.output;
}
} // namespace blueprint
} // namespace nil

#endif // CRYPTO3_ASSIGNER_FIXEDPOINT_DOT_PRODUCT_HPP
