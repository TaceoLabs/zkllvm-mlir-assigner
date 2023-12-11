#ifndef CRYPTO3_ASSIGNER_INTEGER_NEG_HPP
#define CRYPTO3_ASSIGNER_INTEGER_NEG_HPP

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/component.hpp>
#include <nil/blueprint/basic_non_native_policy.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/neg.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/lookup_tables/tester.hpp>

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/policy/policy_manager.hpp>

namespace nil {
namespace blueprint {
namespace detail {

template <typename BlueprintFieldType, typename ArithmetizationParams>
typename components::fix_neg<
    crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType,
                                                ArithmetizationParams>,
    BlueprintFieldType,
    basic_non_native_policy<BlueprintFieldType>>::result_type
handle_integer_neg_component(
    crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>
        x,
    circuit_proxy<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &bp,
    assignment_proxy<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &assignment,
    std::uint32_t start_row) {

  using var = crypto3::zk::snark::plonk_variable<
      typename BlueprintFieldType::value_type>;

  using component_type = components::fix_neg<
      crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType,
                                                  ArithmetizationParams>,
      BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

  using manifest_reader = ManifestReader<component_type, ArithmetizationParams>;
  const auto p = PolicyManager::get_parameters(manifest_reader::get_witness(0));
  component_type component_instance(p.witness, manifest_reader::get_constants(),
                                    manifest_reader::get_public_inputs());

  // TACEO_TODO in the previous line I hardcoded 1 for now!!! CHANGE THAT
  // TACEO_TODO make an assert that both have the same scale?
  // TACEO_TODO we probably have to extract the field element from the type here
  if constexpr (nil::blueprint::use_custom_lookup_tables<component_type>()) {
    auto lookup_tables = component_instance.component_custom_lookup_tables();
    for (auto &t : lookup_tables) {
      bp.register_lookup_table(
          std::shared_ptr<nil::crypto3::zk::snark::lookup_table_definition<
              BlueprintFieldType>>(t));
    }
  };

  if constexpr (nil::blueprint::use_lookups<component_type>()) {
    auto lookup_tables = component_instance.component_lookup_tables();
    for (auto &[k, v] : lookup_tables) {
      bp.reserve_table(k);
    }
  };

  components::generate_circuit(component_instance, bp, assignment, {x},
                               start_row);
  return components::generate_assignments(component_instance, assignment, {x},
                                          start_row);
}

} // namespace detail
} // namespace blueprint
} // namespace nil

#endif // CRYPTO3_ASSIGNER_INTEGER_NEG_HPP
