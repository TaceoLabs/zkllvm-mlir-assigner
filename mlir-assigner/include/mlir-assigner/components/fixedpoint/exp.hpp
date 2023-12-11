#ifndef CRYPTO3_ASSIGNER_FIXEDPOINT_EXP_HPP
#define CRYPTO3_ASSIGNER_FIXEDPOINT_EXP_HPP

#include <mlir/Dialect/Math/IR/Math.h>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/component.hpp>
#include <nil/blueprint/basic_non_native_policy.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/exp.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/lookup_tables/tester.hpp>    // TODO: check if there is a new mechanism for this in nil upstream

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>
#include <mlir-assigner/policy/policy_manager.hpp>

namespace nil {
    namespace blueprint {
        namespace detail {

            template<typename BlueprintFieldType, typename ArithmetizationParams>
            typename components::fix_exp<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>::result_type
                handle_fixedpoint_exp_component(
                    crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>
                        x,
                    circuit_proxy<
                        crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
                    assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType,
                                                                                 ArithmetizationParams>> &assignment,
                    std::uint32_t start_row) {

                using var = crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>;

                using component_type = components::fix_exp<
                    crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                    BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;
                const auto p = PolicyManager::get_parameters(
                    ManifestReader<component_type, ArithmetizationParams, 1>::get_witness(0));
                component_type component_instance(
                    p.witness,
                    ManifestReader<component_type, ArithmetizationParams, 1>::get_constants(),
                    ManifestReader<component_type, ArithmetizationParams, 1>::get_public_inputs(),
                    1);

                if constexpr (nil::blueprint::use_custom_lookup_tables<component_type>()) {
                    auto lookup_tables = component_instance.component_custom_lookup_tables();
                    for (auto &t : lookup_tables) {
                        bp.register_lookup_table(
                            std::shared_ptr<nil::crypto3::zk::snark::lookup_table_definition<BlueprintFieldType>>(t));
                    }
                };

                if constexpr (nil::blueprint::use_lookups<component_type>()) {
                    auto lookup_tables = component_instance.component_lookup_tables();
                    for (auto &[k, v] : lookup_tables) {
                        bp.reserve_table(k);
                    }
                };

                // TACEO_TODO in the previous line I hardcoded 1 for now!!! CHANGE THAT
                // TACEO_TODO make an assert that both have the same scale?
                // TACEO_TODO we probably have to extract the field element from the type here

                components::generate_circuit(component_instance, bp, assignment, {x}, start_row);
                return components::generate_assignments(component_instance, assignment, {x}, start_row);
            }

        }    // namespace detail
        template<typename BlueprintFieldType, typename ArithmetizationParams>
        void handle_fixedpoint_exp_component(
            mlir::math::ExpOp &operation,
            stack_frame<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &frame,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {
            auto operand = frame.locals.find(mlir::hash_value(operation.getOperand()));
            ASSERT(operand != frame.locals.end());

            auto x = operand->second;

            // TACEO_TODO: check types

            auto result = detail::handle_fixedpoint_exp_component(x, bp, assignment, start_row);
            frame.locals[mlir::hash_value(operation.getResult())] = result.output;
        }
    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_FIXEDPOINT_EXP_HPP
