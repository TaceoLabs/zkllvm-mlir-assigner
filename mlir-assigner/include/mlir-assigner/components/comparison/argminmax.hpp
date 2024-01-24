#ifndef CRYPTO3_ASSIGNER_FIXEDPOINT_ARGMINMAX_HPP
#define CRYPTO3_ASSIGNER_FIXEDPOINT_ARGMINMAX_HPP

#include "mlir/Dialect/zkml/IR/ArgMin.h"
#include "mlir/Dialect/zkml/IR/ArgMax.h"
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/component.hpp>
#include <nil/blueprint/basic_non_native_policy.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/lookup_tables/tester.hpp>    // TODO: check if there is a new mechanism for this in nil upstream

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>
#include <mlir-assigner/components/handle_component.hpp>

namespace nil {
    namespace blueprint {

        template<typename BlueprintFieldType, typename ArithmetizationParams>
        void handle_argmin(
            mlir::zkml::ArgMinOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type> &nextIndex,
            std::uint32_t start_row) {
            using component_type = components::fix_argmin<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            using input_type = typename component_type::input_type;
            input_type instance_input;
            instance_input.x = stack.get_local(operation.getAcc());
            instance_input.y = stack.get_local(operation.getNext());
            instance_input.index_x = stack.get_local(operation.getAccIndex());

            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams, 1, 1>;
            const auto p = detail::PolicyManager::get_parameters(
                detail::ManifestReader<component_type, ArithmetizationParams, 1, 1>::get_witness(0, 1, 1));
            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs(),
                                     1, 1, var_value(assignment, nextIndex), operation.getSelectLastIndex());

            if constexpr (nil::blueprint::use_custom_lookup_tables<component_type>()) {
                auto lookup_tables = component.component_custom_lookup_tables();
                for (auto &t : lookup_tables) {
                    bp.register_lookup_table(
                        std::shared_ptr<nil::crypto3::zk::snark::lookup_table_definition<BlueprintFieldType>>(t));
                }
            };

            if constexpr (nil::blueprint::use_lookups<component_type>()) {
                auto lookup_tables = component.component_lookup_tables();
                for (auto &[k, v] : lookup_tables) {
                    bp.reserve_table(k);
                }
            };

            handle_component_input<BlueprintFieldType, ArithmetizationParams, component_type>(assignment,
                                                                                              instance_input);

            components::generate_circuit(component, bp, assignment, instance_input, start_row);
            auto result = components::generate_assignments(component, assignment, instance_input, start_row);
            stack.push_local(operation.getResult(0), result.min);
            stack.push_local(operation.getResult(1), result.index);
        }

        template<typename BlueprintFieldType, typename ArithmetizationParams>
        void handle_argmax(
            mlir::zkml::ArgMaxOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type> &nextIndex,
            std::uint32_t start_row) {
            using component_type = components::fix_argmax<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            using input_type = typename component_type::input_type;
            input_type instance_input;
            instance_input.x = stack.get_local(operation.getAcc());
            instance_input.y = stack.get_local(operation.getNext());
            instance_input.index_x = stack.get_local(operation.getAccIndex());

            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams, 1, 1>;
            const auto p = detail::PolicyManager::get_parameters(
                detail::ManifestReader<component_type, ArithmetizationParams, 1, 1>::get_witness(0, 1, 1));
            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs(),
                                     1, 1, var_value(assignment, nextIndex), operation.getSelectLastIndex());

            if constexpr (nil::blueprint::use_custom_lookup_tables<component_type>()) {
                auto lookup_tables = component.component_custom_lookup_tables();
                for (auto &t : lookup_tables) {
                    bp.register_lookup_table(
                        std::shared_ptr<nil::crypto3::zk::snark::lookup_table_definition<BlueprintFieldType>>(t));
                }
            };

            if constexpr (nil::blueprint::use_lookups<component_type>()) {
                auto lookup_tables = component.component_lookup_tables();
                for (auto &[k, v] : lookup_tables) {
                    bp.reserve_table(k);
                }
            };

            handle_component_input<BlueprintFieldType, ArithmetizationParams, component_type>(assignment,
                                                                                              instance_input);

            components::generate_circuit(component, bp, assignment, instance_input, start_row);
            auto result = components::generate_assignments(component, assignment, instance_input, start_row);
            stack.push_local(operation.getResult(0), result.max);
            stack.push_local(operation.getResult(1), result.index);
        }
    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_FIXEDPOINT_ARGMINMAX_HPP
