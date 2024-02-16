#ifndef CRYPTO3_ASSIGNER_INTEGER_ARITH_HPP
#define CRYPTO3_ASSIGNER_INTEGER_ARITH_HPP

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/component.hpp>
#include <nil/blueprint/basic_non_native_policy.hpp>

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>
#include <mlir-assigner/components/handle_component.hpp>

namespace nil {
    namespace blueprint {

        template<typename MlirOp, typename BlueprintFieldType, typename ArithmetizationParams>
        void handle_add(
            MlirOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            const common_component_parameters<
                crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &compParams) {
            using component_type = components::addition<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType,
                basic_non_native_policy<BlueprintFieldType>>;
            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams>;
            auto input = PREPARE_BINARY_INPUT(MlirOp);
            const auto p = detail::PolicyManager::get_parameters(manifest_reader::get_witness(0));
            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs());
            fill_trace(component, input, operation, stack, bp, assignment, compParams);
        }

        template<typename MlirOp, typename BlueprintFieldType, typename ArithmetizationParams>
        void handle_sub(
            MlirOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            const common_component_parameters<
                crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &compParams) {
            using component_type = components::subtraction<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType,
                basic_non_native_policy<BlueprintFieldType>>;
            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams>;
            auto input = PREPARE_BINARY_INPUT(MlirOp);
            const auto p = detail::PolicyManager::get_parameters(manifest_reader::get_witness(0));
            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs());
            fill_trace(component, input, operation, stack, bp, assignment, compParams);
        }
    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_INTEGER_ARITH_HPP
