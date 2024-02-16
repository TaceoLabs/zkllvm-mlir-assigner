#ifndef CRYPTO3_ASSIGNER_FIXEDPOINT_NEG_HPP
#define CRYPTO3_ASSIGNER_FIXEDPOINT_NEG_HPP

#include <mlir/Dialect/Arith/IR/Arith.h>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/component.hpp>
#include <nil/blueprint/basic_non_native_policy.hpp>

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>
#include <mlir-assigner/components/handle_component.hpp>

namespace nil {
    namespace blueprint {

        template<typename BlueprintFieldType, typename ArithmetizationParams>
        void handle_neg(
            mlir::arith::NegFOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            const common_component_parameters<
                crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &compParams) {
            using component_type = components::fix_neg<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams>;
            const auto p = detail::PolicyManager::get_parameters(manifest_reader::get_witness(0));
            auto input = PREPARE_UNARY_INPUT(mlir::arith::NegFOp);
            component_type component_instance(p.witness, manifest_reader::get_constants(),
                                              manifest_reader::get_public_inputs());
            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs());
            fill_trace(component, input, operation, stack, bp, assignment, compParams);
        }
    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_FIXEDPOINT_NEG_HPP
