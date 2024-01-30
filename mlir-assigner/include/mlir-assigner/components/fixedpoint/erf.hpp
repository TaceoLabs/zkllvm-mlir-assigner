#ifndef CRYPTO3_ASSIGNER_FIXEDPOINT_ERF_HPP
#define CRYPTO3_ASSIGNER_FIXEDPOINT_ERF_HPP

#include <mlir/Dialect/Math/IR/Math.h>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/component.hpp>
#include <nil/blueprint/basic_non_native_policy.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/lookup_tables/tester.hpp>    // TODO: check if there is a new mechanism for this in nil upstream

#include <mlir-assigner/components/handle_component.hpp>

namespace nil {
    namespace blueprint {

        template<std::uint8_t PreLimbs, std::uint8_t PostLimbs, typename BlueprintFieldType,
                 typename ArithmetizationParams>
        void handle_erf(
            mlir::math::ErfOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {
            using component_type = components::fix_erf<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            auto input = PREPARE_UNARY_INPUT(mlir::math::ErfOp);
            using manifest_reader =
                detail::ManifestReader<component_type, ArithmetizationParams, PreLimbs, PostLimbs>;
            const auto p =
                detail::PolicyManager::get_parameters(manifest_reader::get_witness(0, PreLimbs, PostLimbs));

            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs(),
                                     PreLimbs, PostLimbs);
            fill_trace(component, input, operation, stack, bp, assignment, start_row);
        }
    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_FIXEDPOINT_ERF_HPP
