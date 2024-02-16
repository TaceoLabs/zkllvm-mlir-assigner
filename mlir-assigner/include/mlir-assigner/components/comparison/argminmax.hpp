#ifndef CRYPTO3_ASSIGNER_FIXEDPOINT_ARGMINMAX_HPP
#define CRYPTO3_ASSIGNER_FIXEDPOINT_ARGMINMAX_HPP

#include "mlir/Dialect/zkml/IR/ArgMin.h"
#include "mlir/Dialect/zkml/IR/ArgMax.h"
#include <cstdint>
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/component.hpp>
#include <nil/blueprint/basic_non_native_policy.hpp>

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>
#include <mlir-assigner/components/handle_component.hpp>

namespace nil {
    namespace blueprint {

        template<std::uint32_t PreLimbs, std::uint32_t PostLimbs, typename BlueprintFieldType,
                 typename ArithmetizationParams>
        void handle_argmin(
            mlir::zkml::ArgMinOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type> &nextIndex,
            const common_component_parameters<
                crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &compParams) {
            using component_type = components::fix_argmin<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            typename component_type::input_type input;
            input.x = stack.get_local(operation.getAcc());
            input.y = stack.get_local(operation.getNext());
            input.index_x = stack.get_local(operation.getAccIndex());

            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams, PreLimbs, PostLimbs>;
            const auto p = detail::PolicyManager::get_parameters(manifest_reader::get_witness(0, PreLimbs, PostLimbs));
            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs(),
                                     PreLimbs, PostLimbs, var_value(assignment, nextIndex),
                                     operation.getSelectLastIndex());

            auto result = fill_trace_get_result(component, input, operation, stack, bp, assignment, compParams);
            stack.push_local(operation.getResult(0), result.min);
            stack.push_local(operation.getResult(1), result.index);
        }

        template<std::uint32_t PreLimbs, std::uint32_t PostLimbs, typename BlueprintFieldType,
                 typename ArithmetizationParams>
        void handle_argmax(
            mlir::zkml::ArgMaxOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type> &nextIndex,
            const common_component_parameters<
                crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &compParams) {
            using component_type = components::fix_argmax<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            typename component_type::input_type input;
            input.x = stack.get_local(operation.getAcc());
            input.y = stack.get_local(operation.getNext());
            input.index_x = stack.get_local(operation.getAccIndex());

            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams, PreLimbs, PostLimbs>;
            const auto p = detail::PolicyManager::get_parameters(manifest_reader::get_witness(0, PreLimbs, PostLimbs));
            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs(),
                                     PreLimbs, PostLimbs, var_value(assignment, nextIndex),
                                     operation.getSelectLastIndex());

            auto result = fill_trace_get_result(component, input, operation, stack, bp, assignment, compParams);
            stack.push_local(operation.getResult(0), result.max);
            stack.push_local(operation.getResult(1), result.index);
        }
    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_FIXEDPOINT_ARGMINMAX_HPP
