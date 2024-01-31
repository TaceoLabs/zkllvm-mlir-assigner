#ifndef CRYPTO3_ASSIGNER_FIXEDPOINT_TRIGONOMETRIC_HPP
#define CRYPTO3_ASSIGNER_FIXEDPOINT_TRIGONOMETRIC_HPP

#include "mlir/Dialect/zkml/IR/DotProduct.h"
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

        template<std::uint8_t PreLimbs, std::uint8_t PostLimbs, typename BlueprintFieldType,
                 typename ArithmetizationParams>
        void handle_sin(
            mlir::math::SinOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {
            using component_type = components::fix_sin<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            auto input = PREPARE_UNARY_INPUT(mlir::math::SinOp);
            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams, PreLimbs, PostLimbs>;
            const auto p = detail::PolicyManager::get_parameters(manifest_reader::get_witness(0, PreLimbs, PostLimbs));

            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs(),
                                     PreLimbs, PostLimbs);
            fill_trace(component, input, operation, stack, bp, assignment, start_row);
        }

        template<std::uint8_t PreLimbs, std::uint8_t PostLimbs, typename BlueprintFieldType,
                 typename ArithmetizationParams>
        void handle_sinh(
            mlir::zkml::SinhOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {
            using component_type = components::fix_sinh<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            auto input = PREPARE_UNARY_INPUT(mlir::zkml::SinhOp);
            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams, PreLimbs, PostLimbs>;
            const auto p = detail::PolicyManager::get_parameters(manifest_reader::get_witness(0, PreLimbs, PostLimbs));
            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs(),
                                     PreLimbs, PostLimbs);
            fill_trace(component, input, operation, stack, bp, assignment, start_row);
        }

        template<std::uint8_t PreLimbs, std::uint8_t PostLimbs, typename BlueprintFieldType,
                 typename ArithmetizationParams>
        void handle_cos(
            mlir::math::CosOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {
            using component_type = components::fix_cos<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            auto input = PREPARE_UNARY_INPUT(mlir::math::CosOp);
            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams, PreLimbs, PostLimbs>;
            const auto p = detail::PolicyManager::get_parameters(manifest_reader::get_witness(0, PreLimbs, PostLimbs));

            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs(),
                                     PreLimbs, PostLimbs);
            fill_trace(component, input, operation, stack, bp, assignment, start_row);
        }

        template<std::uint8_t PreLimbs, std::uint8_t PostLimbs, typename BlueprintFieldType,
                 typename ArithmetizationParams>
        void handle_cosh(
            mlir::zkml::CoshOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {
            using component_type = components::fix_cosh<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            auto input = PREPARE_UNARY_INPUT(mlir::zkml::CoshOp);
            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams, PreLimbs, PostLimbs>;
            const auto p = detail::PolicyManager::get_parameters(manifest_reader::get_witness(0, PreLimbs, PostLimbs));

            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs(),
                                     PreLimbs, PostLimbs);
            fill_trace(component, input, operation, stack, bp, assignment, start_row);
        }

        template<std::uint8_t PreLimbs, std::uint8_t PostLimbs, typename BlueprintFieldType,
                 typename ArithmetizationParams>
        void handle_tan(
            mlir::KrnlTanOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {
            using component_type = components::fix_tan<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            auto input = PREPARE_UNARY_INPUT(mlir::KrnlTanOp);
            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams, PreLimbs, PostLimbs>;
            const auto p = detail::PolicyManager::get_parameters(manifest_reader::get_witness(0, PreLimbs, PostLimbs));

            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs(),
                                     PreLimbs, PostLimbs);
            fill_trace(component, input, operation, stack, bp, assignment, start_row);
        }
        template<std::uint8_t PreLimbs, std::uint8_t PostLimbs, typename BlueprintFieldType,
                 typename ArithmetizationParams>
        void handle_tanh(
            mlir::math::TanhOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {
            using component_type = components::fix_tanh<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            auto input = PREPARE_UNARY_INPUT(mlir::math::TanhOp);
            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams, PreLimbs, PostLimbs>;
            const auto p = detail::PolicyManager::get_parameters(manifest_reader::get_witness(0, PreLimbs, PostLimbs));

            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs(),
                                     PreLimbs, PostLimbs);
            fill_trace(component, input, operation, stack, bp, assignment, start_row);
        }

        template<std::uint8_t PreLimbs, std::uint8_t PostLimbs, typename BlueprintFieldType,
                 typename ArithmetizationParams>
        void handle_atan(
            mlir::KrnlAtanOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {
            using component_type = components::fix_atan<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            auto input = PREPARE_UNARY_INPUT(mlir::KrnlAtanOp);
            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams, PreLimbs, PostLimbs>;
            const auto p = detail::PolicyManager::get_parameters(manifest_reader::get_witness(0));

            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs(),
                                     PreLimbs, PostLimbs);
            fill_trace(component, input, operation, stack, bp, assignment, start_row);
        }
    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_FIXEDPOINT_TRIGONOMETRIC_HPP
