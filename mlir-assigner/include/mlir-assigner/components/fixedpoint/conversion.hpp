#ifndef CRYPTO3_ASSIGNER_FIXEDPOINT_TO_FIXEDPOINT_HPP
#define CRYPTO3_ASSIGNER_FIXEDPOINT_TO_FIXEDPOINT_HPP

#include "mlir/Dialect/zkml/IR/DotProduct.h"
#include <cstdint>
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
                 typename ArithmetizationParams, typename MlirOp>
        void handle_to_fixedpoint(
            MlirOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            const common_component_parameters<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &compParams) {
            using component_type = components::int_to_fix<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            auto input = PREPARE_UNARY_INPUT(MlirOp);
            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams, PreLimbs, PostLimbs>;
            // TODO are those manifest readers correct?
            const auto p = detail::PolicyManager::get_parameters(
                detail::ManifestReader<component_type, ArithmetizationParams>::get_witness(0));

            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs(),
                                     PostLimbs);
            fill_trace(component, input, operation, stack, bp, assignment, compParams);
        }
        namespace detail {
            template<std::uint8_t PreLimbs, std::uint8_t PostLimbs, typename BlueprintFieldType,
                     typename ArithmetizationParams, typename MlirOp, uint8_t OutputType>
            void handle_to_int(
                MlirOp &operation,
                stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
                circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                    &bp,
                assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                    &assignment,
                const common_component_parameters<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &compParams) {
                using component_type = components::fix_to_int<
                    crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                    BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;
                typename component_type::OutputType outputType =
                    static_cast<typename component_type::OutputType>(OutputType);
                auto input = PREPARE_UNARY_INPUT(MlirOp);
                using manifest_reader =
                    detail::ManifestReader<component_type, ArithmetizationParams, PreLimbs, PostLimbs, OutputType>;
                const auto p = detail::PolicyManager::get_parameters(manifest_reader::get_witness(0));

                component_type component(p.witness, manifest_reader::get_constants(),
                                         manifest_reader::get_public_inputs(), PreLimbs, PostLimbs, outputType);
                fill_trace(component, input, operation, stack, bp, assignment, compParams);
            }
        }    // namespace detail

#define HANDLE_TO_INT(TY)                                                                                             \
    detail::handle_to_int<PreLimbs, PostLimbs, BlueprintFieldType, ArithmetizationParams, mlir::arith::FPToSIOp, TY>( \
        operation, stack, bp, assignment, compParams);

#define HANDLE_TO_UINT(TY)                                                                                            \
    detail::handle_to_int<PreLimbs, PostLimbs, BlueprintFieldType, ArithmetizationParams, mlir::arith::FPToUIOp, TY>( \
        operation, stack, bp, assignment, compParams);

        template<std::uint8_t PreLimbs, std::uint8_t PostLimbs, typename BlueprintFieldType,
                 typename ArithmetizationParams>
        void handle_to_int(
            mlir::arith::FPToSIOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            const common_component_parameters<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &compParams) {
            using component_type = components::fix_to_int<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;
            switch (operation->getResult(0).getType().getIntOrFloatBitWidth()) {
                case 8:
                    HANDLE_TO_INT(component_type::OutputType::I8);
                    break;
                case 16:
                    HANDLE_TO_INT(component_type::OutputType::I16);
                    break;
                case 32:
                    HANDLE_TO_INT(component_type::OutputType::I32);
                    break;
                case 64:
                    HANDLE_TO_INT(component_type::OutputType::I64);
                    break;
            }
        }

        template<std::uint8_t PreLimbs, std::uint8_t PostLimbs, typename BlueprintFieldType,
                 typename ArithmetizationParams>
        void handle_to_int(
            mlir::arith::FPToUIOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            const common_component_parameters<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &compParams) {
            using component_type = components::fix_to_int<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;
            switch (operation->getResult(0).getType().getIntOrFloatBitWidth()) {
                case 8:
                    HANDLE_TO_UINT(component_type::OutputType::U8);
                    break;
                case 16:
                    HANDLE_TO_UINT(component_type::OutputType::U16);
                    break;
                case 32:
                    HANDLE_TO_UINT(component_type::OutputType::U32);
                    break;
                case 64:
                    HANDLE_TO_UINT(component_type::OutputType::U64);
                    break;
            }
        }

#undef HANDLE_TO_INT
#undef HANDLE_TO_UINT
    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_FIXEDPOINT_TO_FIXEDPOINT_HPP
