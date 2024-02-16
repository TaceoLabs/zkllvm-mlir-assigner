
//---------------------------------------------------------------------------//
// Copyright (c) 2023 Nikita Kaskov <nbering@nil.foundation>
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//---------------------------------------------------------------------------//

#ifndef CRYPTO3_ASSIGNER_F_COMPARISON_HPP
#define CRYPTO3_ASSIGNER_F_COMPARISON_HPP

#include <cstdint>
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/components/algebra/fixedpoint/plonk/cmp_extended.hpp>

#include <nil/blueprint/component.hpp>

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>
#include <mlir-assigner/policy/policy_manager.hpp>
#include <mlir-assigner/components/handle_component.hpp>

namespace nil {
    namespace blueprint {
        namespace {
            template<std::uint8_t PreLimbs, std::uint8_t PostLimbs, typename FCmpComponent, typename BlueprintFieldType,
                     typename ArithmetizationParams, typename MlirOp>
            typename FCmpComponent::result_type call_component(
                MlirOp &operation,
                stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
                circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                    &bp,
                assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                    &assignment,
                const common_component_parameters<
                    crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &compParams) {
                using component_type = FCmpComponent;
                using manifest_reader =
                    detail::ManifestReader<FCmpComponent, ArithmetizationParams, PreLimbs, PostLimbs>;
                auto input = PREPARE_BINARY_INPUT(MlirOp);
                const auto p = detail::PolicyManager::get_parameters(manifest_reader::get_witness(0));

                FCmpComponent component(p.witness, manifest_reader::get_constants(),
                                        manifest_reader::get_public_inputs(), PreLimbs, PostLimbs);
                return fill_trace_get_result(component, input, operation, stack, bp, assignment, compParams);
            }
        }    // namespace

        template<std::uint8_t PreLimbs, std::uint8_t PostLimbs, typename BlueprintFieldType,
                 typename ArithmetizationParams>
        void handle_fcmp(
            mlir::arith::CmpFOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            const common_component_parameters<
                crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &compParams) {
            using component_type = components::fix_cmp_extended<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;
            // we compare 64 bits with this configuration
            auto result =
                call_component<PreLimbs, PostLimbs, component_type>(operation, stack, bp, assignment, compParams);
            switch (operation.getPredicate()) {
                case mlir::arith::CmpFPredicate::UGT:
                case mlir::arith::CmpFPredicate::OGT: {
                    stack.push_local(operation.getResult(), result.gt);
                    break;
                }
                case mlir::arith::CmpFPredicate::ULT:
                case mlir::arith::CmpFPredicate::OLT: {
                    stack.push_local(operation.getResult(), result.lt);
                    break;
                }
                case mlir::arith::CmpFPredicate::UGE:
                case mlir::arith::CmpFPredicate::OGE: {
                    stack.push_local(operation.getResult(), result.geq);
                    break;
                }
                case mlir::arith::CmpFPredicate::ULE:
                case mlir::arith::CmpFPredicate::OLE: {
                    stack.push_local(operation.getResult(), result.leq);
                    break;
                }
                case mlir::arith::CmpFPredicate::UNE:
                case mlir::arith::CmpFPredicate::ONE: {
                    stack.push_local(operation.getResult(), result.neq);
                    break;
                }
                case mlir::arith::CmpFPredicate::UEQ:
                case mlir::arith::CmpFPredicate::OEQ: {
                    stack.push_local(operation.getResult(), result.eq);
                    break;
                }
                case mlir::arith::CmpFPredicate::UNO:
                case mlir::arith::CmpFPredicate::ORD:
                case mlir::arith::CmpFPredicate::AlwaysFalse:
                case mlir::arith::CmpFPredicate::AlwaysTrue: {
                    UNREACHABLE("Unsupported fcmp predicate (UNO, ORD, AlwaysFalse, AlwaysTrue)");
                    break;
                }
            }
        }

        template<typename BlueprintFieldType, typename ArithmetizationParams>
        void handle_icmp(
            mlir::arith::CmpIOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            const common_component_parameters<
                crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &compParams) {
            using component_type = components::fix_cmp_extended<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;
            // we compare 64 bits with this configuration
            auto result = call_component<2, 2, component_type>(operation, stack, bp, assignment, compParams);
            switch (operation.getPredicate()) {
                case mlir::arith::CmpIPredicate::sgt:
                    stack.push_local(operation.getResult(), result.gt);
                    break;
                case mlir::arith::CmpIPredicate::slt:
                    stack.push_local(operation.getResult(), result.lt);
                    break;
                case mlir::arith::CmpIPredicate::sge:
                    stack.push_local(operation.getResult(), result.geq);
                    break;
                case mlir::arith::CmpIPredicate::sle:
                    stack.push_local(operation.getResult(), result.leq);
                    break;
                case mlir::arith::CmpIPredicate::ne:
                    stack.push_local(operation.getResult(), result.neq);
                    break;
                case mlir::arith::CmpIPredicate::eq:
                    stack.push_local(operation.getResult(), result.eq);
                    break;
                default:
                    UNREACHABLE("unsupported predicate for cmpi");
            }
        }
    }    // namespace blueprint
}    // namespace nil
#endif    // CRYPTO3_ASSIGNER_F_COMPARISON_HPP
