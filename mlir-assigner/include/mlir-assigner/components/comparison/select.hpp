
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

#ifndef CRYPTO3_ASSIGNER_SELECT_HPP
#define CRYPTO3_ASSIGNER_SELECT_HPP

#include <mlir/Dialect/Arith/IR/Arith.h>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/components/algebra/fixedpoint/plonk/select.hpp>

#include <nil/blueprint/component.hpp>

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>
#include <mlir-assigner/policy/policy_manager.hpp>

namespace nil {
    namespace blueprint {
        namespace detail {

            template<typename BlueprintFieldType, typename ArithmetizationParams>
            typename components::fix_select<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>::result_type
                handle_select_component(
                    const typename crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type> &c,
                    const typename crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type> &x,
                    const typename crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type> &y,
                    circuit_proxy<
                        crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
                    assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType,
                                                                                 ArithmetizationParams>> &assignment,
                    std::uint32_t start_row) {
                using non_native_policy_type = basic_non_native_policy<BlueprintFieldType>;
                using var = crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>;
                using component_type = components::fix_select<
                    crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                    BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;
                const auto params = PolicyManager::get_parameters(
                    ManifestReader<component_type, ArithmetizationParams>::get_witness(0));
                component_type component_instance(
                    params.witness,
                    ManifestReader<component_type, ArithmetizationParams>::get_constants(),
                    ManifestReader<component_type, ArithmetizationParams>::get_public_inputs());

                components::generate_circuit(component_instance, bp, assignment, {c, x, y}, start_row);
                return components::generate_assignments(component_instance, assignment, {c, x, y}, start_row);
            }
        }    // namespace detail
        template<typename BlueprintFieldType, typename ArithmetizationParams>
        void handle_select_component(
            mlir::arith::SelectOp &operation,
            stack_frame<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &frame,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {

            auto false_value = frame.locals.find(mlir::hash_value(operation.getFalseValue()));
            auto true_value = frame.locals.find(mlir::hash_value(operation.getTrueValue()));
            auto condition = frame.locals.find(mlir::hash_value(operation.getCondition()));
            ASSERT(false_value != frame.locals.end());
            ASSERT(true_value != frame.locals.end());
            ASSERT(condition != frame.locals.end());

            auto c = condition->second;
            auto x = true_value->second;
            auto y = false_value->second;

            // TACEO_TODO: check types

            auto result = detail::handle_select_component(c, x, y, bp, assignment, start_row);
            frame.locals[mlir::hash_value(operation.getResult())] = result.output;
        }
    }    // namespace blueprint
}    // namespace nil
#endif    // CRYPTO3_ASSIGNER_SELECT_HPP
