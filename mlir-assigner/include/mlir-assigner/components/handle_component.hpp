
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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//---------------------------------------------------------------------------//

#ifndef CRYPTO3_ASSIGNER_HANDLE_COMPONENT_HPP
#define CRYPTO3_ASSIGNER_HANDLE_COMPONENT_HPP

#include <functional>
#include <mlir-assigner/memory/stack_frame.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/to_fixedpoint.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/sin.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/cos.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/argmin.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/argmax.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/sqrt.hpp>
#include <nil/blueprint/components/algebra/fields/plonk/non_native/lookup_logic_ops.hpp>
#include <nil/blueprint/components/algebra/fields/plonk/logic_or_flag.hpp>

#define PREPARE_UNARY_INPUT(OP)                                                             \
    prepare_unary_operation_input<BlueprintFieldType, ArithmetizationParams, OP,     \
                                   typename component_type::input_type>(operation, frame, bp, assignment);
#define PREPARE_BINARY_INPUT(OP)                                                             \
    prepare_binary_operation_input<BlueprintFieldType, ArithmetizationParams, OP,     \
                                   typename component_type::input_type>(operation, frame, bp, assignment);


namespace nil {
    namespace blueprint {
        template<typename BlueprintFieldType, typename ArithmetizationParams, typename UnaryOp, typename input_type>
        input_type prepare_unary_operation_input(
            UnaryOp &operation,
            stack_frame<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &frame,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment) {

            assert(operation->getNumOperands() == 1 && "unary operand must have only one operand");
            auto operand = frame.locals.find(mlir::hash_value(operation->getOperand(0)));
            ASSERT(operand != frame.locals.end());

            input_type instance_input;
            instance_input.x = operand->second;
            return instance_input;
        }
        template<typename BlueprintFieldType, typename ArithmetizationParams, typename BinOp, typename input_type>
        input_type prepare_binary_operation_input(
            BinOp &operation,
            stack_frame<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &frame,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment) {

            auto lhs = frame.locals.find(mlir::hash_value(operation.getLhs()));
            ASSERT(lhs != frame.locals.end());
            auto rhs = frame.locals.find(mlir::hash_value(operation.getRhs()));
            ASSERT(rhs != frame.locals.end());

            auto x = lhs->second;
            auto y = rhs->second;

            input_type instance_input;
            instance_input.input[0] = x;
            instance_input.input[1] = y;
            return instance_input;
        }

        template<typename BlueprintFieldType, typename ArithmetizationParams, typename ComponentType>
        void handle_component_input(
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            typename ComponentType::input_type &instance_input) {

            using var = crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>;

            std::vector<std::reference_wrapper<var>> input = instance_input.all_vars();
            const auto &used_rows = assignment.get_used_rows();

            for (auto &v : input) {
                bool found = (used_rows.find(v.get().rotation) != used_rows.end());
                if (!found &&
                    (v.get().type == var::column_type::witness || v.get().type == var::column_type::constant)) {
                    const auto new_v = save_shared_var(assignment, v);
                    v.get().index = new_v.index;
                    v.get().rotation = new_v.rotation;
                    v.get().relative = new_v.relative;
                    v.get().type = new_v.type;
                }
            }
        }

        template<typename BlueprintFieldType, typename ArithmetizationParams, typename component_type, typename Op>
        void fill_trace(
            component_type &component,
            typename component_type::input_type &input,
            Op &mlir_op,
            stack_frame<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &frame,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {

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

            handle_component_input<BlueprintFieldType, ArithmetizationParams, component_type>(assignment, input);

            components::generate_circuit(component, bp, assignment, input, start_row);
            auto result = components::generate_assignments(component, assignment, input, start_row);
            frame.locals[mlir::hash_value(mlir_op.getResult())] = result.output;
        }
    }    // namespace blueprint
}    // namespace nil
#endif    // CRYPTO3_ASSIGNER_HANDLE_COMPONENT_HPP
