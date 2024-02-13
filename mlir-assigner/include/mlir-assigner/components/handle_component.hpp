
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
#include <nil/blueprint/components/algebra/fixedpoint/plonk/sign_abs.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/mul_rescale.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/ceil.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/floor.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/to_fixedpoint.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/to_int.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/sin.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/sinh.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/asin.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/asinh.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/cos.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/cosh.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/acos.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/acosh.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/tan.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/tanh.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/atan.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/atanh.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/exp_ranged.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/argmin.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/argmax.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/neg.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/sqrt.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/cmp_set.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/gather_acc.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/erf.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/div.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/rem.hpp>
#include <nil/blueprint/components/algebra/fields/plonk/bitwise_and.hpp>
#include <nil/blueprint/components/algebra/fields/plonk/bitwise_or.hpp>
#include <nil/blueprint/components/algebra/fields/plonk/bitwise_xor.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/dot_rescale_2_gates.hpp>
#include <nil/blueprint/components/algebra/fields/plonk/non_native/logic_ops.hpp>
#include <nil/blueprint/components/algebra/fields/plonk/addition.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/select.hpp>
#include <nil/blueprint/components/algebra/fields/plonk/subtraction.hpp>
#include <nil/blueprint/components/algebra/fields/plonk/multiplication.hpp>

#define PREPARE_UNARY_INPUT(OP)                                                                                        \
    prepare_unary_operation_input<BlueprintFieldType, ArithmetizationParams, OP, typename component_type::input_type>( \
        operation, stack, bp, assignment);
#define PREPARE_BINARY_INPUT(OP)                          \
    prepare_binary_operation_input<BlueprintFieldType,    \
                                   ArithmetizationParams, \
                                   OP,                    \
                                   typename component_type::input_type>(operation, stack, bp, assignment);

namespace nil {
    namespace blueprint {
        template<typename BlueprintFieldType, typename ArithmetizationParams, typename UnaryOp, typename input_type>
        input_type prepare_unary_operation_input(
            UnaryOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment) {
            assert(operation->getNumOperands() == 1 && "unary operand must have only one operand");
            input_type instance_input;
            instance_input.x = stack.get_local(operation->getOperand(0));
            return instance_input;
        }
        template<typename BlueprintFieldType, typename ArithmetizationParams, typename BinOp, typename input_type>
        input_type prepare_binary_operation_input(
            BinOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment) {
            input_type instance_input;
            instance_input.x = stack.get_local(operation.getLhs());
            instance_input.y = stack.get_local(operation.getRhs());
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
        typename component_type::result_type fill_trace_get_result(
            component_type &component,
            typename component_type::input_type &input,
            Op &mlir_op,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
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
            return components::generate_assignments(component, assignment, input, start_row);
        }
        template<typename BlueprintFieldType, typename ArithmetizationParams, typename component_type, typename Op>
        void fill_trace(
            component_type &component,
            typename component_type::input_type &input,
            Op &mlir_op,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {
            auto result = fill_trace_get_result(component, input, mlir_op, stack, bp, assignment, start_row);
            stack.push_local(mlir_op.getResult(), result.output);
        }
    }    // namespace blueprint
}    // namespace nil
#endif    // CRYPTO3_ASSIGNER_HANDLE_COMPONENT_HPP
