
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
#include <mlir-assigner/components/handle_component.hpp>

namespace nil {
    namespace blueprint {
        template<typename BlueprintFieldType, typename ArithmetizationParams>
        void handle_select(
            mlir::arith::SelectOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row, generation_mode gen_mode) {
            auto c = stack.get_local(operation.getCondition());
            auto x = stack.get_local(operation.getTrueValue());
            auto y = stack.get_local(operation.getFalseValue());
            using component_type = components::fix_select<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams>;
            typename component_type::input_type input;
            input.c = stack.get_local(operation.getCondition());
            input.x = stack.get_local(operation.getTrueValue());
            input.y = stack.get_local(operation.getFalseValue());
            const auto p = detail::PolicyManager::get_parameters(manifest_reader::get_witness(0));
            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs());
            fill_trace(component, input, operation, stack, bp, assignment, start_row, gen_mode);
        }
    }    // namespace blueprint
}    // namespace nil
#endif    // CRYPTO3_ASSIGNER_SELECT_HPP
