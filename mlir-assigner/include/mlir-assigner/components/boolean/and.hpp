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

#ifndef CRYPTO3_ASSIGNER_AND_HPP
#define CRYPTO3_ASSIGNER_AND_HPP

#include <mlir/Dialect/Arith/IR/Arith.h>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/components/algebra/fields/plonk/non_native/lookup_logic_ops.hpp>
#include <nil/blueprint/component.hpp>
#include <nil/blueprint/basic_non_native_policy.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/lookup_tables/tester.hpp>    // TODO: check if there is a new mechanism for this in nil upstream

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>
#include <mlir-assigner/components/handle_component.hpp>

namespace nil {
    namespace blueprint {
        template<typename BlueprintFieldType, typename ArithmetizationParams>
        void handle_logic_and(
            mlir::arith::AndIOp &operation,
            stack_frame<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &frame,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {
            using component_type = components::lookup_logic_and<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>;

            auto input = PREPARE_BINARY_INPUT(mlir::arith::AndIOp);
            const auto p = detail::PolicyManager::get_parameters(
                detail::ManifestReader<component_type, ArithmetizationParams>::get_witness(0));

            component_type component(p.witness);
            fill_trace(component, input, operation, frame, bp, assignment, start_row);
        }

        template<typename BlueprintFieldType, typename ArithmetizationParams>
        void handle_logic_or(
            mlir::arith::OrIOp &operation,
            stack_frame<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &frame,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {
            //FIXME logic_or is commented out. As soon as it is enabled, remove add the liens above and it SHOULD work
            UNREACHABLE("LogicOR not enabled in blueprint");
            // using component_type = components::lookup_logic_or<
            //     crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>;
            //
            // auto input = PREPARE_BINARY_INPUT(mlir::arith::OrIOp);
            // const auto p = detail::PolicyManager::get_parameters(
            //     detail::ManifestReader<component_type, ArithmetizationParams>::get_witness(0));
            //
            // component_type component(p.witness);
            // fill_trace(component, input, operation, frame, bp, assignment, start_row);
        }

        template<typename BlueprintFieldType, typename ArithmetizationParams>
        void handle_logic_xor(
            mlir::arith::XOrIOp &operation,
            stack_frame<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &frame,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {
            using component_type = components::lookup_logic_xor<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>;

            auto input = PREPARE_BINARY_INPUT(mlir::arith::XOrIOp);
            const auto p = detail::PolicyManager::get_parameters(
                detail::ManifestReader<component_type, ArithmetizationParams>::get_witness(0));

            component_type component(p.witness);
            fill_trace(component, input, operation, frame, bp, assignment, start_row);
        }

    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_LOGIC_OPS_HPP
