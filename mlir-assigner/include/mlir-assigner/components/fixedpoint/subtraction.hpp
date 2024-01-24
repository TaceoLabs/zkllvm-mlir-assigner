
#ifndef CRYPTO3_ASSIGNER_FIXEDPOINT_SUBTRACTION_HPP
#define CRYPTO3_ASSIGNER_FIXEDPOINT_SUBTRACTION_HPP

#include <mlir/Dialect/Arith/IR/Arith.h>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <mlir-assigner/components/fields/subtraction.hpp>

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>

namespace nil {
    namespace blueprint {

        template<typename BlueprintFieldType, typename ArithmetizationParams>
        void handle_fixedpoint_subtraction_component(
            mlir::arith::SubFOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {
            auto lhs = stack.get_local(operation.getLhs());
            auto rhs = stack.get_local(operation.getRhs());

            auto result = detail::handle_native_field_subtraction_component(lhs, rhs, bp, assignment, start_row);
            stack.push_local(operation.getResult(), result.output);
        }
    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_FIXEDPOINT_SUBTRACTION_HPP
