#ifndef CRYPTO3_ASSIGNER_INTEGER_ARITH_HPP
#define CRYPTO3_ASSIGNER_INTEGER_ARITH_HPP

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/component.hpp>
#include <nil/blueprint/basic_non_native_policy.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/neg.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/lookup_tables/tester.hpp>

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/policy/policy_manager.hpp>

namespace nil {
    namespace blueprint {

        template<typename BlueprintFieldType, typename ArithmetizationParams>
        void handle_integer_addition_component(
            mlir::arith::AddIOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {

            auto lhs = stack.get_local(operation.getLhs());
            auto rhs = stack.get_local(operation.getRhs());
            // TACEO_TODO: check types
            auto result = detail::handle_native_field_addition_component(lhs, rhs, bp, assignment, start_row);
            stack.push_local(operation.getResult(), result.output);
        }
          
    }        // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_INTEGER_ARITH_HPP
