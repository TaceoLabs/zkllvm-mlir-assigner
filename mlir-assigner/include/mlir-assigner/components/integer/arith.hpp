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
            stack_frame<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &frame,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {

            auto lhs = frame.locals.find(mlir::hash_value(operation.getLhs()));
            auto rhs = frame.locals.find(mlir::hash_value(operation.getRhs()));
            ASSERT(lhs != frame.locals.end());
            ASSERT(rhs != frame.locals.end());

            auto x = lhs->second;
            auto y = rhs->second;

            // TACEO_TODO: check types
            auto result = detail::handle_native_field_addition_component(x, y, bp, assignment, start_row);
            frame.locals[mlir::hash_value(operation.getResult())] = result.output;
        }
          
    }        // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_INTEGER_ARITH_HPP
