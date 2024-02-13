#ifndef CRYPTO3_ASSIGNER_FIXEDPOINT_DOT_PRODUCT_HPP
#define CRYPTO3_ASSIGNER_FIXEDPOINT_DOT_PRODUCT_HPP

#include "mlir/Dialect/zkml/IR/DotProduct.h"
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/component.hpp>
#include <nil/blueprint/basic_non_native_policy.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/dot_rescale_2_gates.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/lookup_tables/tester.hpp>    // TODO: check if there is a new mechanism for this in nil upstream

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>
#include <mlir-assigner/components/handle_component.hpp>

namespace nil {
    namespace blueprint {

        template<std::uint8_t PreLimbs, std::uint8_t PostLimbs, typename BlueprintFieldType,
                 typename ArithmetizationParams>
        void handle_dot_product(
            mlir::zkml::DotProductOp &operation,
            crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type> &zero_var,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            std::uint32_t start_row) {
            using component_type = components::fix_dot_rescale_2_gates<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams, PreLimbs, PostLimbs>;

            auto &x = stack.get_memref(operation.getLhs());
            auto &y = stack.get_memref(operation.getRhs());
            auto dims = x.getDims();
            ASSERT(dims.size() == 1 && "must be one-dim for dot product");
            const auto p =
                detail::PolicyManager::get_parameters(manifest_reader::get_witness(0, dims.front(), PostLimbs));
            component_type component_instance(p.witness, manifest_reader::get_constants(),
                                              manifest_reader::get_public_inputs(), dims.front(), PostLimbs);

            component_type component(p.witness, manifest_reader::get_constants(), manifest_reader::get_public_inputs(),
                                     dims.front(), PostLimbs);
            typename component_type::input_type input = {x.getData(), y.getData(), zero_var};

            fill_trace(component, input, operation, stack, bp, assignment, start_row);
        }
    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_FIXEDPOINT_DOT_PRODUCT_HPP
