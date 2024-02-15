#ifndef CRYPTO3_ASSIGNER_FIXEDPOINT_GATHER_HPP
#define CRYPTO3_ASSIGNER_FIXEDPOINT_GATHER_HPP

#include "mlir/Dialect/zkml/IR/Gather.h"
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/component.hpp>
#include <nil/blueprint/basic_non_native_policy.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/lookup_tables/tester.hpp>    // TODO: check if there is a new mechanism for this in nil upstream

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>
#include <mlir-assigner/components/handle_component.hpp>

namespace nil {
    namespace blueprint {

        template<typename BlueprintFieldType, typename ArithmetizationParams>
        void handle_gather(
            mlir::zkml::GatherOp &operation,
            stack<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &stack,
            circuit_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
            assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>>
                &assignment,
            crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type> &nextIndex,
            const common_component_parameters<crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &compParams) {
            using component_type = components::fix_gather_acc<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;

            using input_type = typename component_type::input_type;
            input_type input;
            input.prev_acc = stack.get_local(operation.getPrevAcc());
            input.data = stack.get_local(operation.getData());
            input.index_a = stack.get_local(operation.getGatherIndex());

            using manifest_reader = detail::ManifestReader<component_type, ArithmetizationParams>;
            const auto p = detail::PolicyManager::get_parameters(
                detail::ManifestReader<component_type, ArithmetizationParams>::get_witness(0));
            component_type component(p.witness,
                                     manifest_reader::get_constants(),
                                     manifest_reader::get_public_inputs(),
                                     var_value(assignment, nextIndex));

            fill_trace(component, input, operation, stack, bp, assignment, compParams);
        }

    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_FIXEDPOINT_GATHER_HPP
