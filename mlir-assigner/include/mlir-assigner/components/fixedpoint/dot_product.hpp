#ifndef CRYPTO3_ASSIGNER_FIXEDPOINT_DOT_PRODUCT_HPP
#define CRYPTO3_ASSIGNER_FIXEDPOINT_DOT_PRODUCT_HPP

#include "mlir-assigner/memory/memref.hpp"
#include "mlir/Dialect/zkml/IR/DotProduct.h"
#include <cstdint>
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/component.hpp>
#include <nil/blueprint/basic_non_native_policy.hpp>
#include <nil/blueprint/components/algebra/fixedpoint/plonk/dot_rescale_2_gates.hpp>

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
            const common_component_parameters<
                crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>> &compParams) {

            mlir::Value lhs = operation.getLhs();
            mlir::Value rhs = operation.getRhs();
            assert(lhs.getType() == rhs.getType() && "memrefs must be same type for DotProduct");
            mlir::MemRefType MemRefType = mlir::cast<mlir::MemRefType>(lhs.getType());
            assert(MemRefType.getShape().size() == 1 && "DotProduct must have tensors of rank 1");

            auto &x = stack.get_memref(operation.getLhs());
            auto &y = stack.get_memref(operation.getRhs());
            auto dims = x.getDims();
            ASSERT(dims.size() == 1 && "must be one-dim for dot product");
            if (MemRefType.getElementType().isa<mlir::IntegerType>()) {
                using ComponentType = components::dot_2_gates<
                    crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                    BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;
                using manifest_reader = detail::ManifestReader<ComponentType, ArithmetizationParams>;
                const auto p = detail::PolicyManager::get_parameters(manifest_reader::get_witness(0, dims.front()));
                ComponentType component(p.witness, manifest_reader::get_constants(),
                                        manifest_reader::get_public_inputs(), dims.front());
                typename ComponentType::input_type input = {x.getData(), y.getData(), zero_var};
                fill_trace(component, input, operation, stack, bp, assignment, compParams);
            } else if (MemRefType.getElementType().isa<mlir::FloatType>()) {
                using ComponentType = components::fix_dot_rescale_2_gates<
                    crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                    BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;
                using manifest_reader = detail::ManifestReader<ComponentType, ArithmetizationParams, PostLimbs>;
                const auto p =
                    detail::PolicyManager::get_parameters(manifest_reader::get_witness(0, dims.front(), PostLimbs));
                ComponentType component(p.witness, manifest_reader::get_constants(),
                                        manifest_reader::get_public_inputs(), dims.front(), PostLimbs);
                typename ComponentType::input_type input = {x.getData(), y.getData(), zero_var};
                fill_trace(component, input, operation, stack, bp, assignment, compParams);
            } else {
                UNREACHABLE("Unsupported type for dot-product. Only floats and ints supported");
            }
        }

    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_FIXEDPOINT_DOT_PRODUCT_HPP
