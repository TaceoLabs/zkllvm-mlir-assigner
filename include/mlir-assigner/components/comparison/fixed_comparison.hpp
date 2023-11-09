
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

#ifndef CRYPTO3_ASSIGNER_F_COMPARISON_HPP
#define CRYPTO3_ASSIGNER_F_COMPARISON_HPP

#include <mlir/Dialect/Arith/IR/Arith.h>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/components/algebra/fixedpoint/plonk/cmp_extended.hpp>

#include <nil/blueprint/component.hpp>

#include <mlir-assigner/helper/asserts.hpp>
#include <mlir-assigner/memory/stack_frame.hpp>
#include <mlir-assigner/policy/policy_manager.hpp>

namespace nil {
namespace blueprint {
namespace detail {

template <typename BlueprintFieldType, typename ArithmetizationParams>
typename crypto3::zk::snark::plonk_variable<
    typename BlueprintFieldType::value_type>
handle_f_comparison_component(
    mlir::arith::CmpFPredicate p,
    const typename crypto3::zk::snark::plonk_variable<
        typename BlueprintFieldType::value_type> &x,
    const typename crypto3::zk::snark::plonk_variable<
        typename BlueprintFieldType::value_type> &y,
    circuit<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &bp,
    assignment<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &assignment,
    std::uint32_t start_row) {
  using non_native_policy_type = basic_non_native_policy<BlueprintFieldType>;
  using var = crypto3::zk::snark::plonk_variable<
      typename BlueprintFieldType::value_type>;
  using component_type = components::fix_cmp_extended<
      crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType,
                                                  ArithmetizationParams>,
      BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;
  const auto params = PolicyManager::get_parameters(
      ManifestReader<component_type, ArithmetizationParams, 1, 1>::get_witness(
          0));
  component_type component_instance(
      params.witness,
      ManifestReader<component_type, ArithmetizationParams, 1,
                     1>::get_constants(),
      ManifestReader<component_type, ArithmetizationParams, 1,
                     1>::get_public_inputs(),
      1, 1);

  components::generate_circuit(component_instance, bp, assignment, {x, y},
                               start_row);
  auto cmp_result = components::generate_assignments(
      component_instance, assignment, {x, y}, start_row);

  switch (p) {
  case mlir::arith::CmpFPredicate::UGT:
  case mlir::arith::CmpFPredicate::OGT: {
    return cmp_result.gt;
  }
  case mlir::arith::CmpFPredicate::ULT:
  case mlir::arith::CmpFPredicate::OLT: {
    return cmp_result.lt;
  }
  case mlir::arith::CmpFPredicate::UGE:
  case mlir::arith::CmpFPredicate::OGE: {
    return cmp_result.geq;
  }
  case mlir::arith::CmpFPredicate::ULE:
  case mlir::arith::CmpFPredicate::OLE: {
    return cmp_result.leq;
  }
  case mlir::arith::CmpFPredicate::UNE:
  case mlir::arith::CmpFPredicate::ONE: {
    return cmp_result.neq;
  }
  case mlir::arith::CmpFPredicate::UEQ:
  case mlir::arith::CmpFPredicate::OEQ: {
    return cmp_result.eq;
  }
  case mlir::arith::CmpFPredicate::UNO:
  case mlir::arith::CmpFPredicate::ORD:
  case mlir::arith::CmpFPredicate::AlwaysFalse:
  case mlir::arith::CmpFPredicate::AlwaysTrue: {
    UNREACHABLE("TACEO_TODO implement fcmp");
    break;
  }
  default:
    UNREACHABLE("Unsupported fcmp predicate");
    break;
  }
}
} // namespace detail
template <typename BlueprintFieldType, typename ArithmetizationParams>
void handle_fixedpoint_comparison_component(
    mlir::arith::CmpFOp &operation,
    stack_frame<crypto3::zk::snark::plonk_variable<
        typename BlueprintFieldType::value_type>> &frame,
    circuit<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &bp,
    assignment<crypto3::zk::snark::plonk_constraint_system<
        BlueprintFieldType, ArithmetizationParams>> &assignment,
    std::uint32_t start_row) {

  auto lhs = frame.locals.find(mlir::hash_value(operation.getLhs()));
  ASSERT(lhs != frame.locals.end());
  auto rhs = frame.locals.find(mlir::hash_value(operation.getRhs()));
  ASSERT(rhs != frame.locals.end());

  auto pred = operation.getPredicate();

  auto x = lhs->second;
  auto y = rhs->second;

  std::cout << var_value(assignment, x) << " " << var_value(assignment, y)
            << "\n";

  // TACEO_TODO: check types

  auto result = detail::handle_f_comparison_component(pred, x, y, bp,
                                                      assignment, start_row);
  frame.locals[mlir::hash_value(operation.getResult())] = result;
}
} // namespace blueprint
} // namespace nil
#endif // CRYPTO3_ASSIGNER_F_COMPARISON_HPP
