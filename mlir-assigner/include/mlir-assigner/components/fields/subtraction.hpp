//---------------------------------------------------------------------------//
// Copyright (c) 2022 Mikhail Komarov <nemo@nil.foundation>
// Copyright (c) 2022 Nikita Kaskov <nbering@nil.foundation>
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

#ifndef CRYPTO3_ASSIGNER_FIELD_SUBTRACTION_HPP
#define CRYPTO3_ASSIGNER_FIELD_SUBTRACTION_HPP

#include <nil/crypto3/algebra/curves/bls12.hpp>
#include <nil/crypto3/algebra/curves/ed25519.hpp>
#include <nil/crypto3/algebra/curves/curve25519.hpp>
#include <nil/crypto3/algebra/curves/pallas.hpp>
#include <nil/crypto3/algebra/curves/vesta.hpp>

#include <nil/crypto3/zk/snark/arithmetization/plonk/constraint_system.hpp>

#include <nil/blueprint/component.hpp>
#include <nil/blueprint/basic_non_native_policy.hpp>
#include <nil/blueprint/components/algebra/fields/plonk/subtraction.hpp>
#include <nil/blueprint/components/algebra/fields/plonk/non_native/subtraction.hpp>

#include <mlir-assigner/policy/policy_manager.hpp>

namespace nil {
    namespace blueprint {
        namespace detail {

            template<typename BlueprintFieldType, typename ArithmetizationParams>
            typename components::subtraction<
                crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>::result_type
                handle_native_field_subtraction_component(
                    crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>
                        x,
                    crypto3::zk::snark::plonk_variable<typename BlueprintFieldType::value_type>
                        y,
                    circuit_proxy<
                        crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>> &bp,
                    assignment_proxy<crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType,
                                                                                 ArithmetizationParams>> &assignment,
                    std::uint32_t start_row) {

                using component_type = components::subtraction<
                    crypto3::zk::snark::plonk_constraint_system<BlueprintFieldType, ArithmetizationParams>,
                    BlueprintFieldType, basic_non_native_policy<BlueprintFieldType>>;
                const auto p = PolicyManager::get_parameters(
                    ManifestReader<component_type, ArithmetizationParams>::get_witness(0));
                component_type component_instance(
                    p.witness,
                    ManifestReader<component_type, ArithmetizationParams>::get_constants(),
                    ManifestReader<component_type, ArithmetizationParams>::get_public_inputs());

                components::generate_circuit(component_instance, bp, assignment, {x, y}, start_row);
                return components::generate_assignments(component_instance, assignment, {x, y}, start_row);
            }

        }    // namespace detail

    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_FIELD_SUBTRACTION_HPP
