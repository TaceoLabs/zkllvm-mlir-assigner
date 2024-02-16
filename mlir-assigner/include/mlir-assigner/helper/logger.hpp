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

#ifndef CRYPTO3_ASSIGNER_NIL_BLUEPRINT_LOGGER_HPP
#define CRYPTO3_ASSIGNER_NIL_BLUEPRINT_LOGGER_HPP

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include <string_view>
#include <llvm/IR/Instructions.h>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/format.hpp>

namespace nil {
    namespace blueprint {

        enum print_format { no_print, dec, hex };

        class logger {
        public:
            logger(boost::log::trivial::severity_level lvl = boost::log::trivial::info) : lvl(lvl) {
                boost::log::core::get()->set_filter(boost::log::trivial::severity >= lvl);
            }

            void set_level(boost::log::trivial::severity_level lvl) {
                this->lvl = lvl;
                boost::log::core::get()->set_filter(boost::log::trivial::severity >= lvl);
            }

            template<typename... Args>
            void trace(const char *fmt, const Args &...args) {
                // https://stackoverflow.com/questions/25859672/boostformat-with-variadic-template-arguments
                BOOST_LOG_TRIVIAL(trace) << boost::str((boost::format(fmt) % ... % args));
            }

            template<typename... Args>
            void debug(const char *fmt, const Args &...args) {
                // https://stackoverflow.com/questions/25859672/boostformat-with-variadic-template-arguments
                BOOST_LOG_TRIVIAL(debug) << boost::str((boost::format(fmt) % ... % args));
            }

            template<typename... Args>
            void info(const char *fmt, const Args &...args) {
                // https://stackoverflow.com/questions/25859672/boostformat-with-variadic-template-arguments
                BOOST_LOG_TRIVIAL(info) << boost::str((boost::format(fmt) % ... % args));
            }

            template<typename... Args>
            void error(const char *fmt, const Args &...args) {
                // https://stackoverflow.com/questions/25859672/boostformat-with-variadic-template-arguments
                BOOST_LOG_TRIVIAL(error) << boost::str((boost::format(fmt) % ... % args));
            }

            void debug(boost::basic_format<char> formated_debug_message) {
                BOOST_LOG_TRIVIAL(debug) << boost::str(formated_debug_message);
            }

            void debug(std::string_view debug_message) {
                BOOST_LOG_TRIVIAL(debug) << debug_message;
            }

            void log_value(const mlir::Value &Value) {
                if (lvl > boost::log::trivial::debug) {
                    return;
                }
                std::string str;
                llvm::raw_string_ostream ss(str);
                ss << Value;
                BOOST_LOG_TRIVIAL(debug) << str;
            }

            void log_affine_map(const mlir::AffineMap &AffineMap) {
                if (lvl > boost::log::trivial::debug) {
                    return;
                }
                std::string str;
                llvm::raw_string_ostream ss(str);
                ss << AffineMap;
                BOOST_LOG_TRIVIAL(debug) << str;
            }

            void log_attribute(const mlir::Attribute &Attr) {
                if (lvl > boost::log::trivial::debug) {
                    return;
                }
                std::string str;
                llvm::raw_string_ostream ss(str);
                ss << Attr;
                BOOST_LOG_TRIVIAL(debug) << str;
            }

            template<typename T>
            void operator<<(const T &Val) {
                if (lvl > boost::log::trivial::debug) {
                    return;
                }
                std::string str;
                llvm::raw_string_ostream ss(str);
                ss << Val;
                BOOST_LOG_TRIVIAL(debug) << str;
            }

            template<typename T>
            void operator<<(const llvm::ArrayRef<T> &Val) {
                if (lvl > boost::log::trivial::debug) {
                    return;
                }
                std::string str;
                llvm::raw_string_ostream ss(str);
                ss << "[";
                for (auto V : Val) {
                    ss << V << ", ";
                }
                ss << "]";
                BOOST_LOG_TRIVIAL(debug) << str;
            }

        private:
            boost::log::trivial::severity_level lvl;
        };
    }    // namespace blueprint
}    // namespace nil

#endif    // CRYPTO3_ASSIGNER_NIL_BLUEPRINT_LOGGER_HPP
