#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/Passes/PassBuilder.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Version/Version.hpp"

#include "mlir/Dialect/zkml/ZkMlDialect.h"
#include <Passes/mlir/Transform/ElimCopySignPass.hpp>
#include <Passes/mlir/Transform/PowFToGenericExpPass.hpp>
#include <Passes/mlir/Transform/ElimCopySignPass.hpp>
#include <Passes/mlir/Transform/AddTracingOperationsPass.hpp>
#include <Passes/mlir/Transform/SetPrivateInputPass.hpp>
#include <vector>

#define STDOUT_MARKER "stdout"
#define EMPTY_MARKER "NOT_SET"
#define ALL_PUB_MARKER "ALL_PUBLIC"

enum EmitLevel { zkMLIR, ONNX, MLIR, LLVMIR };

llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::Required);
llvm::cl::opt<std::string> OutputFilename("i", llvm::cl::desc("Specify output filename"),
                                          llvm::cl::value_desc("filename"), llvm::cl::init(STDOUT_MARKER));

llvm::cl::opt<EmitLevel> EmitLevel(llvm::cl::desc("Which lowering level do you want?"),
                                   llvm::cl::values(clEnumVal(ONNX, "Lower to \"ONNX\" dialect."),
                                                    clEnumVal(MLIR, "Lower to \"MLIR-IR\"."),
                                                    clEnumVal(zkMLIR, "Lower to \"zkMLIR-IR\"."),
                                                    clEnumVal(LLVMIR, "Lower to \"LLVM-IR\".")));

llvm::cl::opt<std::string> PrivateInputs("zk", llvm::cl::desc("Specify output filename"), llvm::cl::init("NOT_SET"));

llvm::cl::opt<bool> ZkMlDebugFlag("DEBUG", llvm::cl::desc("turns on debugging log"), llvm::cl::init(false));

bool hasEnding(std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

std::string dirName(llvm::StringRef inputFilename) {
    llvm::SmallVector<char> path(inputFilename.begin(), inputFilename.end());
    llvm::sys::path::remove_filename(path);
    return std::string(path.data(), path.size());
}
int loadOnnxFile(llvm::StringRef inputFilename, mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module,
                 std::string *errorMessage) {
    // we use default options for now from onnx-mlir, lets see if we need
    // something else
    onnx_mlir::ImportOptions options;
    options.useOnnxModelTypes = true;
    options.invokeOnnxVersionConverter = false;
    // TODO check the default value
    options.shapeInformation = onnx_mlir::shapeInformation;
    options.allowSorting = true;
    options.externalDataDir = dirName(inputFilename);
    return onnx_mlir::ImportFrontendModelFile(inputFilename, context, module, errorMessage, options);
}

std::unique_ptr<llvm::Module> lowerToLLVM(llvm::LLVMContext &llvmContext, mlir::OwningOpRef<mlir::ModuleOp> &mlirModule,
                                          int *error_code) {
    std::error_code error;

    // TODO do we want to emit .bc? Or at least make it configureable
    //   mlir::registerLLVMDialectTranslation(*mlirModule->getContext());
    //   std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(*mlirModule, llvmContext);
    //   if (!llvmModule) {
    //       llvm::errs() << "Failed to translate module to LLVMIR.\n";
    //       *error_code = -1;
    //       return nullptr;
    //   }

    mlir::registerBuiltinDialectTranslation(*(mlirModule.get().getContext()));
    mlir::registerLLVMDialectTranslation(*(mlirModule.get().getContext()));
    std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(*mlirModule, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Failed to translate module to LLVMIR.\n";
        exit(-1);
    }
    // Tailor LLVMIR to add features that cannot be done with MLIR LLVMIR.
    // tailorLLVMIR(*llvmModule);
    // Write LLVMIR to a file.
    return llvmModule;
}

void outputModule(mlir::OwningOpRef<mlir::ModuleOp> &module, std::string &outputFilename,
                  int64_t largeElementLimit = -1) {
    mlir::OpPrintingFlags flags;
    if (onnx_mlir::preserveLocations)
        flags.enableDebugInfo();
    if (largeElementLimit >= 0)
        flags.elideLargeElementsAttrs(largeElementLimit);
    // yeah zero means equal....
    if (outputFilename.compare(STDOUT_MARKER) == 0) {
        module->print(llvm::outs(), flags);
    } else {
        std::error_code fileError;
        llvm::raw_fd_ostream fileStream(llvm::StringRef(outputFilename), fileError);
        module->print(fileStream, flags);
        fileStream.close();
    }
}

std::unique_ptr<llvm::Module> translateToLLVMIR(mlir::ModuleOp module, llvm::LLVMContext &llvm_context) {
    mlir::registerLLVMDialectTranslation(*module.getContext());
    std::unique_ptr<llvm::Module> llvm_module = mlir::translateModuleToLLVMIR(module, llvm_context);
    if (!llvm_module) {
        llvm::errs() << "Failed to translate module to LLVMIR.\n";
    }
    return llvm_module;
}

bool parseComaSeperatedList(std::string &str, std::vector<bool> &vect) {
    std::stringstream ss(str);
    while (ss.good()) {
        std::string substr;
        getline(ss, substr, ',');
        if (substr == "1" || substr == "true") {
            vect.push_back(true);
        } else if (substr == "0" || substr == "false") {
            vect.push_back(false);
        } else {
            llvm::errs() << "unexpected str: \"" << substr
                         << "\" in comma seperated list. Expected one of {0,1,false,true}, or just \"ALL_PUBLIC\"\n";
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv);
    std::string inputFilename = InputFilename.c_str();
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    mlir::registerDefaultTimingManagerCLOptions();
    mlir::registerAsmPrinterCLOptions();

    llvm::cl::SetVersionPrinter(onnx_mlir::getVersionPrinter);

    onnx_mlir::removeUnrelatedOptions({&onnx_mlir::OnnxMlirCommonOptions, &onnx_mlir::OnnxMlirOptions});
    onnx_mlir::initCompilerConfig();
    //===========================
    //
    mlir::MLIRContext context;
    mlir::registerOpenMPDialectTranslation(context);
    onnx_mlir::loadDialects(context);
    // does not exist at commit a04f518c1
    // context.appendDialectRegistry(onnx_mlir::registerDialects(onnx_mlir::maccel));
    // context.loadAllAvailableDialects();
    // onnx_mlir::registerDialects(context);
    context.getOrLoadDialect<mlir::zkml::ZkMlDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::string errorMessage;
    if (int rc = loadOnnxFile(llvm::StringRef(inputFilename), context, module, &errorMessage)) {
        llvm::errs() << "Cannot load .onnx file:\n";
        llvm::errs() << errorMessage << "\n";
        return rc;
    }
    std::string outputFilename = OutputFilename.c_str();
    onnx_mlir::setupModule(module, context, outputFilename);
    bool EmitMLIR = EmitLevel::zkMLIR == EmitLevel || EmitLevel::MLIR == EmitLevel;
    onnx_mlir::configurePasses();
    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    if (ZkMlDebugFlag) {
        pm.addPass(mlir::zk_ml::createAddTracingOperationsPass());
    }
    if (EmitLevel::ONNX == EmitLevel) {
        onnx_mlir::addPasses(module, pm, onnx_mlir::EmissionTargetType::EmitONNXIR, outputFilename);
    } else {
        onnx_mlir::addPasses(module, pm, onnx_mlir::EmissionTargetType::EmitMLIR, outputFilename,
                             EmitLevel == EmitLevel::zkMLIR);
        if (EmitLevel::zkMLIR == EmitLevel) {
            // parse setting for public/private
            if (EMPTY_MARKER == PrivateInputs) {
                llvm::errs() << "When compiling zkMLIR you have to specify which inputs are public/private by passing "
                                "a comma seperated list to --zk\n";
                return -2;
            }
            pm.addPass(mlir::zk_ml::createElimCopySignPass());
            pm.addPass(mlir::zk_ml::createPowFToGenericExpPass());
            if (ALL_PUB_MARKER != PrivateInputs) {
                std::vector<bool> pubPrivMarkers;
                if (!parseComaSeperatedList(PrivateInputs, pubPrivMarkers)) {
                    return -2;
                }
                pm.addPass(mlir::zk_ml::createSetPrivateInputPass(pubPrivMarkers));
            }
        }
        if (!EmitMLIR) {
            // third parameter here is optional in onnx-mlir. Maybe we should do that
            // too?
            onnx_mlir::addKrnlToLLVMPasses(pm, outputFilename, true);
        }
    }

    (void)mlir::applyPassManagerCLOptions(pm);
    mlir::applyDefaultTimingPassManagerCLOptions(pm);
    if (mlir::failed(pm.run(*module))) {
        llvm::errs() << "Passmanager failed to run!\n";
        return -1;
    }

    if (EmitMLIR || EmitLevel::ONNX == EmitLevel) {
        outputModule(module, outputFilename);
        return 0;
    } else {
        int error_code;
        llvm::LLVMContext llvmContext;
        std::unique_ptr<llvm::Module> llvm_module = lowerToLLVM(llvmContext, module, &error_code);
        if (!llvm_module)
            return error_code;

        if (outputFilename.compare(STDOUT_MARKER) == 0) {
            llvm_module->print(llvm::outs(), nullptr);
        } else {
            std::error_code fileError;
            llvm::raw_fd_ostream fileStream(llvm::StringRef(outputFilename), fileError);
            llvm_module->print(fileStream, nullptr);
            fileStream.close();
        }
    }

    return 0;
}
