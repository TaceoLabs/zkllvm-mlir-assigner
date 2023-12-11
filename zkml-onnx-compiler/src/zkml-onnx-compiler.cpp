#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Passes/PassBuilder.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Compiler/CompilerUtils.hpp"

#include "mlir/Dialect/zkml/ZkMlDialect.h"
#include "Passes/mlir/Transform/ElimCopySignPass.h"

#define STDOUT_MARKER "stdout"

enum EmitLevel { zkMLIR, ONNX, MLIR, LLVMIR };

llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::Required);
llvm::cl::opt<std::string> OutputFilename("i", llvm::cl::desc("Specify output filename"),
                                          llvm::cl::value_desc("filename"), llvm::cl::init(STDOUT_MARKER));

llvm::cl::opt<EmitLevel> EmitLevel(llvm::cl::desc("Which lowering level do you want?"),
                                   llvm::cl::values(clEnumVal(ONNX, "Lower to \"ONNX\" dialect."),
                                                    clEnumVal(MLIR, "Lower to \"MLIR-IR\"."),
                                                    clEnumVal(zkMLIR, "Lower to \"zkMLIR-IR\"."),
                                                    clEnumVal(LLVMIR, "Lower to \"LLVM-IR\".")));

llvm::cl::opt<bool> ZkMlDebugFlag("DEBUG", llvm::cl::desc("turns on debugging log"), llvm::cl::init(false));

bool hasEnding(std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

std::string dirName(StringRef inputFilename) {
    llvm::SmallVector<char> path(inputFilename.begin(), inputFilename.end());
    llvm::sys::path::remove_filename(path);
    return std::string(path.data(), path.size());
}
int loadOnnxFile(StringRef inputFilename, mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module,
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
    // does not exist at commit a04f518c1
    // options.functionsToDecompose.insert(options.functionsToDecompose.end(),
    //                                  onnx_mlir::functionsToDecompose.begin(),
    //                                  onnx_mlir::functionsToDecompose.end());
    return onnx_mlir::ImportFrontendModelFile(inputFilename, context, module, errorMessage);
    // return onnx_mlir::ImportFrontendModelFile(inputFilename, context, module,
    // errorMessage, options);
}

std::unique_ptr<llvm::Module> lowerToLLVM(llvm::LLVMContext &llvmContext, mlir::OwningOpRef<mlir::ModuleOp> &mlirModule,
                                          int *error_code) {
    std::error_code error;

    // TODO do we want to emit .bc? Or at least make it configureable
    mlir::registerLLVMDialectTranslation(*mlirModule->getContext());
    std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(*mlirModule, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Failed to translate module to LLVMIR.\n";
        *error_code = -1;
        return nullptr;
    }
    return llvmModule;
}

void runZkMlPasses(std::unique_ptr<llvm::Module> &llvm_module, llvm::OptimizationLevel OptimizationLevel) {
    // create all analyses
    // llvm::ModuleAnalysisManager MAM;
    // llvm::LoopAnalysisManager LAM;
    // llvm::FunctionAnalysisManager FAM;
    // llvm::CGSCCAnalysisManager CGAM;

    // llvm::PassBuilder PB;
    //// Register all the basic analyses with the managers.
    // PB.registerModuleAnalyses(MAM);
    // PB.registerCGSCCAnalyses(CGAM);
    // PB.registerFunctionAnalyses(FAM);
    // PB.registerLoopAnalyses(LAM);
    // PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // This one corresponds to a typical -O2 optimization pipeline.
    // llvm::ModulePassManager MPM =
    //     PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O1);
    // // we got default Module Passmanager for corresponding OptimizationLevel
    // // now add our passes
    // llvm::FunctionPassManager FPM;
    // FPM.addPass(zk_ml::AddCircuitFnAttrPass());
    // MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
    // MPM.run(*llvm_module, MAM);
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

int main(int argc, char **argv) {

    /* int optLevel = std::stoi(argv[2]);
     switch (optLevel) {
     case 0:
       onnx_mlir::setOptLevel(onnx_mlir::O0);
       break;
     case 1:
       onnx_mlir::setOptLevel(onnx_mlir::O1);
       break;
     case 2:
       onnx_mlir::setOptLevel(onnx_mlir::O2);
       break;
     case 3:
       onnx_mlir::setOptLevel(onnx_mlir::O3);
       break;
     default:
       llvm::outs() << "opt level must be on of {0,1,2,3}";
       return -2;
     }*/
    llvm::cl::ParseCommandLineOptions(argc, argv);
    std::string inputFilename = InputFilename.c_str();
    //===========================
    // LETS SEE IF WE NEED THIS

    // copied from onnx-mlir.cpp (lets see what we need)
    //  Register MLIR command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    mlir::registerDefaultTimingManagerCLOptions();
    mlir::registerAsmPrinterCLOptions();

    llvm::cl::SetVersionPrinter(onnx_mlir::getVersionPrinter);
    //===========================
    //
    mlir::MLIRContext context;
    // does not exist at commit a04f518c1
    // context.appendDialectRegistry(onnx_mlir::registerDialects(onnx_mlir::maccel));
    // context.loadAllAvailableDialects();
    onnx_mlir::registerDialects(context);
    context.getOrLoadDialect<zkml::ZkMlDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::string errorMessage;
    if (int rc = loadOnnxFile(llvm::StringRef(inputFilename), context, module, &errorMessage)) {
        llvm::errs() << "Cannot load .onnx file:\n";
        llvm::errs() << errorMessage << "\n";
        return rc;
    }
    bool EmitMLIR = EmitLevel::zkMLIR == EmitLevel || EmitLevel::MLIR == EmitLevel;
    mlir::PassManager pm(&context, mlir::OpPassManager::Nesting::Implicit);
    if (EmitLevel == EmitLevel::ONNX) {
        onnx_mlir::addPasses(module, pm, onnx_mlir::EmissionTargetType::EmitONNXIR);
    } else {
        onnx_mlir::addPasses(module, pm, onnx_mlir::EmissionTargetType::EmitMLIR, EmitLevel == EmitLevel::zkMLIR);
        pm.addPass(zk_ml_toolchain::createElimCopySignPass());
        if (!EmitMLIR) {
            // third parameter here is optional in onnx-mlir. Maybe we should do that
            // too?
            onnx_mlir::addKrnlToLLVMPasses(pm, true, true);
        }
    }

    (void)mlir::applyPassManagerCLOptions(pm);
    mlir::applyDefaultTimingPassManagerCLOptions(pm);
    if (mlir::failed(pm.run(*module))) {
        llvm::errs() << "Passmanager failed to run!\n";
        return -1;
    }
    std::string outputFilename = OutputFilename.c_str();

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
