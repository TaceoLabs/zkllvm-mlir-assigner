#include "src/Builder/FrontendDialectTransformer.hpp"
#include <Passes/mlir/Transform/SetPrivateInputPass.hpp>
#include "mlir/Dialect/zkml/IR/ZkMlAttributes.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include <unordered_map>

using namespace mlir;

void zk_ml::SetPrivateInputPass::runOnOperation() {
    mlir::ModuleOp Module = getOperation();
    Region &BodyRegion = Module.getBodyRegion();
    assert(BodyRegion.getBlocks().size() == 1 && "module must have exactly one block");
    Block &BodyBlock = BodyRegion.getBlocks().front();
    std::string MainFuncSym = "";
    std::unordered_map<std::string, func::FuncOp> FuncDecls;
    for (Operation &Op : BodyBlock) {
        if (func::FuncOp FuncDecl = llvm::dyn_cast<func::FuncOp>(Op)) {
            auto res = FuncDecls.insert({FuncDecl.getSymName().str(), FuncDecl});
            assert(res.second && "Redifintion of function");
        } else if (KrnlEntryPointOp EntryPoint = llvm::dyn_cast<KrnlEntryPointOp>(Op)) {
            for (auto Attr : EntryPoint->getAttrs()) {
                if (Attr.getName() == EntryPoint.getEntryPointFuncAttrName()) {
                    MainFuncSym = Attr.getValue().cast<SymbolRefAttr>().getLeafReference().str();
                } else if (Attr.getName() == EntryPoint.getNumInputsAttrName()) {
                    unsigned numInputs = Attr.getValue().cast<IntegerAttr>().getInt();
                    if (numInputs != privateInputs.size()) {
                        llvm::errs() << "Entrypoint defined " << numInputs << " inputs, but " << privateInputs.size()
                                     << " provided\n";
                        return signalPassFailure();
                    }
                }
            }
        } else {
            llvm::errs() << "unknown operation: " << Op.getName().getIdentifier() << "\n";
            return signalPassFailure();
        }
    }

    if (FuncDecls.find(MainFuncSym) == FuncDecls.end()) {
        llvm::errs() << "could not find entry point function: " << MainFuncSym << "\n";
        return signalPassFailure();
    }
    func::FuncOp &MainFunc = FuncDecls[MainFuncSym];
    if (MainFunc.getBody().getArguments().size() != privateInputs.size()) {
        llvm::errs() << "Corrupted .mlir (KrnlEntryPoint gave wrong numInputs).\n";
        llvm::errs() << "Also expected " << MainFunc.getBody().getArguments().size() << " inputs, but got "
                     << privateInputs.size() << "\n";
        return signalPassFailure();
    }
    MLIRContext *Context = &getContext();
    for (unsigned i = 0; i < privateInputs.size(); ++i) {
        StringAttr Name = StringAttr::get(Context, "zkML.input");
        if (privateInputs[i]) {
            MainFunc.setArgAttrs(i, {NamedAttribute(Name, zkml::ZkMlPrivateInputAttr::get(Context))});
        }
    }
}

std::unique_ptr<Pass> zk_ml::createSetPrivateInputPass(std::vector<bool> &privateInputs) {
    return std::make_unique<SetPrivateInputPass>(privateInputs);
}
