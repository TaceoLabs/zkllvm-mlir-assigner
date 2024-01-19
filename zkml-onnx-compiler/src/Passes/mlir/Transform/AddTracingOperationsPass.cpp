#include "src/Builder/FrontendDialectTransformer.hpp"
#include <Passes/mlir/Transform/AddTracingOperationsPass.hpp>
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "mlir/Dialect/zkml/IR/Trace.h"
#include "mlir/Dialect/zkml/IR/OnnxAmount.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

void zk_ml::AddTracingOperationsPass::runOnOperation() {
    OpBuilder Builder(&getContext());
    func::FuncOp FuncOp = getOperation();
    assert(FuncOp.getBlocks().size() == 1 && "main func must have one block");
    Block &MainBlock = FuncOp.getBlocks().front();
    unsigned OnnxAmount = MainBlock.getOperations().size();
    for (Operation &OnnxOp : MainBlock.getOperations()) {
        assert(llvm::isa<ONNXDialect>(OnnxOp.getDialect()));
        if (llvm::isa<ONNXReturnOp>(OnnxOp)) {
            continue;
        }
        std::string opName = OnnxOp.getName().getIdentifier().str();
        Builder.setInsertionPoint(&OnnxOp);
        Builder.create<zkml::TraceOp>(OnnxOp.getLoc(), opName);
        //
        // llvm::outs() << "==============="
        //              << "\n";
        // llvm::outs() << "got operation: " << opName << "\n";
        // llvm::outs() << "input: [";
        // for (auto Operands : OnnxOp.getOperands()) {
        //     llvm::outs() << Operands << ", ";
        // }
        // llvm::outs() << "]\n";
    }
    Builder.setInsertionPointToStart(&MainBlock);
    //ignore the return
    Builder.create<zkml::OnnxAmountOp>(FuncOp.getLoc(), OnnxAmount - 1);
}

std::unique_ptr<Pass> zk_ml::createAddTracingOperationsPass() {
    return std::make_unique<AddTracingOperationsPass>();
}
