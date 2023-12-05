
#include "PrintPass.h"

StringRef zk_ml_toolchain::PrintPass::getArgument() const { return "print-pass"; }

StringRef zk_ml_toolchain::PrintPass::getDescription() const
{
  return "Prints some Debug Information (copied from Tutorial)";
}

void zk_ml_toolchain::PrintPass::runOnOperation()
{
  Operation *op = getOperation();
  resetIndent();
  std::vector<llvm::hash_code> typeIds;
  printOperation(op, typeIds);
}

void zk_ml_toolchain::PrintPass::printVector(std::vector<llvm::hash_code> &typeIds)
{
  std::cout << "[";
  for (auto element : typeIds)
  {
    std::cout << element << ", ";
  }
  std::cout << "]" << std::endl;
}

void zk_ml_toolchain::PrintPass::printOperation(Operation *op, std::vector<llvm::hash_code> &typeIds)
{
  // Print the operation itself and some of its properties
  std::string opName = op->getName().getIdentifier().str();
  if (opName == "krnl.gloabl")
  {
    printIndent() << "visiting: krnl.global";
    return;
  }
  unsigned numOperands = op->getNumOperands();
  unsigned numResults = op->getNumResults();
  printIndent() << "visiting op: '" << op->getName() << "' with "
                << numOperands << " operands and " << numResults
                << " results\n";
  // Print the operation attributes
  if (!op->getAttrs().empty())
  {
    printIndent() << op->getAttrs().size() << " attributes:\n";
    for (NamedAttribute attr : op->getAttrs())
      printIndent() << " - '" << attr.getName().getValue() << "' : '"
                    << attr.getValue() << "'\n";
  }

  // Recurse into each of the regions attached to the operation.
  printIndent() << " " << op->getNumRegions() << " nested regions:\n";
  if (opName == "arith.constant")
  {
    if (numResults != 1)
    {
      std::cout << "whaaaat" << std::endl;
      exit(0);
    }
    llvm::hash_code hash = hash_value(op->getResults()[0]);
    std::cout << hash << std::endl;
    printVector(typeIds);
    if (std::find(typeIds.begin(), typeIds.end(), hash) != typeIds.end())
    {
      std::cout << "whaaaaaaaaat already in vector" << std::endl;
      std::cout << *(std::find(typeIds.begin(), typeIds.end(), hash))
                << std::endl;
    }
    else
    {
      typeIds.emplace_back(hash);
    }
  }
  else if (opName == "affine.for" && numOperands > 0)
  {
    OperandRange operands = op->getOperands();
    for (uint64_t i = 0; i < operands.size(); ++i)
    {
      llvm::hash_code hash = hash_value(operands[i].getType());
      if (std::find(typeIds.begin(), typeIds.end(), hash) == typeIds.end())
      {
        std::cout << "whaaaaaaaaat not in vector" << std::endl;
        exit(0);
      }
    }
  }
  auto indent = pushIndent();
  for (Region &region : op->getRegions())
    printRegion(region, typeIds);
}

void zk_ml_toolchain::PrintPass::printRegion(Region &region, std::vector<llvm::hash_code> &typeIds)
{
  // A region does not hold anything by itself other than a list of blocks.
  printIndent() << "Region with " << region.getBlocks().size()
                << " blocks:\n";
  auto indent = pushIndent();
  for (Block &block : region.getBlocks())
    printBlock(block, typeIds);
}

void zk_ml_toolchain::PrintPass::printBlock(Block &block, std::vector<llvm::hash_code> &typeIds)
{
  // Print the block intrinsics properties (basically: argument list)
  printIndent()
      << "Block with " << block.getNumArguments() << " arguments, "
      << block.getNumSuccessors()
      << " successors, and "
      // Note, this `.size()` is traversing a linked-list and is O(n).
      << block.getOperations().size() << " operations\n";

  // Block main role is to hold a list of Operations: let's recurse.
  auto indent = pushIndent();
  for (Operation &op : block.getOperations())
    printOperation(&op, typeIds);
}

zk_ml_toolchain::PrintPass::IdentRAII::IdentRAII(int &indent) : indent(indent) {}

zk_ml_toolchain::PrintPass::IdentRAII::~IdentRAII() { --indent; }

void zk_ml_toolchain::PrintPass::resetIndent() { indent = 0; }

zk_ml_toolchain::PrintPass::IdentRAII zk_ml_toolchain::PrintPass::pushIndent() { return IdentRAII(++indent); }

llvm::raw_ostream &zk_ml_toolchain::PrintPass::printIndent()
{
  for (int i = 0; i < indent; ++i)
    llvm::outs() << "  ";
  return llvm::outs();
}

std::unique_ptr<Pass> zk_ml_toolchain::createPrintPass()
{
  return std::make_unique<PrintPass>();
}
