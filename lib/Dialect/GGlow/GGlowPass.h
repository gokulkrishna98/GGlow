#ifndef GGLOW_OPS_INTERFACE_H
#define GGLOW_OPS_INTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace gglow{
    std::unique_ptr<mlir::Pass> createAffineLoweringPass();
    std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
} // namespace gglow 
} // namespace mlir

#endif