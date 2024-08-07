#include "mlir/include/mlir/IR/Builders.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"

#include "lib/Dialect/GGlow/GGlowDialect.h"

#include "lib/Dialect/GGlow/GGlowDialect.cpp.inc"

namespace mlir::gglow {

void GlowDialect::initialize()
{
    // add types and operations
    return;
}

}
