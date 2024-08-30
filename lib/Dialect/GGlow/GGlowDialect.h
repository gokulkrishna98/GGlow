#ifndef GGLOWDIALECT_H
#define GGLOWDIALECT_H
#include "mlir/include/mlir/IR/DialectImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "lib/Dialect/GGlow/GGlowDialect.h.inc"

#include "GGlowOpsInterface.h"

#define GET_OP_CLASSES
#include "lib/Dialect/GGlow/GGlowOps.h.inc"

void dumpMLIR(std::string ir_content);

#endif
