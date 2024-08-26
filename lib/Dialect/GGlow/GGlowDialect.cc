#include "mlir/include/mlir/IR/Builders.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"


#include "lib/Dialect/GGlow/GGlowDialect.h"

#include "lib/Dialect/GGlow/GGlowDialect.cpp.inc"

#include <memory>
#include <string>
#include <system_error>
#include <utility>

#include <iostream>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#include "mlir/Support/LLVM.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

// gokul addition
#include "mlir/InitAllDialects.h"

// for passes and pass manager
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#define GET_OP_CLASSES
#include "lib/Dialect/GGlow/GGlowOps.cpp.inc"


bool enableOpt = true;

namespace mlir::gglow {
void GlowDialect::initialize()
{
    addOperations<
    #define GET_OP_LIST
        #include "lib/Dialect/GGlow/GGlowOps.cpp.inc"
    >();
    return;
}

}

void dumpMLIR(std::string ir_content){
    mlir::MLIRContext context;

    context.getOrLoadDialect<mlir::gglow::GlowDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(ir_content, &context);
    if (!module){
        llvm::errs() << "Failed to parse MLIR module\n";
        return;
    }

    if(enableOpt){
        mlir::PassManager pm(module.get()->getName());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());

        if (mlir::failed(pm.run(*module)))
            llvm::errs() << "Failed to canonicalize\n";
    }

    module->dump();
}
