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
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Dialect/Affine/Passes.h"

// for lowering to llvmir
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/Module.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

// jit
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Support/TargetSelect.h"

#define GET_OP_CLASSES
#include "lib/Dialect/GGlow/GGlowOps.cpp.inc"


bool enableOpt = true;
bool affine_lowering = true;
bool lowering_to_llvm = false;

namespace mlir::gglow {

void GlowDialect::initialize()
{
    addOperations<
    #define GET_OP_LIST
        #include "lib/Dialect/GGlow/GGlowOps.cpp.inc"
    >();

}

}

void _printAvailablePasses(mlir::OpPassManager &pm) {
  llvm::errs() << "Available passes in the PassManager:\n";
  auto passes = pm.getPasses();
  for (const auto &passIt : llvm::enumerate(passes)) {
    const auto &passInfo = passIt.value();
    llvm::errs() << passIt.index() + 1 << ". " << passInfo.getName() << "\n";
    llvm::errs() << passInfo.getOpName() << "\n\n";
  }
}

int runJit(mlir::ModuleOp module) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());

    auto optPipeline = mlir::makeOptimizingTransformer(3, /*sizeLevel=*/0,/*targetMachine=*/nullptr);

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();
    // Invoke the JIT-compiled function.
    auto invocationResult = engine->invokePacked("main");
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        return -1;
    }
    return 0;
}

void dumpMLIR(std::string ir_content){
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::gglow::GlowDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(ir_content, &context);
    if (!module){
        llvm::errs() << "Failed to parse MLIR module\n";
        return;
    }

    if(enableOpt){
        mlir::PassManager pm(module.get()->getName());

        if(affine_lowering){
            pm.addPass(mlir::gglow::createAffineLoweringPass());
            auto &optPM = pm.nest<mlir::func::FuncOp>();
            optPM.addPass(mlir::createCanonicalizerPass());
            optPM.addPass(mlir::createCSEPass());

            // adding other affine based optimization
            optPM.addPass(mlir::affine::createLoopFusionPass());
            optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
        }

        if(lowering_to_llvm){
            pm.addPass(mlir::gglow::createLowerToLLVMPass());
            pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
        }

        if (mlir::failed(pm.run(*module)))
            llvm::errs() << "Failed to run opt passes\n";
    }

    module->dump();
    // runJit(*module);
}
