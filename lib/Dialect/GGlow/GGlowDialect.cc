#include "GGlowPass.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"

#include "lib/Dialect/GGlow/GGlowDialect.h"
#include "lib/Dialect/GGlow/GGlowDialect.cpp.inc"

#include <memory>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <string>
#include <system_error>
#include <utility>
#include <iostream>

// general includes, need to clean it later
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

// init all dialect.
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

// for passes and pass manager
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

// for lowering to llvmir
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
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

// lowering 
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"

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
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllPasses();
    mlir::MLIRContext context(registry);

    auto module = mlir::parseSourceString<mlir::ModuleOp>(ir_content, &context);
    if (!module){
        llvm::errs() << "Failed to parse MLIR module\n";
        return;
    }

    if(enableOpt){
        mlir::PassManager pm(module.get()->getName());
        if(affine_lowering){
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createConvertTensorToLinalgPass());
            pm.addPass(mlir::bufferization::createOneShotBufferizePass());
            mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
            mlir::bufferization::buildBufferDeallocationPipeline(pm, deallocationOptions);
            pm.addPass(mlir::createConvertLinalgToLoopsPass());
            pm.addPass(mlir::createLowerAffinePass());
            pm.addPass(mlir::createConvertSCFToCFPass());
            pm.addPass(mlir::createConvertControlFlowToLLVMPass());
            pm.addPass(mlir::createArithToLLVMConversionPass());
            pm.addPass(mlir::createConvertFuncToLLVMPass());
            pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
            pm.addPass(mlir::createReconcileUnrealizedCastsPass());
            pm.addPass(mlir::createCSEPass());
            pm.addPass(mlir::createCanonicalizerPass());
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
