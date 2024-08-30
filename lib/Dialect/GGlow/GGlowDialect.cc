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

struct GGlowInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(Operation *call, Operation *callable,
                            bool wouldBeCloned) const final {
        return true;
    }

    bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
        return true;
    }

    bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
        return true;
    }

    void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
        auto returnOp = cast<ReturnOp>(op);

        assert(returnOp.getNumOperands() == valuesToRepl.size());
        for (const auto &it : llvm::enumerate(returnOp.getOperands()))
            valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }

    Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
        return builder.create<CastOp>(conversionLoc, resultType, input);
    }
};

void GlowDialect::initialize()
{
    addOperations<
    #define GET_OP_LIST
        #include "lib/Dialect/GGlow/GGlowOps.cpp.inc"
    >();

    addInterface<GGlowInlinerInterface>();

}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  TensorType input = llvm::dyn_cast<TensorType>(inputs.front());
  TensorType output = llvm::dyn_cast<TensorType>(outputs.front());
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  return !input.hasRank() || !output.hasRank() || input == output;
}

void CastOp::inferShapes() {
    getResult().setType(getInput().getType());
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//
void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs)
{
    buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result)
{
    auto buildFuncType =
        [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
           llvm::ArrayRef<mlir::Type> results,
           mlir::function_interface_impl::VariadicFlag,
           std::string &)
    { return builder.getFunctionType(argTypes, results); };

    return mlir::function_interface_impl::parseFunctionOp(
        parser, result, /*allowVariadic=*/false,
        getFunctionTypeAttrName(result.name), buildFuncType,
        getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p)
{
    mlir::function_interface_impl::printFunctionOp(
        p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
        getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//
void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments)
{
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(arguments);
    state.addAttribute("callee",
                       mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

CallInterfaceCallable GenericCallOp::getCallableForCallee()
{
    return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee)
{
    (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

MutableOperandRange GenericCallOp::getArgOperandsMutable()
{
    return getInputsMutable();
}

}


void printAvailablePasses(mlir::OpPassManager &pm) {
  llvm::errs() << "Available passes in the PassManager:\n";
  
  // Get all passes in the PassManager
  auto passes = pm.getPasses();
  
  // Iterate through all passes and print their names
  for (const auto &passIt : llvm::enumerate(passes)) {
    const auto &passInfo = passIt.value();
    llvm::errs() << passIt.index() + 1 << ". " << passInfo.getName() << "\n";
    llvm::errs() << passInfo.getOpName() << "\n\n";
  }
}

auto dumpMLIR(std::string ir_content) -> void {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::gglow::GlowDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(ir_content, &context);
    if (!module){
        llvm::errs() << "Failed to parse MLIR module\n";
        return;
    }

    if(enableOpt){
        mlir::PassManager pm(module.get()->getName());
        pm.addPass(mlir::createInlinerPass());

        auto &optPM = pm.nest<mlir::gglow::FuncOp>();
        optPM.addPass(mlir::gglow::createShapeInferencePass());
        optPM.addPass(mlir::createCanonicalizerPass());

        if (mlir::failed(pm.run(*module)))
            llvm::errs() << "Failed to canonicalize\n";
    }

    module->dump();
}
