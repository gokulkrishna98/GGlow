#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "GGlowDialect.h"
#include "GGlowOpsInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace mlir::gglow {

#include "lib/Dialect/GGlow/GGlowOpsInterface.cpp.inc"

struct ShapeInferencePass : public mlir::PassWrapper<ShapeInferencePass, OperationPass<mlir::gglow::FuncOp>> {

    static bool returnsDynamicShape(Operation* op) {
        return llvm::any_of(op->getResultTypes(), [](Type resultType){
            return !llvm::isa<RankedTensorType>(resultType);
        });
    }

    static bool allOperandsInferred(Operation* op){
        return llvm::all_of(op->getOperandTypes(), [](Type operandType) {
            return llvm::isa<RankedTensorType>(operandType);
        });
    }

    StringRef getArgument() const final { return "shape-inference"; }
    StringRef getDescription() const final{
        return "calculate shapes from xdd";
    }

    void runOnOperation() override {
        auto f = getOperation();

        llvm::SmallVector<mlir::Operation*, 16> opWorklist;
        f.walk([&](mlir::Operation* op) {
            if(returnsDynamicShape(op)){
                opWorklist.push_back(op);
            }
        });

        for(size_t i =0; i<opWorklist.size(); i++) {
            Operation *op = opWorklist[i];
            if (auto shapeOp = dyn_cast<ShapeInference>(op)){
                shapeOp.inferShapes();
            } else {
                op->emitError("unable to infer shape of operation without shape "
                            "inference interface");
                return signalPassFailure();
            }
        }

        opWorklist.clear();
        return;
    }
};


std::unique_ptr<mlir::Pass> createShapeInferencePass(){
    return std::make_unique<ShapeInferencePass>();
}

}