#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/TypeID.h"

#include "GGlowDialect.h" // IWYU pragma: keep
#include "GGlowPass.h" // IWYU pragma: keep

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"


using namespace mlir;

/*
GGLOW to Affine Lowering
*/

namespace {
/*
LOWERING FROM GGLOW to AFFINE: Rewrite Patterns
*/

// LOWERING PRINT OPERATION
// we do not lower it, but have to convert the operands type from tensor to memref
struct PrintOpLowering : public OpConversionPattern<gglow::PrintOp>{
    using OpConversionPattern<gglow::PrintOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(gglow::PrintOp op, OpAdaptor adapter, ConversionPatternRewriter &rewriter) const final {
        rewriter.modifyOpInPlace(op, [&](){
            op->setOperands(adapter.getOperands());
        });

        return success();
    }
};

} // namespace


namespace {

struct GGlowToAffineLoweringPass
    : public PassWrapper<GGlowToAffineLoweringPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GGlowToAffineLoweringPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<
            affine::AffineDialect, 
            func::FuncDialect, 
            memref::MemRefDialect,
            scf::SCFDialect>();
    }
    void runOnOperation() final;
};

} // namespace

void GGlowToAffineLoweringPass::runOnOperation() {
    ConversionTarget target(getContext());

    target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                            arith::ArithDialect, func::FuncDialect,
                            memref::MemRefDialect, scf::SCFDialect>();

    target.addIllegalDialect<gglow::GlowDialect>();
    target.addDynamicallyLegalOp<gglow::PrintOp>([](gglow::PrintOp op) {
        return llvm::none_of(op->getOperandTypes(), [](Type type) { return llvm::isa<TensorType>(type); });
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<PrintOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::gglow::createAffineLoweringPass() {
  return std::make_unique<GGlowToAffineLoweringPass>();
}


