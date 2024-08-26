#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "GGlowDialect.h"

namespace {
    #include "lib/Dialect/GGlow/CanonicalizePatterns.inc"
}

// C++ implementation
// struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<mlir::gglow::TransposeOp> {
//     SimplifyRedundantTranspose(mlir::MLIRContext *context)
//         : OpRewritePattern<mlir::gglow::TransposeOp>(context, /*benefit=*/1) {}

//     mlir::LogicalResult matchAndRewrite(mlir::gglow::TransposeOp op,
//                                         mlir::PatternRewriter &rewriter) const override
//     {
//         mlir::Value transposeInput = op.getOperand();
//         auto transposeInputOp = transposeInput.getDefiningOp<mlir::gglow::TransposeOp>();
//         if (!transposeInputOp)
//             return mlir::failure();

//         rewriter.replaceOp(op, {transposeInputOp.getOperand()});
//         return mlir::success();
//     }
// };

void mlir::gglow::TransposeOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results, 
    mlir::MLIRContext *context) {
    results.add<SimplifyRedundantTranspose>(context);
}

void mlir::gglow::ReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
    mlir::MLIRContext *context) {
    results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern, FoldConstantReshapePattern>(context);
}