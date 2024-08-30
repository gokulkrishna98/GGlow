#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "GGlowDialect.h"

namespace {
    #include "lib/Dialect/GGlow/CanonicalizePatterns.inc"
}

void mlir::gglow::TransposeOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results, 
    mlir::MLIRContext *context) {
    results.add<SimplifyRedundantTranspose>(context);
}

void mlir::gglow::ReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
    mlir::MLIRContext *context) {
    results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern, FoldConstantReshapePattern>(context);
}