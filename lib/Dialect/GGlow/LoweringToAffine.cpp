#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "lib/Dialect/GGlow/GGlowDialect.h"
#include "lib/Dialect/GGlow/GGlowPass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>


using namespace mlir;

/*
GGLOW to Affine Lowering
*/

static MemRefType convertTensorToMemRef(RankedTensorType type){
    return MemRefType::get(type.getShape(), type.getElementType());
}


static Value insertAllocAndDeAlloc(MemRefType type, Location loc, PatternRewriter &rewriter){
    auto alloc = rewriter.create<memref::AllocaOp>(loc, type);

    auto *parent_block = alloc->getBlock();
    alloc->moveBefore(&parent_block->front());

    auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
    dealloc->moveBefore(&parent_block->back());

    return alloc;
}

using LoopIterationFunction = 
    function_ref<Value(OpBuilder &rewriter, ValueRange mem_ref_operands, ValueRange loop_ivs)>;


/*
This function does three main things:
1. convert result from tensor to memref.
2. build the affine computation.
3. replace the op to the affine form.
*/
static void lowerOpToLoops(Operation *op, ValueRange operands, PatternRewriter &rewriter,
    LoopIterationFunction process_iteration){
    auto tensor_type = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    auto loc = op->getLoc();


    auto mem_ref_type = convertTensorToMemRef(tensor_type);
    auto alloc = insertAllocAndDeAlloc(mem_ref_type, loc, rewriter);

    SmallVector<int64_t, 4> lbs(tensor_type.getRank(), 0);
    SmallVector<int64_t, 4> steps(tensor_type.getRank(), 1);

    affine::buildAffineLoopNest(
        rewriter, loc, lbs, tensor_type.getShape(), steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
            // assumption is that process iteration is going to do a store operation
            // else the loop is meaningless ?
            Value value_to_store = process_iteration(builder, operands, ivs);
            builder.create<affine::AffineStoreOp>(loc, value_to_store, alloc, ivs);
        }
    );

    return rewriter.replaceOp(op, alloc);
}


namespace {
/*
    LOWERING FROM GGLOW to AFFINE: Rewrite Patterns
*/


// LOWERING BINARY OPS
template<typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
    BinaryOpLowering(MLIRContext *ctx) : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}; 
    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, 
        ConversionPatternRewriter &rewriter) const final {
    
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter, [loc](OpBuilder &builder, ValueRange mem_ref_operands, 
            ValueRange ivs){
            typename BinaryOp::Adaptor binaryAdaptor(mem_ref_operands);

            auto loaded_lhs = builder.create<affine::AffineLoadOp>(loc, binaryAdaptor.getLhs(), ivs);
            auto loaded_rhs = builder.create<affine::AffineLoadOp>(loc, binaryAdaptor.getRhs(), ivs);

            return builder.create<LoweredBinaryOp>(loc, loaded_lhs, loaded_rhs);
        });

        return success();
    }
};

using AddOpLowering = BinaryOpLowering<gglow::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<gglow::MulOp, arith::MulFOp>;

} // ending namespace


// LOWERING CONSTANT OPERATION
struct ConstantOpLowering : public OpRewritePattern<gglow::ConstantOp> {
    using OpRewritePattern<gglow::ConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(gglow::ConstantOp op, PatternRewriter &rewriter) const final {
        DenseElementsAttr constant_value = op.getValue();
        Location loc = op.getLoc();

        auto tensor_type = llvm::cast<RankedTensorType>(op.getType());
        auto memref_type = convertTensorToMemRef(tensor_type);
        auto alloc = insertAllocAndDeAlloc(memref_type, loc, rewriter);

        auto value_shape = memref_type.getShape();
        SmallVector<Value> constant_indices;
    
        if(!value_shape.empty()){
            for(auto i: llvm::seq<int64_t>(0, *std::max_element(value_shape.begin(), value_shape.end()))){
                constant_indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, i));
            }
        }else{
            constant_indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
        } 

        SmallVector<Value> indices;
        auto value_iterator = constant_value.value_begin<FloatAttr>();

        auto storeElements = [&](auto && storeElements, uint64_t dim_size) -> void {
            if(dim_size == value_shape.size()){
                rewriter.create<affine::AffineStoreOp>(
                    loc, 
                    rewriter.create<arith::ConstantOp>(loc, *value_iterator),
                    alloc,
                    llvm::ArrayRef(indices)
                );
                value_iterator++;
                return;
            }

            for(int64_t i = 0; i < value_shape[dim_size]; i++){
                indices.push_back(constant_indices[i]);
                storeElements(storeElements, dim_size+1);
                indices.pop_back();
            }
            return;
        };

        storeElements(storeElements, 0);
        rewriter.replaceOp(op, alloc);
        return success();
    }
};

// LOWERING FUNCTION OP
struct FuncOpLowering : public OpConversionPattern<gglow::FuncOp> {
    using OpConversionPattern<gglow::FuncOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(gglow::FuncOp op, OpAdaptor adapter, ConversionPatternRewriter &rewriter) const final {
        
        if(op.getName() != "main"){
            return failure();
        }

        if(op.getNumArguments() || op.getFunctionType().getNumResults()){
            return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
                diag << "expected 'main' to have 0 inputs and 0 results";
            });
        }

        auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(), op.getFunctionType());
        rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
        rewriter.eraseOp(op);

        return success();
    }
};

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


// LOWERING RETURN OPERATION
struct ReturnOpLowering : public OpRewritePattern<gglow::ReturnOp> {
    using OpRewritePattern<gglow::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(gglow::ReturnOp op, PatternRewriter &rewriter) const final {
        if(op.hasOperand()){
            return failure();
        }

        rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
        return success();
    }
};


// LOWERING TRANSPOSE OPERATION
struct TransposeOpLowering : public ConversionPattern {
    TransposeOpLowering(MLIRContext *ctx) : 
        ConversionPattern(gglow::TransposeOp::getOperationName(), 1, ctx) {};

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands, 
        ConversionPatternRewriter &rewriter) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter, [loc](OpBuilder &builder, 
            ValueRange mem_ref_operands, ValueRange ivs){
            gglow::TransposeOpAdaptor transpose_adopter(mem_ref_operands);
            Value input = transpose_adopter.getInput();

            SmallVector<Value> reverse_ivs(llvm::reverse(ivs));
            return builder.create<affine::AffineLoadOp>(loc, input, reverse_ivs);
        });

        return success();
    }
};



namespace {

struct GGlowToAffineLoweringPass
    : public PassWrapper<GGlowToAffineLoweringPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GGlowToAffineLoweringPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<affine::AffineDialect, func::FuncDialect,
                    memref::MemRefDialect>();
    }
    void runOnOperation() final;
};

} // namespace

void GGlowToAffineLoweringPass::runOnOperation() {
    ConversionTarget target(getContext());

    target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                            arith::ArithDialect, func::FuncDialect,
                            memref::MemRefDialect>();

    target.addIllegalDialect<gglow::GlowDialect>();
    target.addDynamicallyLegalOp<gglow::PrintOp>([](gglow::PrintOp op) {
        return llvm::none_of(op->getOperandTypes(), [](Type type) { return llvm::isa<TensorType>(type); });
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<
        AddOpLowering, 
        MulOpLowering, 
        ConstantOpLowering, 
        FuncOpLowering,
        ReturnOpLowering,
        TransposeOpLowering,
        PrintOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::gglow::createAffineLoweringPass() {
  return std::make_unique<GGlowToAffineLoweringPass>();
}


