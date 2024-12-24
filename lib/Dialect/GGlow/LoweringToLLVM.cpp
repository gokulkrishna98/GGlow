#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "lib/Dialect/GGlow/GGlowDialect.h"
#include "lib/Dialect/GGlow/GGlowPass.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <utility>

using namespace mlir;

namespace {

class PrintOpLowering : public ConversionPattern {
private:
    static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context){
        auto llvmI32Ty = IntegerType::get(context, 32);
        auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
        return LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy, true);
    }

    static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter, ModuleOp module){
        auto *context = module.getContext();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
            return SymbolRefAttr::get(context, "printf");

        // Insert the printf function into the body of the parent module.
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", getPrintfType(context));
        return SymbolRefAttr::get(context, "printf");
    }

    static Value getOrCreateGlobalString(Location loc, OpBuilder &builder, StringRef name, StringRef value, ModuleOp module){
        LLVM::GlobalOp global;
        if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
            OpBuilder::InsertionGuard insertGuard(builder);
            builder.setInsertionPointToStart(module.getBody());
            auto type = LLVM::LLVMArrayType::get(IntegerType::get(builder.getContext(), 8), value.size());
            global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value),
                                              /*alignment=*/0);
        }

        Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
        Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));
        return builder.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(), 
            globalPtr, ArrayRef<Value>({cst0, cst0}));
    }
public:
    explicit PrintOpLowering(MLIRContext* context) : 
        ConversionPattern(gglow::PrintOp::getOperationName(), 1, context) {} 

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, 
        ConversionPatternRewriter &rewriter) const override {
        auto *context = rewriter.getContext();
        auto memRefType = llvm::cast<MemRefType>((*op->operand_type_begin()));
        auto memRefShape = memRefType.getShape();
        auto loc = op->getLoc();

        ModuleOp parentModule = op->getParentOfType<ModuleOp>();

        auto printfRef = getOrInsertPrintf(rewriter, parentModule);
        Value formatSpecifierCst = getOrCreateGlobalString(loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
        Value newLineCst = getOrCreateGlobalString(loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

        SmallVector<Value, 4> loopIvs;
        for (size_t i = 0; i < memRefShape.size(); ++i) {
            auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
            auto upperBound = rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
            auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
            auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
            for (Operation &nested : *loop.getBody())
                rewriter.eraseOp(&nested);
            loopIvs.push_back(loop.getInductionVar());
            rewriter.setInsertionPointToEnd(loop.getBody());
            if(i != memRefShape.size() - 1)
                rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef, newLineCst);
            rewriter.create<scf::YieldOp>(loc);
            rewriter.setInsertionPointToStart(loop.getBody());
        }

        // Generate a call to printf for the current element of the loop.
        auto printOp = cast<gglow::PrintOp>(op);
        auto elementLoad = rewriter.create<memref::LoadOp>(loc, printOp.getInput(), loopIvs);
        rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef, ArrayRef<Value>({formatSpecifierCst, elementLoad}));

        rewriter.eraseOp(op);
        return success();
    }

};

} // namespace

namespace {
struct GGlowToLLVMLoweringPass : public PassWrapper<GGlowToLLVMLoweringPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GGlowToLLVMLoweringPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
    }
    void runOnOperation() final;
};
} // namespace


void GGlowToLLVMLoweringPass::runOnOperation() {
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();
  LLVMTypeConverter typeConverter(&getContext());

  RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  patterns.add<PrintOpLowering>(&getContext());

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::gglow::createLowerToLLVMPass() {
  return std::make_unique<GGlowToLLVMLoweringPass>();
}