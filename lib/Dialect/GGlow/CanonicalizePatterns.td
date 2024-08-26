#ifndef CANONICALIZE_PATTERNS_TD
#define CANONICALIZE_PATTERNS_TD

include "mlir/IR/PatternBase.td"
include "GGlowOps.td"

// Tablegen way:
// transpose(transpose(x)) = x
def SimplifyRedundantTranspose : Pat<(TransposeOp(TransposeOp $arg)), (replaceWithValue $arg) >;

// reshape does the same operation
// reshape(reshape(x)) = x
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)), (ReshapeOp $arg)>;

// if input and output are of same shape, then reshape is meaningless
def TypesIdentical : Constraint<CPred<"$0.getType() == $1.getType()">> ;
def RedundantReshapeOptPattern : Pat <(ReshapeOp:$res $arg), (replaceWithValue $arg),
    [(TypesIdentical $res, $arg)]>;

def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<mlir::ShapedType>())">;
def FoldConstantReshapePattern : Pat <
    (ReshapeOp:$res (ConstantOp $arg)),
    (ConstantOp (ReshapeConstant $arg, $res))>;


#endif 