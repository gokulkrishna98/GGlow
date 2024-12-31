include "GGlowDialect.td"

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

/*
NOTE: Not implementing verifier,
will do a readup and implement it after general implementation
is ready
*/

/*
---------- Operation List -------------
- PrintOp
*/

class GGlowOp<string mnemonic, list<Trait> traits = []> :
    Op<Glow_Dialect, mnemonic, traits>;


def PrintOp : GGlowOp<"print"> {
    let summary = "print operation";
    let description = [{
        The "print" builtin operation prints a given input tensor, and produces
        no results.
    }];

    // The print operation takes an input tensor to print.
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
    let assemblyFormat = "$input attr-dict `:` type($input)";
}
