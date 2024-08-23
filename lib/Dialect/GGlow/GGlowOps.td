include "GGlowDialect.td"

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class GGlowOp<string mnemonic, list<Trait> traits = []> :
    Op<Glow_Dialect, mnemonic, traits>;

def ConstantOp : GGlowOp <"constant", [Pure]> {
    let summary = "constant";
    let description = [{
        Constant operation turns a literal into an SSA value. The data is attached
        to the operation as an attribute. For example:

        ```mlir
        %0 = "gglow.constant"() { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                            : tensor<2x3xf64> } : () -> tensor<2x3xf64>
        ```
    }]; 

    let arguments = (ins F64ElementsAttr:$value);
    let results = (outs F64Tensor);

    let assemblyFormat = [{ 
        `(` $value attr-dict `)` `->` type(results)
    }];
}