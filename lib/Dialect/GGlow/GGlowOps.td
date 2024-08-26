include "GGlowDialect.td"

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

/*
NOTE: Not implementing verifier,
will do a readup and implement it after general implementation
is ready
*/

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


def TransposeOp : GGlowOp <"transpose", [Pure]> {
    let summary = "transpose operation";

    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor);

    let assemblyFormat = [{
        `(` $input `:` type($input) attr-dict `)` `->` type(results)
    }];

    let hasCanonicalizer = 1;

}

/*

def ReshapeOp : Toy_Op<"reshape", [Pure]> {
  let summary = "tensor reshape operation";
  let description = [{
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
       %0 = toy.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
    ```
  }];

  let arguments = (ins F64Tensor:$input);

  // We expect that the reshape operation returns a statically shaped tensor.
  let results = (outs StaticShapeTensorOf<[F64]>);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;
}
*/

def ReshapeOp : GGlowOp <"reshape", [Pure]> {
    let summary = "reshape operation";
    let description = [{
        Reshape operation is transforming its input tensor into a new tensor with
        the same number of elements but different shapes. For example:

        ```mlir
        %0 = gglow.reshape (%arg1 : tensor<10xf64>) -> tensor<5x2xf64>
        ```
    }];

    let arguments = (ins F64Tensor:$input);
    let results = (outs StaticShapeTensorOf<[F64]>);

    let assemblyFormat = [{
        `(` $input `:` type($input) attr-dict `)` `->` type(results) 
    }];

    let hasCanonicalizer = 1;
}