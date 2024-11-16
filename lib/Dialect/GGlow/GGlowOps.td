include "GGlowDialect.td"
include "GGlowOpsInterface.td"

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
- CastOp
- ConstantOp
- GenericCallOp (function calls)
- FuncOp
- PrintOp
- ReshapeOp
- ReturnOp
- TransposeOp

*/

class GGlowOp<string mnemonic, list<Trait> traits = []> :
    Op<Glow_Dialect, mnemonic, traits>;

def AddOp: GGlowOp <"add",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]>{
    let summary = "element-wise addition operation";
    let description = [{
        Does element-wise addition between two tensors and shapes of both of them have to match.
    }];
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor:$output);

    let assemblyFormat = [{
        $lhs $rhs attr-dict `:` type($lhs) type($rhs) `->` type($output)
    }];
}

def CastOp : GGlowOp <"cast", [ 
    DeclareOpInterfaceMethods<CastOpInterface>,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    Pure, 
    SameOperandsAndResultShape]> {
    let summary = "shape cast operation";
    let description = [{
        The "cast" operation converts a tensor from one type to an equivalent type
        without changing any data elements. The source and destination types
        must both be tensor types with the same element type. If both are ranked,
        then shape is required to match. The operation is invalid if converting
        to a mismatching constant dimension.
    }];

    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor:$output);
    let assemblyFormat = [{
        $input attr-dict `:` type($input) `->` type($output)
    }];


}


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

def GenericCallOp : GGlowOp <"generic_call", [DeclareOpInterfaceMethods<CallOpInterface>]> {
    let summary = "generic call operation";
    let description = [{
        Generic calls represent calls to a user defined function that needs to
        be specialized for the shape of its arguments. The callee name is attached
        as a symbol reference via an attribute. The arguments list must match the
        arguments expected by the callee. For example:

        ```mlir
        %4 = gglow.generic_call @my_func(%1, %3)
            : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
        ```

        This is only valid if a function named "my_func" exists and takes two
        arguments.
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);

    let results = (outs F64Tensor);

    let assemblyFormat = [{
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
    }];
    
    let builders = [
        OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
    ];
}

def FuncOp : GGlowOp<"func", [FunctionOpInterface, IsolatedFromAbove]> {
    let summary = "user defined function operation";
    let description = [{
        The "gglow.func" operation represents a user defined function. These are
        callable SSA-region operations that contain toy computations.

        Example:

        ```mlir
        gglow.func @main() {
        %0 = gglow.constant dense<5.500000e+00> : tensor<f64>
        %1 = gglow.reshape(%0 : tensor<f64>) -> tensor<2x2xf64>
        gglow.print %1 : tensor<2x2xf64>
        gglow.return
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );

    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
        "StringRef":$name, "FunctionType":$type,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >];

    let extraClassDeclaration = [{
        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
        Region *getCallableRegion() { return &getBody(); }
    }];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}


def MulOp: GGlowOp <"mul",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]>{
    let summary = "element-wise multiplication operation";
    let description = [{
        Does element-wise multiplication between two tensors and shapes of both of them have to match.
    }];
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor:$output);

    let assemblyFormat = [{
        `(` $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `)` attr-dict `->` type($output)
    }];
}


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


def ReturnOp : GGlowOp<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
    let summary = "return operation";
    let description = [{
        The "return" operation represents a return operation within a function.
        The operation takes an optional tensor operand and produces no results.
        The operand type must match the signature of the function that contains
        the operation. For example:

        ```mlir
            gglow.func @foo() -> tensor<2xf64> {
            ...
            gglow.return %0 : tensor<2xf64>
        }
        ```
    }];

    let arguments = (ins Variadic<F64Tensor>:$input);
    let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

    let builders = [
        OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
    ];

    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
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
