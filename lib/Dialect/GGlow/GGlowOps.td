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

class GGlowOp<string mnemonic, list<Trait> traits = []> :
    Op<Glow_Dialect, mnemonic, traits>;

def LoadOp : GGlowOp<"load"> {
    let summary = "load operation";
    let description = [{
        The "load" operation takes in parameter name in the model state_dict
        and returns a tensor of that parameter in the model.
    }];
    let arguments = (ins StrAttr:$parameter_name);
    let results = (outs AnyTensor:$out_tensor);
    let assemblyFormat = [{
        `(` $parameter_name `)` attr-dict `:` type($out_tensor)
    }];
}

def Conv2dOp : GGlowOp<"conv2d"> {
    let summary = "convolution 2d operation in pytorch";
    let description = [{
        Applies a 2D convolution over an input signal composed of several input planes.
        example code:
        ```
        nil
        ```
    }];
    let arguments = (ins 
        // tensor arguments
        AnyTensor:$x, AnyTensor:$weight, Optional<AnyTensor>:$bias,
        // attributes
        I64Attr:$in_channels,
        I64Attr:$out_channels,
        ArrayAttr:$kernel_size,
        ArrayAttr:$stride,
        ArrayAttr:$padding,
        ArrayAttr:$dilation,
        I64Attr:$groups,
        BoolAttr:$is_bias,
        StrAttr:$padding_mode
    );
    let results = (outs AnyTensor:$out);
    let assemblyFormat = [{  `(` $x `,` $weight `,` $bias `)` attr-dict `:` 
        `(` type($x) `,` type($weight) `,` type($bias) `)` `->` type($out)
    }];
}

def ReluOp : GGlowOp<"relu">{
    let summary = "relu operation";
    let description = [{
        Applies the rectified linear unit function element-wise.
        relu(x) = max(0, x) // elementwise
    }];

    let arguments = (ins AnyTensor:$x);
    let results = (outs AnyTensor:$out);
    let assemblyFormat = [{
        `(` $x `)` attr-dict `:` `(`type($x)`)` `->` type($out)
    }];
}



def BatchNorm2d : GGlowOp <"batch_norm2d">{
    let summary = "batch-norm operation";
    let description = [{
       Applies Batch Normalization over a 4D input 
       usage:
       ```
       ```
    }];
    let arguments = (ins 
        // input tensor
        AnyTensor:$x,
        // attributes     
        I64Attr:$num_features,
        F64Attr:$eps,
        DefaultValuedAttr<F64Attr, "1.0">:$gamma,
        DefaultValuedAttr<F64Attr, "0.0">:$beta,
        F64Attr:$mean,
        F64Attr:$var
    );
    let results = (outs AnyTensor:$out);
    let assemblyFormat = [{
        `(` $x `)` attr-dict `:` `(` type($x) `)` `->` type($out)
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
