include "mlir/IR/DialectBase.td"

def Glow_Dialect : Dialect
{
    let name = "gglow";
    let summary = "A dialect for simple Graph Lowering";
    let description = [{
    The Dialect, that is a Highlevel IR that models Glow IR
    and implements smart lowering.
  }];

    let cppNamespace = "::mlir::gglow";
}