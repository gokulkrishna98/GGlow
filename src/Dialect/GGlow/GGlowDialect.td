include "mlir/IR/DialectBase.td"

def Poly_Dialect : Dialect {
  let name = "GGlow";
  let summary = "A dialect for simple Graph Lowering";
  let description = [{
        To be defined TBD
  }];

  let cppNamespace = "::mlir::tutorial::GGlow";
}