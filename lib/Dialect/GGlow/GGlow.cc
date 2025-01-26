#include "GGlowDialect.h"
#include "GGlow.h"

void test_GGlow(){
//     auto test_string = R"(
// module {
//   func.func @main() -> f64 {
//     %c0 = arith.constant 0 : index
//     %cst = arith.constant dense<0.000000e+00> : tensor<1x2x5x5xf64>
//     %cst_0 = arith.constant dense<0.000000e+00> : tensor<4x2x3x3xf64>
//     %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x4x3x3xf64>
//     %0 = linalg.conv_2d_nchw_fchw ins(%cst, %cst_0 : tensor<1x2x5x5xf64>, tensor<4x2x3x3xf64>) outs(%cst_1 : tensor<1x4x3x3xf64>) -> tensor<1x4x3x3xf64>
//     %extracted = tensor.extract %0[%c0, %c0, %c0, %c0] : tensor<1x4x3x3xf64>
//     return %extracted : f64
//   }
// }
//     )";
    std::string test_string = R"(
module {
  func.func @main_ch(){
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<1.000000e+00> : tensor<1x2x5x5xf64>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<4x2x3x3xf64>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x4x3x3xf64>
    %0 = linalg.conv_2d_nchw_fchw ins(%cst, %cst_0 : tensor<1x2x5x5xf64>, tensor<4x2x3x3xf64>) outs(%cst_1 : tensor<1x4x3x3xf64>) -> tensor<1x4x3x3xf64>
    %extracted = tensor.extract %0[%c0, %c0, %c0, %c0] : tensor<1x4x3x3xf64>
    vector.print %extracted : f64
    return 
  }
}
    )";
    dumpMLIR(test_string);
    return;
} 