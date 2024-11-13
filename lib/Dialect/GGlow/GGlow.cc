#include "GGlowDialect.h"
#include "GGlow.h"

void test_GGlow(){
    auto constantop_string = R"(
        module {
            gglow.func @main() {
                %0 = gglow.constant ( dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                    : tensor<2x3xf64> ) -> tensor<2x3xf64>
                gglow.return
            }
        }
    )";

    auto transposeop_string = R"(
        module {
            gglow.func @transpose_simplify() -> tensor<2x3xf64> {
                %0 = gglow.constant ( dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                    : tensor<2x3xf64> ) -> tensor<2x3xf64>
                %1 = gglow.transpose (%0: tensor<2x3xf64>) -> tensor<3x2xf64>
                %2 = gglow.transpose (%1: tensor<3x2xf64>) -> tensor<2x3xf64>
                gglow.return %2 : tensor<2x3xf64>
            }
        }
    )";

    auto reshapeop_string = R"(
        module {
            gglow.func @reshape_simplify() -> tensor<2x1xf64> {
                %0 = gglow.constant ( dense<[1.0, 2.0]> : tensor<2xf64> ) -> tensor<2xf64>
                %1 = gglow.reshape (%0: tensor<2xf64>) -> tensor<2x1xf64>
                %2 = gglow.reshape (%1: tensor<2x1xf64>) -> tensor<2x1xf64>
                %3 = gglow.reshape (%2: tensor<2x1xf64>) -> tensor<2x1xf64>
                gglow.return %3 : tensor<2x1xf64>
            }
        }
    )";

    auto inline_test = R"(
        module {
            gglow.func @transpose_simplify() -> tensor<2x3xf64> {
                %0 = gglow.constant ( dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                    : tensor<2x3xf64> ) -> tensor<2x3xf64>
                %1 = gglow.transpose (%0: tensor<2x3xf64>) -> tensor<3x2xf64>
                %2 = gglow.transpose (%1: tensor<3x2xf64>) -> tensor<2x3xf64>
                gglow.return %2 : tensor<2x3xf64>
            }

            gglow.func @main() {
                %0 = gglow.generic_call @transpose_simplify() : () -> tensor<2x3xf64>
                gglow.print %0 : tensor<2x3xf64>
                gglow.return
            }
        }
    )";

    auto shape_inference = R"(
            gglow.func @transpose_simplify(%arg0 : tensor<*xf64>) -> tensor<*xf64> {
                %0 = gglow.transpose (%arg0: tensor<*xf64>) -> tensor<*xf64>
                %1 = gglow.transpose (%0: tensor<*xf64>) -> tensor<*xf64>
                gglow.print %1 : tensor<*xf64>
                gglow.return %1 : tensor<*xf64>
            }
            gglow.func @main() {
                %0 = gglow.constant ( dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                    : tensor<2x3xf64> ) -> tensor<2x3xf64>
                %1 = gglow.generic_call @transpose_simplify(%0) : (tensor<2x3xf64>) -> tensor<*xf64>
                gglow.print %1 : tensor<*xf64>
                gglow.return
            }
    )";

    auto partial_lowering = R"(
            gglow.func @main() {
                %0 = gglow.constant ( dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], 
                    [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>) -> tensor<2x3xf64>
                %1 = gglow.transpose (%0 : tensor<2x3xf64>) -> tensor<3x2xf64>
                %2 = gglow.mul (%1, %1) -> tensor<3x2xf64>
                gglow.print %2 : tensor<3x2xf64>
                gglow.return
            }
    )";

    // dumpMLIR(constantop_string);
    // dumpMLIR(transposeop_string);
    // dumpMLIR(reshapeop_string);
    // dumpMLIR(inline_test);
    // dumpMLIR(shape_inference);
    dumpMLIR(partial_lowering);
    return;
} 