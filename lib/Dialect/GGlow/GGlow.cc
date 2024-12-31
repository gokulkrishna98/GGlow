#include "GGlowDialect.h"
#include "GGlow.h"

void test_GGlow(){
    auto test_string = R"(
            func.func @main() {
                %0 = arith.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], 
                    [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> 
                gglow.print %0 : tensor<2x3xf64>
                func.return
            }
    )";
    dumpMLIR(test_string);
    return;
} 