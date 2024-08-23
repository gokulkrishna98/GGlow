#include "GGlowDialect.h"
#include "GGlow.h"

void test_GGlow(){
    auto ir_string = R"(
        module {
            func.func @main() {
                %0 = gglow.constant ( dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                    : tensor<2x3xf64> ) -> tensor<2x3xf64>
                return
            }
        }
    )";

    dumpMLIR(ir_string);
    return;
} 