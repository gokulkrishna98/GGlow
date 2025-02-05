#include <cstdio>

#include "lib/Dialect/GGlow/GGlow.h"

int main(){
    run_model("/home/gokul/projects/GGlow/models/resnet_block.pt");
    // test_GGlow();
    return 0;
}