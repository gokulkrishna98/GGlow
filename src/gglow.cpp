#include <iostream>
#include <torch/torch.h>

int main() {
    std::cout << "hello\n";
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    return 0;
}