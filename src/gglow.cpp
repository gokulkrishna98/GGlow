#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

int main() {
    std::cout << "hello\n";
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    try {
        auto module = torch::jit::load("/home/gokul/projects/GGlow/models/mobilenetv2.pt");
        module.dump(false, false, false);
    }catch(const c10::Error &e){
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }


    return 0;
}