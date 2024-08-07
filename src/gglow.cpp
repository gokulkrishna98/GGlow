#include <torch/script.h>
#include <torch/torch.h>

#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>

auto print_model_details(std::string model_path) -> void
{
    // layer dict
    std::unordered_map<std::string, int> layer_dict;

    // reading the module
    auto module = torch::jit::load(model_path);

    // dfs traversal using recursive lambda
    auto traverse_graph =
        [&layer_dict](auto&& traverse_graph,
            const torch::jit::Module& _module) -> void {
        for (const auto& child : _module.named_children()) {
            auto layertype_name = child.value.type()->name()->name();
            layer_dict[layertype_name] += 1;
            traverse_graph(traverse_graph, child.value);
        }
    };

    traverse_graph(traverse_graph, module);

    // print the number of layers
    std::printf("Number of unique Layers %ld\n", layer_dict.size());

    // print the unique layer names
    std::printf("The Dict info: \n");
    std::printf("|%20s|%5s|\n", "LayerName", "Count");
    std::printf("----------------------------\n");
    for (auto layer : layer_dict) {
        std::printf("|%20s|%5s|\n", layer.first.c_str(), std::to_string(layer.second).c_str());
    }
}

auto main() -> int
{
    try {
        print_model_details("/home/gokul/projects/GGlow/models/resnet18.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}