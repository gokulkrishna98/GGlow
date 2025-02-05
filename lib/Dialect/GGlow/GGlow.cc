#include "GGlowDialect.h"
#include "GGlow.h"
#include <cstdint>
#include <cstdio>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Value.h>
#include <torch/torch.h>
#include <torch/script.h>

struct NNLayer {
  std::string name;
  std::string layer_name;
};
struct NNTensor {
  std::string debug_name;
  // mlir::Value glow_value;
}; 
struct Graph {
  std::string name;
  torch::jit::Module pytorch_module;
  std::vector<torch::jit::Value*> inputs;
  std::vector<torch::jit::Value*> outputs;

  std::unordered_map<torch::jit::Value*, NNLayer> layer_info;  
  std::unordered_map<torch::jit::Value*, NNTensor> tensor_to_val;

  Graph(std::string name, torch::jit::Module module) : name(name), 
    pytorch_module(module) {}
  void get_glow_mlir();
};

mlir::Type _create_f64tensor_type(mlir::MLIRContext &ctx, std::vector<int64_t> shape){
  auto f64type =  mlir::FloatType::getF64(&ctx);
  return mlir::RankedTensorType::get(shape, f64type);
}

void Graph::get_glow_mlir() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::gglow::GlowDialect>();

  mlir::OpBuilder builder(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
  
  // creating function in mlir module
  auto shape_info = pytorch_module.attr("input_tensor_info").toIntList();
  std::vector<int64_t> vec_shape_info(shape_info.begin(), shape_info.end());
  auto input_type = _create_f64tensor_type(context, vec_shape_info);
  auto funcType = builder.getFunctionType({input_type}, {});
  auto func_op = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(), "main_ch", funcType);
  module.push_back(func_op);
  mlir::Block* entry_block = func_op.addEntryBlock();
  builder.setInsertionPointToStart(entry_block); 
  auto input_value = entry_block->getArgument(0);

  auto unknown_type = _create_f64tensor_type(context, {mlir::ShapedType::kDynamic});
  builder.create<mlir::gglow::LoadOp>(builder.getUnknownLoc(), unknown_type, "ggwp");
  module.print(llvm::outs());
  return;
}

void create_graph(std::string path){
  auto module = torch::jit::load(path);
  std::shared_ptr<torch::jit::Graph> graph = module.get_method("forward").graph();
  std::string module_name = graph->inputs()[0]->type()->str();
  Graph computation(module_name, module);

  // TODO: handle the case of multiple outputs from operation in pytorch ir 
  for (const auto &node : graph->nodes()) {
    if(node->kind() == c10::prim::GetAttr){
      std::string name = node->s(torch::jit::attr::name);
      std::string layer_name = !node->outputs().empty() ? 
        node->outputs()[0]->type()->str() : "error";
      computation.layer_info[node->outputs()[0]] = {name, layer_name};
    }
  }

  for(int i=0; i<graph->inputs().size(); i++){
    computation.inputs.push_back(graph->inputs()[i]);
    computation.tensor_to_val[graph->inputs()[i]] = {"tensor from input " 
      + std::to_string(i)};
  }

  for (const auto &node : graph->nodes()){
    if(node->kind() == c10::prim::CallMethod){
      if(node->outputs().size() == 1){
        computation.tensor_to_val[node->outputs()[0]] = {("tensor from ->" 
          + computation.layer_info[node->inputs()[0]].name)};
      }
    }
  }
  // printf("printing both the maps in the graph\n");
  // for(auto it : computation.layer_info){
  //   printf("%s -- %s\n", it.second.name.c_str(), it.second.layer_name.c_str());
  // }

  // for(auto it : computation.tensor_to_val){
  //   printf("%s\n", it.second.debug_name.c_str());
  // }

  computation.get_glow_mlir();
  return;
}

void run_model(std::string path){	
  create_graph(path);
  return;
}

void test_GGlow(){
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