#include "GGlowDialect.h"
#include "GGlow.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/ValueRange.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <unordered_map>
#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

struct NNLayer {
  std::string name;
  std::string layer_name;
};
struct NNTensor {
  std::string debug_name;
  mlir::Value glow_value;
}; 

struct Graph {
  std::string name;
  torch::jit::Module pytorch_module;
  std::vector<torch::jit::Value*> inputs;
  std::vector<torch::jit::Value*> outputs;

  std::unordered_map<std::string, mlir::Value> const_tensors;
  std::unordered_map<std::string, std::function<mlir::Operation*(mlir::MLIRContext&, mlir::OpBuilder&, torch::jit::Node*)>> parse_layers;

  std::unordered_map<torch::jit::Value*, NNLayer> layer_info;  
  std::unordered_map<torch::jit::Value*, NNTensor> tensor_to_val;

  Graph(std::string name, torch::jit::Module module) : name(name), pytorch_module(module) {
    register_parsers("__torch__.torch.nn.modules.conv.Conv2d",
      std::bind(&Graph::parse_conv2d_layer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    register_parsers("__torch__.torch.nn.modules.batchnorm.BatchNorm2d", 
      std::bind(&Graph::parse_conv2d_layer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    register_parsers("__torch__.torch.nn.modules.activation.ReLU", 
      std::bind(&Graph::parse_conv2d_layer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  }


  void register_parsers(std::string key, std::function<mlir::Operation*(mlir::MLIRContext&, mlir::OpBuilder&, torch::jit::Node*)> fn);
  mlir::Type _create_f64tensor_type(mlir::MLIRContext &ctx, std::vector<int64_t> shape);
  mlir::Value _create_load_op_from_tensor(mlir::MLIRContext &ctx, mlir::OpBuilder &builder, const torch::Tensor &tensor, 
    std::string tensor_name);
  void _create_load_ops(mlir::MLIRContext &ctx, mlir::OpBuilder &builder);
  void _create_computation_ops(mlir::MLIRContext &ctx, mlir::OpBuilder &builder);
  void get_glow_mlir();

  mlir::Operation* parse_conv2d_layer(mlir::MLIRContext &ctx, mlir::OpBuilder& builder, torch::jit::Node* node);
  mlir::Operation* parse_bn2d_layer(mlir::MLIRContext &ctx, mlir::OpBuilder &builder, torch::jit::Node* node);
  mlir::Operation* parse_relu_layer(mlir::MLIRContext &ctx, mlir::OpBuilder &builder, torch::jit::Node* node);
};

mlir::Operation* Graph::parse_bn2d_layer(mlir::MLIRContext& ctx, mlir::OpBuilder& builder, torch::jit::Node* node){
  return nullptr;
}

mlir::Operation* Graph::parse_relu_layer(mlir::MLIRContext& ctx, mlir::OpBuilder& builder, torch::jit::Node* node){
  return nullptr;
}

mlir::Operation* Graph::parse_conv2d_layer(mlir::MLIRContext& ctx, mlir::OpBuilder& builder, torch::jit::Node* node){
  // params: input, weight, bias
  auto name = layer_info[node->inputs()[0]].name;
  auto conv_module = pytorch_module.attr(name).toModule();

  mlir::Value x = tensor_to_val[node->inputs()[1]].glow_value;
  mlir::Value bias = nullptr, weight = nullptr;
  for(const auto &param : conv_module.named_parameters()){
    auto param_id = name + "_" + param.name;
    if(param.name == "weight"){
      weight = const_tensors[param_id];
    }else if(param.name == "bias"){
      bias = const_tensors[param_id];
    }
  }

  // converting tuple to int64 vector (for conv2d params)
  auto convert_tuple2vec = [](c10::ivalue::Tuple &tuple) -> std::vector<int64_t> {
    std::vector<int64_t> res;
    for (const auto& element : tuple.elements()) {
        res.push_back(element.toInt());
    }
    return res;
  };

  // attrs: in_channels, out_channels, kernel_size, stride,
  // padding, dilation, groups, is_bias, padding_mode
  auto in_channels = conv_module.attr("in_channels").toInt();
  auto in_channels_attr = builder.getI64IntegerAttr(in_channels);

  auto out_channels = conv_module.attr("out_channels").toInt();
  auto out_channels_attr = builder.getI64IntegerAttr(out_channels);

  auto kernel_size = convert_tuple2vec(*conv_module.attr("kernel_size").toTuple());
  llvm::SmallVector<int64_t> ks_arr(kernel_size.begin(), kernel_size.end());
  auto kernel_size_attr = builder.getI64ArrayAttr(ks_arr);

  auto stride = convert_tuple2vec(*conv_module.attr("stride").toTuple());
  llvm::SmallVector<int64_t> stride_arr(stride.begin(), stride.end());
  auto stride_attr = builder.getI64ArrayAttr(stride_arr);

  auto padding = convert_tuple2vec(*conv_module.attr("padding").toTuple());
  llvm::SmallVector<int64_t> padding_arr(padding.begin(), padding.end());
  auto padding_attr = builder.getI64ArrayAttr(padding_arr);

  auto dilation = convert_tuple2vec(*conv_module.attr("dilation").toTuple());
  llvm::SmallVector<int64_t> dilation_arr(dilation.begin(), dilation.end());
  auto dilation_attr = builder.getI64ArrayAttr(dilation_arr);

  auto groups = conv_module.attr("groups").toInt();
  auto groups_attr = builder.getI64IntegerAttr(groups);

  auto has_bias = conv_module.attr("bias").isTensor();
  auto has_bias_attr = builder.getBoolAttr(has_bias);
  
  auto padding_mode = conv_module.attr("padding_mode").toString()->string();
  auto padding_mode_attr = builder.getStringAttr(padding_mode);

  // @todo : do shape inference
  auto result_type = _create_f64tensor_type(ctx, {mlir::ShapedType::kDynamic});
  auto conv2d_op = builder.create<mlir::gglow::Conv2dOp>(
    builder.getUnknownLoc(), 
    result_type, x, weight, bias,
    in_channels_attr, 
    out_channels_attr, 
    kernel_size_attr, 
    stride_attr, 
    padding_attr, 
    dilation_attr,
    groups_attr,
    has_bias_attr,
    padding_mode_attr
  );
  return conv2d_op.getOperation(); 
}

void Graph::register_parsers(std::string key, std::function<mlir::Operation*(mlir::MLIRContext&, mlir::OpBuilder&, torch::jit::Node*)> fn){
  parse_layers[key] = fn;
}

mlir::Type Graph::_create_f64tensor_type(mlir::MLIRContext &ctx, std::vector<int64_t> shape){
  auto f64type =  mlir::FloatType::getF64(&ctx);
  return mlir::RankedTensorType::get(shape, f64type);
}

mlir::Value Graph::_create_load_op_from_tensor(mlir::MLIRContext &ctx, mlir::OpBuilder &builder, const torch::Tensor &tensor, 
  std::string tensor_name){
  // calculating shape and creating tensor type
  auto shape = tensor.sizes();
  std::vector<int64_t> shape_vector(shape.begin(), shape.end());
  auto mlir_tensor_type = _create_f64tensor_type(ctx, shape_vector);
  auto load_op = builder.create<mlir::gglow::LoadOp>(builder.getUnknownLoc(), mlir_tensor_type, tensor_name);
  return load_op;
}

void Graph::_create_load_ops(mlir::MLIRContext &ctx, mlir::OpBuilder &builder){
  for(auto it : layer_info){
    auto module = pytorch_module.attr(it.second.name).toModule();
    for (const auto &param : module.named_parameters()) {
      auto param_name = param.name;
      const auto &param_tensor = param.value;
      std::string param_id = it.second.name + "_" + param_name;
      auto mlir_value = _create_load_op_from_tensor(ctx, builder, param_tensor, param_id);
      const_tensors[param_id] = mlir_value;
    }
  }
  return;
}

void Graph::_create_computation_ops(mlir::MLIRContext &ctx, mlir::OpBuilder &builder){
  auto graph = pytorch_module.get_method("forward").graph();
  for (const auto &node : graph->nodes()){
    if (node->kind() == c10::prim::CallMethod) {
      auto layer = layer_info[node->inputs()[0]];
      auto name = layer.name;
      auto layer_name = layer.layer_name;
      auto parse_fn = parse_layers[layer_name];
      auto op = parse_fn(ctx, builder, node);
      assert(node->outputs().size() == op->getNumResults());
      for(size_t i = 0; i < op->getNumResults(); i++){
        tensor_to_val[node->outputs()[i]] = {.debug_name = "tensor from -> " + name, .glow_value = op->getResult(i)};
      }
      // @todo: add further relu and bn2d
      break;
    }
  }
}

void Graph::get_glow_mlir() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::gglow::GlowDialect>();
  mlir::OpBuilder builder(&context);

  // creating function in mlir module
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

  // getting pytorch graph
  auto graph = pytorch_module.get_method("forward").graph();
  
  // creating func op
  // @todo, enable a way to accept multiple input tensors
  auto shape_info = pytorch_module.attr("input_tensor_info").toIntList();
  std::vector<int64_t> vec_shape_info(shape_info.begin(), shape_info.end());
  auto input_type = _create_f64tensor_type(context, vec_shape_info);
  auto funcType = builder.getFunctionType({input_type}, {});
  auto func_op = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(), "main_ch", funcType);
  module.push_back(func_op);
  mlir::Block* entry_block = func_op.addEntryBlock();
  builder.setInsertionPointToStart(entry_block); 

  // Adding argument to the func_op and map it to glow_value;
  assert(graph->inputs().size() - 1 == entry_block->getNumArguments());
  for(size_t i=1; i<graph->inputs().size(); i++){
    auto input_value = entry_block->getArgument(i-1);
    tensor_to_val[graph->inputs()[i]] = { .debug_name = "tensor from input " + std::to_string(0), .glow_value = input_value};
  }
  _create_load_ops(context, builder);

  _create_computation_ops(context, builder);
  // load all the constant tensor (weights and biases)
  module.print(llvm::outs());
  return;
}

void create_graph(std::string path){
  auto module = torch::jit::load(path);
  std::shared_ptr<torch::jit::Graph> graph = module.get_method("forward").graph();
  std::string module_name = graph->inputs()[0]->type()->str();
  Graph computation(module_name, module);

  // @todo : do some of this stuff in the constructor
  // parsing layer info
  for (const auto &node : graph->nodes()) {
    if(node->kind() == c10::prim::GetAttr){
      std::string name = node->s(torch::jit::attr::name);
      std::string layer_name = !node->outputs().empty() ? 
        node->outputs()[0]->type()->str() : "error";
      computation.layer_info[node->outputs()[0]] = {name, layer_name};
    }
  }

  for(size_t i=0; i<graph->inputs().size(); i++){
    computation.inputs.push_back(graph->inputs()[i]);
  }

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