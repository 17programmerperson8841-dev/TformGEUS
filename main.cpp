#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <cstdint>
#include <stdio.h>
#include <tensorflow/c/c_api.h>


const int ncategories = 4;
const char* Labels[] = {"GREAT", "EVIL", "UGLY", "SHINY"};

using std::string;
TF_Tensor* create_tensor_string(const char* str){
  //Because They Just felt like it
  const size_t string_size = 24;

  int64_t dims[] = {1, 1};
  int rank = 2;
  //MAKES A VECTOR STRING THAT UHH TECHNICALLY NOT MAKE BUT I DONT WANT TO EXPLAIN SORRY
  TF_Tensor* tensor = TF_AllocateTensor(TF_STRING, dims, rank, string_size);
  void* data = TF_TensorData(tensor);
  TF_TString_Init((TF_TString*)data);
  //PUTS STRING IN TO MAKE IT A TENSOR
  TF_TString_Copy((TF_TString*)data, str, strlen(str));
  return tensor;
}
int wordCount(const string& sentence){
  std::stringstream thing(sentence);
  string word;
  int i = 0;
  while (thing >> word){
    i++;
  }
  return i;
} 
int main() {
  //This just loads what tensorflow needs
  TF_Status* status = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();
  TF_SessionOptions* opt = TF_NewSessionOptions();
  //Now it gets the thing it actually needs
  const char* dir = "./TformGEUS";
  //It makes the AI your servant
  const char* tags[] = {"serve"};
  //NOW THIS IS THE FUN PART
  TF_Session* session = TF_LoadSessionFromSavedModel(opt, nullptr, dir, tags, 1, graph, nullptr, status);
  //WHY AM I STILL IN CAPSLOCK

  if (TF_GetCode(status) != TF_OK){
    std::cerr << "Hi I'm TformGEUS and——AHHHHHHHHHH IM EXPLODINGGGGG I FAILED!" << TF_Message(status) << std::endl;
    return 1;
  }

  //GET USER INPUT
  std::cout << "Hey, Welcome To TformGEUS, the only AI you will ever need!\n";
  std::this_thread::sleep_for(std::chrono::seconds(2));
  int length = 0;
  string input;
  do{
    std::cout << "Type a description and TformGEUS will label it as GREAT EVIL UGLY or SHINY! Your desciption must be less than 10 words\n";
    std::getline(std::cin, input);
    length = wordCount(input);
    if (length > 9){
      std::cout << "Please make sure your description is less than 10 words\n";
    }
    if (length == 0){
      std::cout << "Make sure you actually type something\n";
    }
  }while(length > 9 || length == 0);

  std::cout << "TformGEUS is thinking... Give him a moment\n";

  TF_Tensor* input_tensor = create_tensor_string(input.c_str());
  TF_Output input_o = {TF_GraphOperationByName(graph, "serve_text_input"), 0};
  TF_Output output_o = {TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};
  TF_Tensor* output_tensor = nullptr;
  if (input_o.oper == nullptr || output_o.oper == nullptr){
    std::cerr << "INPUT OR OUTPUT NODES CANNOT BE FOUND AHHHHHHH IM EXPLODINGGGGG" << std::endl;
    return 1;
  }
  //Ok let's make a session so we can do stuff to do stuff which does stuff
  TF_SessionRun(
    session, nullptr, &input_o, &input_tensor, 1, &output_o, &output_tensor, 1, nullptr, 0, nullptr, status
  );
  //That thing up there may seem complicated but it is soooo easy to do
  if (TF_GetCode(status) == TF_OK){
    float* scores = (float*)TF_TensorData(output_tensor);

    int win_index = 0;
    for (int i = 1; i < ncategories; i++){
      if (scores[i] > scores[win_index]){
        win_index = i;
      }
    }
    std::cout << "I have made my descision. " << input << " is " << Labels[win_index] << std::endl;
  }
  //Clean up Clean up
  TF_DeleteTensor(input_tensor);
  TF_DeleteTensor(output_tensor);
  TF_DeleteGraph(graph);
  TF_DeleteSession(session, status);
  TF_DeleteStatus(status);
  TF_DeleteSessionOptions(opt);
  return 0;
}