#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <llvm/ADT/SmallVector.h>
#include <torch/script.h>

#include <nlohmann/json.hpp>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

#include "protorch.hpp"

#include "config.h"

constexpr int EMBED_SIZE = 64;

ProTorch::ProTorch(bool runtime) {
  if (runtime)
    m_model_path = "/usr/WS2/kundu1/RT_Tuner/examples/ts-ex/my_module_model.pt";
  else
    m_model_path = "/usr/WS2/kundu1/RT_Tuner/examples/ts-ex/my_module_model.pt";
}

llvm::SmallVector<int, 3> ProTorch::getInstrCounts(const llvm::Function &func) {
  int n_instrs = 0, n_blocks = 0, n_fp_instrs = 0;
  for (auto &b_blk : func) {
    n_blocks++;
    for (auto &instr : b_blk) {
      n_instrs++;
      for (auto *val : instr.operand_values()) {
        if (val->getType()->isFPOrFPVectorTy()) {
          n_fp_instrs++;
          break;
        }
      }
    }
  }
  std::cout << "Sending OptInfo : " << n_instrs << ", " << n_blocks << ", "
            << n_fp_instrs;

  return llvm::SmallVector<int, 3>{n_instrs, n_blocks, n_fp_instrs};
}

std::vector<std::vector<double>> ProTorch::getEmbeds(const std::string &BCFile, const std::vector<std::string> &FnNames) {
  std::string command = "python __PROJECT_DIR__/python/get_func_embed.py -i " + BCFile + " -f " + FnNames[0];
  for (auto it = FnNames.begin() + 1; it != FnNames.end(); it++) {
    command += "," + *it;
  }

  std::cout << command << std::endl;

  std::system(command.c_str());
  if (fp == NULL) {
    std::cerr << "Failed to open pipe." << std::endl;
    exit(1);
  }

  std::filesystem::path filePath = BCFile;
  filePath.replace_extension(".json");

  std::ifstream inputFile(filePath);
  std::cout << std::filesystem::absolute(filePath) << std::endl;
  if (!inputFile.is_open()) {
    std::cerr << "Error opening file!" << std::endl;
    exit(1);
  }

  nlohmann::json j;
  inputFile >> j;
  inputFile.close(); // Close the file when done

  std::vector<std::vector<double>> Embeds;
  for (const std::string &FnName : FnNames) {
    const auto &Embed = j[FnName].get<std::vector<double>>();

    for (double Ele : Embed)
      std::cout << Ele << ", ";
    std::cout << std::endl;

    Embeds.push_back(Embed);
  }
  pclose(fp);


  return Embeds;
}

void ProTorch::callTorch() {
  torch::jit::script::Module module;
  try {
    torch::Tensor inp = torch::rand({20});
    module = torch::jit::load(m_model_path);
    torch::Tensor out = module.forward({inp}).toTensor();
    std::cout << "Input Tensor Size: " << inp.size(0) << std::endl;
    std::cout << "Output Tensor Size: " << out.size(0) << std::endl;
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
  }
}

void ProTorch::processOptInfo(const llvm::SmallVector<int, 3> &OptInfo) {
  std::cout << "Received OptInfo : " << OptInfo[0] << ", " << OptInfo[1] << ", "
            << OptInfo[2] << std::endl;

  callTorch();
}


void ProTorch::processEmbed(const std::vector<double> &Embed) {
  std::cout << "Received Embed : ";
  for (double Ele : Embed)
       std::cout << Ele << ", ";

  std::cout << std::endl;

  callTorch();
}
