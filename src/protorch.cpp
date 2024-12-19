#include <llvm/ADT/SmallVector.h>
#include <torch/script.h>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include <cstdio>

#include "protorch.hpp"

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

  return llvm::SmallVector<int, 3>{n_instrs, n_blocks, n_fp_instrs};
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
