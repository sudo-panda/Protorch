#ifndef PROTORCH_PROTORCH_HPP
#define PROTORCH_PROTORCH_HPP

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Function.h>
#include <vector>

class ProTorch {
private:
  const char *m_model_path;

public:
  ProTorch(bool runtime = false);

  llvm::SmallVector<int, 3> getInstrCounts(const llvm::Function &Fn);

  std::vector<std::vector<double>> getEmbeds(const std::string &BCFile, const std::vector<std::string> &FnNames);

  void processOptInfo(const llvm::SmallVector<int, 3> &OptInfo);

  void processEmbed(const std::vector<double> &Embed);

  void callTorch();
};

#endif // PROTORCH_PROTORCH_HPP
