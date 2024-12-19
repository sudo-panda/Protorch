#ifndef PROTORCH_PROTORCH_HPP
#define PROTORCH_PROTORCH_HPP

#include "llvm/IR/Module.h"
#include <llvm/ADT/SmallVector.h>

class ProTorch {
private:
  const char *m_model_path;

public:
  ProTorch(bool runtime = false);

  llvm::SmallVector<int, 3> getInstrCounts(const llvm::Function &Fn);

  void processOptInfo(const llvm::SmallVector<int, 3> &OptInfo);

  void callTorch();
};

#endif // PROTORCH_PROTORCH_HPP
