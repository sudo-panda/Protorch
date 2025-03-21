#include <vector>
#include "config.h"
#include "protorch.hpp"

int main() {
  ProTorch PT;
  std::vector<std::vector<double>> Embeds =
    PT.getEmbeds(PROTORCH_DIR "tests/data/fib.ll", {"Fib", "a"});

  ProTorch PT_rt(/* runtime = */ true);
  PT_rt.processEmbed(Embeds[0]);
}
