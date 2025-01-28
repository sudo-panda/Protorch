#include <vector>

#include "protorch.hpp"

int main() {
  ProTorch PT;
  const std::vector<std::vector<double>> Embeds =
    PT.getEmbeds("/usr/WS2/kundu1/RT_Tuner/mltraining/c_src/fib.ll", {"Fib", "a"});

  ProTorch PT_rt(/* runtime = */ true);
  PT_rt.processEmbed(Embeds[0]);
}
