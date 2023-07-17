#include "CommonInclude.hpp"

namespace adso
{

struct SelectCfg {
  int sel_level{1};       // pyramid level for initial selection
  int cell_size{16};      // cell size in top level
  int min_grad{8};        // mininum grad to be selected
  int max_grad{64};       // wont keep searching if we found pix > max_grad
  int nms_size{1};        // nms size when creating mask
  double min_ratio{0.0};  // decrease min_grad when ratio < min_ratio
  double max_ratio{1.0};  // increase min_grad when ratio > max_ratio
  int reselect{3};   // reselect if first round is two low

  std::string Repr() const;
  void Check() const;
};


class PixelSelector
{
private:
    SelectCfg cfg_;

};

} // namespace adso