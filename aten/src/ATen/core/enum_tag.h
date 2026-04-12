#pragma once

// at::Tag is generated into torch/headeronly so it can be used without
// linking libtorch.  This forwarding header keeps the traditional
// <ATen/core/enum_tag.h> include path working.
#include <torch/headeronly/core/enum_tag.h>
