#pragma once

#include <string_view>
#include <vector>

#include "core/ius4v6_adapter.h"

namespace us4 {

const IUS4V6Adapter* FindAdapterByModel(std::string_view modelName);
std::vector<const IUS4V6Adapter*> ListAdapters();

}  // namespace us4
