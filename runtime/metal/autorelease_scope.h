#pragma once

#include <string_view>

namespace us4 {

enum class AutoreleaseBoundaryKind {
  kNoop,
  kObjectiveC,
};

std::string_view ToString(AutoreleaseBoundaryKind kind);

class ScopedAutoreleasePool {
public:
  explicit ScopedAutoreleasePool(bool requested = false);
  ~ScopedAutoreleasePool();

  bool Requested() const;
  bool Active() const;
  AutoreleaseBoundaryKind Kind() const;

private:
  bool requested_ = false;
  bool active_ = false;
  AutoreleaseBoundaryKind kind_ = AutoreleaseBoundaryKind::kNoop;
};

} // namespace us4
