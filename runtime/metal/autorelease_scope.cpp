#include "metal/autorelease_scope.h"

namespace us4 {

std::string_view ToString(const AutoreleaseBoundaryKind kind) {
  switch (kind) {
  case AutoreleaseBoundaryKind::kNoop:
    return "noop";
  case AutoreleaseBoundaryKind::kObjectiveC:
    return "objective-c-autorelease";
  }
  return "noop";
}

ScopedAutoreleasePool::ScopedAutoreleasePool(const bool requested)
    : requested_(requested) {
#if defined(__APPLE__)
  if (requested_) {
    active_ = true;
    kind_ = AutoreleaseBoundaryKind::kObjectiveC;
  }
#endif
}

ScopedAutoreleasePool::~ScopedAutoreleasePool() = default;

bool ScopedAutoreleasePool::Requested() const { return requested_; }

bool ScopedAutoreleasePool::Active() const { return active_; }

AutoreleaseBoundaryKind ScopedAutoreleasePool::Kind() const { return kind_; }

} // namespace us4
