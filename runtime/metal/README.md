# Metal

Home for the Metal execution path.

What already exists:

- `command_queue.{h,cpp}` records dispatch intent for `matmul`, `softmax`, and
  `rmsnorm`
- `command_queue.{h,cpp}` now also surfaces an initialization profile
  (`unavailable -> device-ready -> queue-ready`) and records Objective-C
  autorelease-boundary intent for each dispatch
- `kernel_library.{h,cpp}` exposes a native metadata catalog for the Metal
  kernel set
- `kernels/{matmul,softmax,rmsnorm}.metal` already exist as versioned source
  files inside the repo
- `dense_dispatch.{h,cpp}` materializes the current dense dispatch sequence
  (`matmul -> softmax -> rmsnorm`)
- queue availability is driven by `HardwareProbeResult::hasMetal`
- shared allocations from `UnifiedAllocator` can be attached to dispatch
  records

What is still missing:

- real `MTLDevice` and command-queue ownership
- real dispatch wrappers that compile and run those `.metal` kernels during
  generation
