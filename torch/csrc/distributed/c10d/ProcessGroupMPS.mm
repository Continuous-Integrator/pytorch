#ifdef USE_C10D_MPS

#include <torch/csrc/distributed/c10d/ProcessGroupMPS.hpp>
#include <torch/csrc/distributed/c10d/JACCLTransport.h>

#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/util/irange.h>

namespace c10d {

namespace {

void* mpsHostPtr(const at::Tensor& tensor) {
  if (!tensor.device().is_mps() || !tensor.is_contiguous()) {
    return nullptr;
  }
  id<MTLBuffer> buf = at::native::mps::getMTLBufferStorage(tensor);
  if (buf == nil) {
    return nullptr;
  }
  void* base = [buf contents];
  if (base == nullptr) {
    return nullptr;
  }
  return static_cast<char*>(base) +
      tensor.storage_offset() * tensor.itemsize();
}

std::string probeRDMADevice() {
#if HAVE_JACCL
  if (!jaccl::isAvailable()) {
    return {};
  }
  try {
    int numDevices = 0;
    auto devices = jaccl::ibv().getDeviceList(&numDevices);
    const char* deviceOverride = std::getenv("JACCL_DEVICE");
    std::string chosen;
    for (int i = 0; i < numDevices; i++) {
      std::string name = jaccl::ibv().getDeviceName(devices[i]);
      if (deviceOverride && name != deviceOverride) {
        continue;
      }
      auto ctx = jaccl::ibv().openDevice(devices[i]);
      if (!ctx) {
        continue;
      }
      auto pd = jaccl::ibv().allocPd(ctx);
      if (pd) {
        jaccl::ibv().deallocPd(pd);
        jaccl::ibv().closeDevice(ctx);
        chosen = name;
        break;
      }
      jaccl::ibv().closeDevice(ctx);
    }
    jaccl::ibv().freeDeviceList(devices);
    return chosen;
  } catch (...) {
    return {};
  }
#else
  return {};
#endif
}

} // namespace

ProcessGroupMPS::WorkMPS::WorkMPS(
    OpType opType,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors)
    : Work(-1, opType, profilingTitle, inputTensors) {
  future_ = c10::make_intrusive<at::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
}

bool ProcessGroupMPS::WorkMPS::isCompleted() {
  return Work::isCompleted();
}

bool ProcessGroupMPS::WorkMPS::isSuccess() const {
  return Work::isSuccess();
}

bool ProcessGroupMPS::WorkMPS::wait(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!completed_) {
    if (timeout == kNoTimeout) {
      cv_.wait(lock, [this] { return completed_; });
    } else {
      cv_.wait_for(lock, timeout, [this] { return completed_; });
    }
  }
  if (exception_) {
    std::rethrow_exception(exception_);
  }
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future>
ProcessGroupMPS::WorkMPS::getFuture() {
  return future_;
}

void ProcessGroupMPS::WorkMPS::finishWork() {
  future_->markCompleted(c10::IValue(outputTensors_));
  finish();
}

void ProcessGroupMPS::WorkMPS::finishWorkError(
    const std::exception_ptr& eptr) {
  future_->setError(eptr);
  finishAndThrow(eptr);
}

ProcessGroupMPS::Options::Options(std::chrono::milliseconds timeout)
    : Backend::Options(MPS_BACKEND_NAME, timeout) {}

ProcessGroupMPS::ProcessGroupMPS(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(store),
      options_(std::move(options)) {
#if !HAVE_JACCL
  TORCH_CHECK(
      false,
      "ProcessGroupMPS requires Apple Thunderbolt RDMA (librdma/infiniband verbs) "
      "but this build does not include JACCL support. Use the 'gloo' backend for "
      "CPU-based distributed training.");
#else
  std::string firstDevice = probeRDMADevice();

  std::vector<uint8_t> flag(1, firstDevice.empty() ? 0 : 1);
  store_->set("mps_pg/rdma_avail/" + std::to_string(rank), flag);

  std::vector<std::string> keys;
  keys.reserve(size);
  for (int r = 0; r < size; r++) {
    keys.push_back("mps_pg/rdma_avail/" + std::to_string(r));
  }
  store_->wait(keys);

  int missingRank = -1;
  for (int r = 0; r < size; r++) {
    auto data = store_->get("mps_pg/rdma_avail/" + std::to_string(r));
    if (data.empty() || data[0] == 0) {
      missingRank = r;
      break;
    }
  }

  TORCH_CHECK(
      missingRank < 0,
      "ProcessGroupMPS requires Apple Thunderbolt RDMA on every rank, but rank ",
      missingRank,
      " could not allocate a protection domain (ibv_alloc_pd failed on every "
      "rdma_en* device). Check your Thunderbolt 5 cable and network setup, or "
      "use the 'gloo' backend for CPU-based distributed training.");

  std::vector<std::string> deviceNames(size);
  for (int i = 0; i < size; i++) {
    deviceNames[i] = (i == rank) ? "" : firstDevice;
  }

  std::string coordAddr;
  if (rank == 0) {
    const char* masterAddr = std::getenv("MASTER_ADDR");
    const char* masterPortEnv = std::getenv("MASTER_PORT");
    int basePort = masterPortEnv ? std::atoi(masterPortEnv) : 29500;
    std::string host = masterAddr ? masterAddr : "127.0.0.1";
    coordAddr = host + ":" + std::to_string(basePort + 1);
    store_->set(
        "mps_pg/jaccl_coord",
        std::vector<uint8_t>(coordAddr.begin(), coordAddr.end()));
  } else {
    auto data = store_->get("mps_pg/jaccl_coord");
    coordAddr = std::string(data.begin(), data.end());
  }

  jacclTransport_ = std::make_unique<jaccl::JACCLTransport>(
      rank, size, coordAddr.c_str(), deviceNames);
  TORCH_WARN(
      "ProcessGroupMPS: using JACCL RDMA transport on ", firstDevice);

  workerThread_ = std::thread(&ProcessGroupMPS::runLoop, this);
#endif
}

ProcessGroupMPS::~ProcessGroupMPS() {
  {
    std::lock_guard<std::mutex> lock(workMutex_);
    stop_ = true;
  }
  workCV_.notify_one();
  if (workerThread_.joinable()) {
    workerThread_.join();
  }
}

void ProcessGroupMPS::runLoop() {
  while (true) {
    std::function<void()> fn;
    {
      std::unique_lock<std::mutex> lock(workMutex_);
      workCV_.wait(lock, [this] { return stop_ || !workQueue_.empty(); });
      if (stop_ && workQueue_.empty())
        return;
      fn = std::move(workQueue_.front());
      workQueue_.pop_front();
    }
    fn();
  }
}

void ProcessGroupMPS::enqueue(std::function<void()> fn) {
  {
    std::lock_guard<std::mutex> lock(workMutex_);
    workQueue_.push_back(std::move(fn));
  }
  workCV_.notify_one();
}

at::Tensor ProcessGroupMPS::syncAndCopyToCPU(const at::Tensor& tensor) {
  at::mps::getDefaultMPSStream()->synchronize(
      at::mps::SyncType::COMMIT_AND_WAIT);
  return tensor.to(at::kCPU).contiguous();
}

void ProcessGroupMPS::copyToMPS(
    const at::Tensor& cpuTensor,
    at::Tensor& mpsTensor) {
  mpsTensor.copy_(cpuTensor);
  at::mps::getDefaultMPSStream()->synchronize(
      at::mps::SyncType::COMMIT_AND_WAIT);
}

c10::intrusive_ptr<Work> ProcessGroupMPS::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1,
      "ProcessGroupMPS::allreduce: only single tensor supported");

  auto& tensor = tensors[0];
  auto work = c10::make_intrusive<WorkMPS>(
      OpType::ALLREDUCE,
      "mps:allreduce",
      std::optional<std::vector<at::Tensor>>({tensor}));
  work->outputTensors_ = {tensor};

  auto fn = [this, tensor, reduceOp = opts.reduceOp, work]() mutable {
    try {
      if (void* hostPtr = mpsHostPtr(tensor)) {
        at::mps::getDefaultMPSStream()->synchronize(
            at::mps::SyncType::COMMIT_AND_WAIT);
        jacclTransport_->allReduce(
            hostPtr,
            static_cast<size_t>(tensor.nbytes()),
            tensor.element_size(),
            tensor.scalar_type(),
            reduceOp);
      } else {
        auto cpuTensor = syncAndCopyToCPU(tensor);
        jacclTransport_->allReduce(
            cpuTensor.data_ptr(),
            static_cast<size_t>(cpuTensor.nbytes()),
            cpuTensor.element_size(),
            cpuTensor.scalar_type(),
            reduceOp);
        copyToMPS(cpuTensor, tensor);
      }
      work->finishWork();
    } catch (...) {
      work->finishWorkError(std::current_exception());
    }
  };

  enqueue(std::move(fn));
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupMPS::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1,
      "ProcessGroupMPS::broadcast: only single tensor supported");

  auto& tensor = tensors[0];
  auto work = c10::make_intrusive<WorkMPS>(
      OpType::BROADCAST,
      "mps:broadcast",
      std::optional<std::vector<at::Tensor>>({tensor}));
  work->outputTensors_ = {tensor};

  auto fn = [this, tensor, rootRank = opts.rootRank, work]() mutable {
    try {
      if (void* hostPtr = mpsHostPtr(tensor)) {
        at::mps::getDefaultMPSStream()->synchronize(
            at::mps::SyncType::COMMIT_AND_WAIT);
        jacclTransport_->broadcast(
            hostPtr, static_cast<size_t>(tensor.nbytes()), rootRank);
      } else {
        auto cpuTensor = syncAndCopyToCPU(tensor);
        jacclTransport_->broadcast(
            cpuTensor.data_ptr(),
            static_cast<size_t>(cpuTensor.nbytes()),
            rootRank);
        copyToMPS(cpuTensor, tensor);
      }
      work->finishWork();
    } catch (...) {
      work->finishWorkError(std::current_exception());
    }
  };

  enqueue(std::move(fn));
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupMPS::barrier(
    const BarrierOptions& /*opts*/) {
  auto work = c10::make_intrusive<WorkMPS>(OpType::BARRIER, "mps:barrier");

  auto fn = [this, work]() {
    try {
      jacclTransport_->barrier();
      work->finishWork();
    } catch (...) {
      work->finishWorkError(std::current_exception());
    }
  };

  enqueue(std::move(fn));
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupMPS::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int /*tag*/) {
  TORCH_CHECK(
      tensors.size() == 1,
      "ProcessGroupMPS::send: only single tensor supported");

  auto& tensor = tensors[0];
  auto work = c10::make_intrusive<WorkMPS>(OpType::SEND, "mps:send");

  auto fn = [this, tensor, dstRank, work]() mutable {
    try {
      if (void* hostPtr = mpsHostPtr(tensor)) {
        at::mps::getDefaultMPSStream()->synchronize(
            at::mps::SyncType::COMMIT_AND_WAIT);
        jacclTransport_->send(
            hostPtr, static_cast<size_t>(tensor.nbytes()), dstRank);
      } else {
        auto cpuTensor = syncAndCopyToCPU(tensor);
        jacclTransport_->send(
            cpuTensor.data_ptr(),
            static_cast<size_t>(cpuTensor.nbytes()),
            dstRank);
      }
      work->finishWork();
    } catch (...) {
      work->finishWorkError(std::current_exception());
    }
  };

  enqueue(std::move(fn));
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupMPS::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int /*tag*/) {
  TORCH_CHECK(
      tensors.size() == 1,
      "ProcessGroupMPS::recv: only single tensor supported");

  auto& tensor = tensors[0];
  auto work = c10::make_intrusive<WorkMPS>(OpType::RECV, "mps:recv");
  work->outputTensors_ = {tensor};

  auto fn = [this, tensor, srcRank, work]() mutable {
    try {
      if (void* hostPtr = mpsHostPtr(tensor)) {
        at::mps::getDefaultMPSStream()->synchronize(
            at::mps::SyncType::COMMIT_AND_WAIT);
        jacclTransport_->recv(
            hostPtr, static_cast<size_t>(tensor.nbytes()), srcRank);
      } else {
        auto cpuTensor = syncAndCopyToCPU(tensor);
        jacclTransport_->recv(
            cpuTensor.data_ptr(),
            static_cast<size_t>(cpuTensor.nbytes()),
            srcRank);
        copyToMPS(cpuTensor, tensor);
      }
      work->finishWork();
    } catch (...) {
      work->finishWorkError(std::current_exception());
    }
  };

  enqueue(std::move(fn));
  return work;
}

} // namespace c10d

#endif // USE_C10D_MPS
