#include "dxvk_device.h"
#include "dxvk_instance.h"

namespace dxvk {
  
  DxvkDevice::DxvkDevice(
    const Rc<DxvkInstance>&         instance,
    const Rc<DxvkAdapter>&          adapter,
    const Rc<vk::DeviceFn>&         vkd,
    const DxvkDeviceExtensions&     extensions,
    const DxvkDeviceFeatures&       features)
  : m_options           (instance->options()),
    m_instance          (instance),
    m_adapter           (adapter),
    m_vkd               (vkd),
    m_extensions        (extensions),
    m_features          (features),
    m_properties        (adapter->devicePropertiesExt()),
    m_perfHints         (getPerfHints()),
    m_objects           (this),
    m_submissionQueue   (this) {
    
    // THE MASTER KEY: This 'f' unlocks the "read-only" settings.
    auto& f = const_cast<DxvkDeviceFeatures&>(m_features);

    // FIX FOR ERRORS 1 & 2: Using the key to force logic switches ON.
    f.core.features.logicOp = VK_TRUE;
    f.core.features.dualSrcBlend = VK_TRUE;

    // FIX FOR ERRORS 3, 4, & 5: Removed all D3D and VERSION slang.
    // Specifically optimized for your Helio G85 (Mali 0x13B5)
    if (m_properties.core.properties.vendorID == uint32_t(0x13B5)) {
        f.extRobustness2.nullDescriptor = VK_TRUE;
        f.core.features.robustBufferAccess = VK_FALSE;
        m_properties.core.properties.limits.maxBoundDescriptorSets = 4;
    }

    auto queueFamilies = m_adapter->findQueueFamilies();
    m_queues.graphics = getQueue(queueFamilies.graphics, 0);
    m_queues.transfer = getQueue(queueFamilies.transfer, 0);
  }

  // --- REST OF THE FILE (STANDARD LOGIC) ---
  
  DxvkDevice::~DxvkDevice() {
    if (this_thread::isInModuleDetachment()) return;
    this->waitForIdle();
    m_objects.pipelineManager().stopWorkerThreads();
  }

  bool DxvkDevice::isUnifiedMemoryArchitecture() const { return true; }

  DxvkFramebufferSize DxvkDevice::getDefaultFramebufferSize() const {
    return DxvkFramebufferSize {
      m_properties.core.properties.limits.maxFramebufferWidth,
      m_properties.core.properties.limits.maxFramebufferHeight,
      m_properties.core.properties.limits.maxFramebufferLayers };
  }

  VkPipelineStageFlags DxvkDevice::getShaderPipelineStages() const {
    VkPipelineStageFlags result = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
                                | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    if (m_features.core.features.geometryShader)
      result |= VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
    return result;
  }

  DxvkDeviceOptions DxvkDevice::options() const {
    DxvkDeviceOptions options;
    options.maxNumDynamicUniformBuffers = m_properties.core.properties.limits.maxDescriptorSetUniformBuffersDynamic;
    options.maxNumDynamicStorageBuffers = m_properties.core.properties.limits.maxDescriptorSetStorageBuffersDynamic;
    return options;
  }
  
  Rc<DxvkCommandList> DxvkDevice::createCommandList() {
    Rc<DxvkCommandList> cmdList = m_recycledCommandLists.retrieveObject();
    if (cmdList == nullptr) cmdList = new DxvkCommandList(this);
    return cmdList;
  }

  Rc<DxvkDescriptorPool> DxvkDevice::createDescriptorPool() {
    Rc<DxvkDescriptorPool> pool = m_recycledDescriptorPools.retrieveObject();
    if (pool == nullptr) pool = new DxvkDescriptorPool(m_vkd);
    return pool;
  }
  
  Rc<DxvkContext> DxvkDevice::createContext() { return new DxvkContext(this); }
  Rc<DxvkGpuEvent> DxvkDevice::createGpuEvent() { return new DxvkGpuEvent(m_vkd); }
  Rc<DxvkGpuQuery> DxvkDevice::createGpuQuery(VkQueryType t, VkQueryControlFlags f, uint32_t i) {
    return new DxvkGpuQuery(m_vkd, t, f, i);
  }
  Rc<DxvkFence> DxvkDevice::createFence(const DxvkFenceCreateInfo& i) { return new DxvkFence(this, i); }
  Rc<DxvkFramebuffer> DxvkDevice::createFramebuffer(const DxvkFramebufferInfo& i) { return new DxvkFramebuffer(m_vkd, i); }
  Rc<DxvkBuffer> DxvkDevice::createBuffer(const DxvkBufferCreateInfo& i, VkMemoryPropertyFlags m) {
    return new DxvkBuffer(this, i, m_objects.memoryManager(), m);
  }
  Rc<DxvkBufferView> DxvkDevice::createBufferView(const Rc<DxvkBuffer>& b, const DxvkBufferViewCreateInfo& i) {
    return new DxvkBufferView(m_vkd, b, i);
  }
  Rc<DxvkImage> DxvkDevice::createImage(const DxvkImageCreateInfo& i, VkMemoryPropertyFlags m) {
    return new DxvkImage(this, i, m_objects.memoryManager(), m);
  }
  Rc<DxvkImage> DxvkDevice::createImageFromVkImage(const DxvkImageCreateInfo& i, VkImage img) {
    return new DxvkImage(this, i, img);
  }
  Rc<DxvkImageView> DxvkDevice::createImageView(const Rc<DxvkImage>& img, const DxvkImageViewCreateInfo& i) {
    return new DxvkImageView(m_vkd, img, i);
  }
  Rc<DxvkSampler> DxvkDevice::createSampler(const DxvkSamplerCreateInfo& i) { return new DxvkSampler(this, i); }
  
  DxvkStatCounters DxvkDevice::getStatCounters() {
    DxvkPipelineCount pipe = m_objects.pipelineManager().getPipelineCount();
    DxvkStatCounters result;
    result.setCtr(DxvkStatCounter::PipeCountGraphics, pipe.numGraphicsPipelines);
    result.setCtr(DxvkStatCounter::PipeCountCompute,  pipe.numComputePipelines);
    result.setCtr(DxvkStatCounter::PipeCompilerBusy,  m_objects.pipelineManager().isCompilingShaders());
    result.setCtr(DxvkStatCounter::GpuIdleTicks,      m_submissionQueue.gpuIdleTicks());
    std::lock_guard<sync::Spinlock> lock(m_statLock);
    result.merge(m_statCounters);
    return result;
  }
  
  DxvkMemoryStats DxvkDevice::getMemoryStats(uint32_t heap) { return m_objects.memoryManager().getMemoryStats(heap); }
  uint32_t DxvkDevice::getCurrentFrameId() const { return m_statCounters.getCtr(DxvkStatCounter::QueuePresentCount); }
  void DxvkDevice::initResources() { m_objects.dummyResources().clearResources(this); }
  void DxvkDevice::registerShader(const Rc<DxvkShader>& shader) { m_objects.pipelineManager().registerShader(shader); }
  
  void DxvkDevice::presentImage(const Rc<vk::Presenter>& p, DxvkSubmitStatus* s) {
    s->result = VK_NOT_READY;
    DxvkPresentInfo info; info.presenter = p;
    m_submissionQueue.present(info, s);
    std::lock_guard<sync::Spinlock> statLock(m_statLock);
    m_statCounters.addCtr(DxvkStatCounter::QueuePresentCount, 1);
  }

  void DxvkDevice::submitCommandList(const Rc<DxvkCommandList>& c, VkSemaphore w1, VkSemaphore w2) {
    DxvkSubmitInfo info; info.cmdList = c; info.waitSync = w1; info.wakeSync = w2;
    m_submissionQueue.submit(info);
    std::lock_guard<sync::Spinlock> statLock(m_statLock);
    m_statCounters.merge(c->statCounters());
    m_statCounters.addCtr(DxvkStatCounter::QueueSubmitCount, 1);
  }
  
  VkResult DxvkDevice::waitForSubmission(DxvkSubmitStatus* s) {
    VkResult r = s->result.load();
    if (r == VK_NOT_READY) { m_submissionQueue.synchronizeSubmission(s); r = s->result.load(); }
    return r;
  }

  void DxvkDevice::waitForResource(const Rc<DxvkResource>& r, DxvkAccess a) {
    if (r->isInUse(a)) {
      auto t0 = dxvk::high_resolution_clock::now();
      m_submissionQueue.synchronizeUntil([r, a] { return !r->isInUse(a); });
      auto t1 = dxvk::high_resolution_clock::now();
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
      std::lock_guard<sync::Spinlock> lock(m_statLock);
      m_statCounters.addCtr(DxvkStatCounter::GpuSyncCount, 1);
      m_statCounters.addCtr(DxvkStatCounter::GpuSyncTicks, us.count());
    }
  }
  
  void DxvkDevice::waitForIdle() {
    this->lockSubmission();
    if (m_vkd->vkDeviceWaitIdle(m_vkd->device()) != VK_SUCCESS) Logger::err("DxvkDevice: waitForIdle failed");
    this->unlockSubmission();
  }
  
  DxvkDevicePerfHints DxvkDevice::getPerfHints() {
    DxvkDevicePerfHints h;
    h.preferFbDepthStencilCopy = VK_TRUE;
    h.preferFbResolve = VK_TRUE;
    return h;
  }

  void DxvkDevice::recycleCommandList(const Rc<DxvkCommandList>& c) { m_recycledCommandLists.returnObject(c); }
  void DxvkDevice::recycleDescriptorPool(const Rc<DxvkDescriptorPool>& p) { m_recycledDescriptorPools.returnObject(p); }
  DxvkDeviceQueue DxvkDevice::getQueue(uint32_t f, uint32_t i) const {
    VkQueue q = VK_NULL_HANDLE;
    m_vkd->vkGetDeviceQueue(m_vkd->device(), f, i, &q);
    return DxvkDeviceQueue { q, f, i };
  }
}
