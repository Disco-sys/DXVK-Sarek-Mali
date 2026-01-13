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
    
    // --- DISCO-SYS MALI PERFORMANCE OVERRIDE ---
    // 1. Fix Black Screen: Force nullDescriptor support even if driver is shy
    m_features.extRobustness2.nullDescriptor = VK_TRUE;

    // 2. Optimization: Disable CPU-heavy bounds checking on Mali
    m_features.core.features.robustBufferAccess = VK_FALSE;

    // 3. Compatibility: Fake Dual Source Blending for Feature Level 11_1
    m_features.core.features.dualSrcBlend = VK_TRUE;
    m_features.core.features.logicOp = VK_TRUE;
    
    // 4. Geometry Hack: Prevent Mali crashes by limiting massive tessellation
    if (m_features.core.features.tessellationShader) {
       // We keep the feature enabled but will limit it in the pipeline
    }
    // -------------------------------------------

    auto queueFamilies = m_adapter->findQueueFamilies();
    m_queues.graphics = getQueue(queueFamilies.graphics, 0);
    m_queues.transfer = getQueue(queueFamilies.transfer, 0);
  }
  
  DxvkDevice::~DxvkDevice() {
    if (this_thread::isInModuleDetachment())
      return;

    this->waitForIdle();
    m_objects.pipelineManager().stopWorkerThreads();
  }

  bool DxvkDevice::isUnifiedMemoryArchitecture() const {
    // Mali is UMA (Shared System/Video RAM), this returns true for Helio G85
    return m_adapter->isUnifiedMemoryArchitecture();
  }

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
    
    if (m_features.core.features.tessellationShader) {
      result |= VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT
             |  VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
    }

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
    if (cmdList == nullptr)
      cmdList = new DxvkCommandList(this);
    return cmdList;
  }

  Rc<DxvkDescriptorPool> DxvkDevice::createDescriptorPool() {
    Rc<DxvkDescriptorPool> pool = m_recycledDescriptorPools.retrieveObject();
    if (pool == nullptr)
      pool = new DxvkDescriptorPool(m_vkd);
    return pool;
  }
  
  Rc<DxvkContext> DxvkDevice::createContext() {
    return new DxvkContext(this);
  }

  Rc<DxvkGpuEvent> DxvkDevice::createGpuEvent() {
    return new DxvkGpuEvent(m_vkd);
  }

  Rc<DxvkGpuQuery> DxvkDevice::createGpuQuery(
          VkQueryType           type,
          VkQueryControlFlags   flags,
          uint32_t              index) {
    return new DxvkGpuQuery(m_vkd, type, flags, index);
  }
  
  Rc<DxvkFence> DxvkDevice::createFence(const DxvkFenceCreateInfo& fenceInfo) {
    return new DxvkFence(this, fenceInfo);
  }
  
  Rc<DxvkFramebuffer> DxvkDevice::createFramebuffer(const DxvkFramebufferInfo& info) {
    return new DxvkFramebuffer(m_vkd, info);
  }
  
  Rc<DxvkBuffer> DxvkDevice::createBuffer(const DxvkBufferCreateInfo& createInfo, VkMemoryPropertyFlags memoryType) {
    return new DxvkBuffer(this, createInfo, m_objects.memoryManager(), memoryType);
  }
  
  Rc<DxvkBufferView> DxvkDevice::createBufferView(const Rc<DxvkBuffer>& buffer, const DxvkBufferViewCreateInfo& createInfo) {
    return new DxvkBufferView(m_vkd, buffer, createInfo);
  }
  
  Rc<DxvkImage> DxvkDevice::createImage(const DxvkImageCreateInfo& createInfo, VkMemoryPropertyFlags memoryType) {
    return new DxvkImage(this, createInfo, m_objects.memoryManager(), memoryType);
  }
  
  Rc<DxvkImage> DxvkDevice::createImageFromVkImage(const DxvkImageCreateInfo& createInfo, VkImage image) {
    return new DxvkImage(this, createInfo, image);
  }
  
  Rc<DxvkImageView> DxvkDevice::createImageView(const Rc<DxvkImage>& image, const DxvkImageViewCreateInfo& createInfo) {
    return new DxvkImageView(m_vkd, image, createInfo);
  }
  
  Rc<DxvkSampler> DxvkDevice::createSampler(const DxvkSamplerCreateInfo& createInfo) {
    return new DxvkSampler(this, createInfo);
  }
  
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
  
  DxvkMemoryStats DxvkDevice::getMemoryStats(uint32_t heap) {
    return m_objects.memoryManager().getMemoryStats(heap);
  }

  uint32_t DxvkDevice::getCurrentFrameId() const {
    return m_statCounters.getCtr(DxvkStatCounter::QueuePresentCount);
  }
  
  void DxvkDevice::initResources() {
    m_objects.dummyResources().clearResources(this);
  }

  void DxvkDevice::registerShader(const Rc<DxvkShader>& shader) {
    m_objects.pipelineManager().registerShader(shader);
  }
  
  void DxvkDevice::presentImage(const Rc<vk::Presenter>& presenter, DxvkSubmitStatus* status) {
    status->result = VK_NOT_READY;
    DxvkPresentInfo presentInfo;
    presentInfo.presenter = presenter;
    m_submissionQueue.present(presentInfo, status);
    
    std::lock_guard<sync::Spinlock> statLock(m_statLock);
    m_statCounters.addCtr(DxvkStatCounter::QueuePresentCount, 1);
  }

  void DxvkDevice::submitCommandList(const Rc<DxvkCommandList>& commandList, VkSemaphore waitSync, VkSemaphore wakeSync) {
    DxvkSubmitInfo submitInfo;
    submitInfo.cmdList  = commandList;
    submitInfo.waitSync = waitSync;
    submitInfo.wakeSync = wakeSync;
    m_submissionQueue.submit(submitInfo);

    std::lock_guard<sync::Spinlock> statLock(m_statLock);
    m_statCounters.merge(commandList->statCounters());
    m_statCounters.addCtr(DxvkStatCounter::QueueSubmitCount, 1);
  }
  
  VkResult DxvkDevice::waitForSubmission(DxvkSubmitStatus* status) {
    VkResult result = status->result.load();
    if (result == VK_NOT_READY) {
      m_submissionQueue.synchronizeSubmission(status);
      result = status->result.load();
    }
    return result;
  }

  void DxvkDevice::waitForResource(const Rc<DxvkResource>& resource, DxvkAccess access) {
    if (resource->isInUse(access)) {
      auto t0 = dxvk::high_resolution_clock::now();
      m_submissionQueue.synchronizeUntil([resource, access] {
        return !resource->isInUse(access);
      });
      auto t1 = dxvk::high_resolution_clock::now();
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

      std::lock_guard<sync::Spinlock> lock(m_statLock);
      m_statCounters.addCtr(DxvkStatCounter::GpuSyncCount, 1);
      m_statCounters.addCtr(DxvkStatCounter::GpuSyncTicks, us.count());
    }
  }
  
  void DxvkDevice::waitForIdle() {
    this->lockSubmission();
    if (m_vkd->vkDeviceWaitIdle(m_vkd->device()) != VK_SUCCESS)
      Logger::err("DxvkDevice: waitForIdle: Operation failed");
    this->unlockSubmission();
  }
  
  DxvkDevicePerfHints DxvkDevice::getPerfHints() {
    DxvkDevicePerfHints hints;
    // Mali/MediaTek optimization: favor resolve and depth stencil copy
    hints.preferFbDepthStencilCopy = VK_TRUE;
    hints.preferFbResolve = VK_TRUE;
    return hints;
  }

  void DxvkDevice::recycleCommandList(const Rc<DxvkCommandList>& cmdList) {
    m_recycledCommandLists.returnObject(cmdList);
  }
  
  void DxvkDevice::recycleDescriptorPool(const Rc<DxvkDescriptorPool>& pool) {
    m_recycledDescriptorPools.returnObject(pool);
  }

  DxvkDeviceQueue DxvkDevice::getQueue(uint32_t family, uint32_t index) const {
    VkQueue queue = VK_NULL_HANDLE;
    m_vkd->vkGetDeviceQueue(m_vkd->device(), family, index, &queue);
    return DxvkDeviceQueue { queue, family, index };
  }
  
}
