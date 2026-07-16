#pragma once
// Diligent-based renderer implementing the engine's abstract iRender interface.
// Lives in its own DLL (NukeRenderDiligent) and is loaded as a render module.
// Depends ONLY on the iRender interface (no engine singletons), so the engine
// core can stay a static library.

#include <render/irender.h>
#include <boost/function.hpp>
#include <deque>
#include <mutex>

using namespace nuke;   // iRender / Mesh / Material / NukeCameraDesc live in namespace nuke

struct GLFWwindow;

class NukeDiligent : public iRender
{
public:
	NukeDiligent();
	~NukeDiligent();

	void setShaderSource(const char* name, const char* source) override;
	uint64_t createShaderPipeline(const char* name, const char* vs, const char* ps) override;
	int  init(const WindowDesc& desc) override;
	int  render() override;
	void renderObject(Mesh* mesh, Material* mat,
	                  const float pos[3], const float quat[4], const float scale[3]) override;
	void renderSelectionOutline(Mesh* mesh, const float pos[3], const float quat[4], const float scale[3]) override;
	void setWindowTitle(const char* title) override;
	bool isWindowFocused() override;
	bool isWindowMaximized() override;
	void setWindowMaximized(bool maximized) override;
	void applyWindow(const WindowDesc& desc) override;   // runtime size/mode/decoration/opacity
	void setVSync(bool on) override;                     // main-present vsync (live)
	bool getVSync() override;
	void drawSprite(Texture* tex, const float center[3], const float right[3], const float up[3],
	                const float uv[4], const float tint[4]) override;
	void drawSpriteScreen(Texture* tex, const float rect[4], const float refSize[2],
	                      const float uv[4], const float tint[4], int afterPost) override;
	void drawSpriteScreenEx(Texture* tex, const float rect[4], const float refSize[2],
	                        const float uv[4], const float tint[4], int afterPost, int scaleMode) override;
	void drawDecal(Texture* tex, const float pos[3], const float quat[4], const float scale[3],
	               const float tint[4], float intensity, float angleFade, int mode) override;
	void setWireframe(bool on) override;                 // scene fill mode (world meshes only)
	bool getWireframe() override;
	void drawDebugLineDepth(const float a[3], const float b[3], const float color[4]) override;
	void drawSpriteRun(Texture* tex, const float* verts, int vertCount) override;
	void getCursorPos(double& x, double& y) override;
	bool isMouseButtonDown(int button) override;
	void bindRenderTarget(uint64_t rtId) override;
	void invalidateTexture(Texture* t) override;
	void invalidateMesh(Mesh* m) override;
	void setLights(const NukeLight* lights, int count) override;
	void setSky(const NukeSky& sky) override;
	void setMSAA(int samples) override;
	int  getMSAA() override;
	void setHDR(bool on) override;
	bool getHDR() override;
	void setHDROutput(bool on) override;
	bool getHDROutput() override;
	void setHDRNits(float paperWhite, float peak) override;
	void setShadowSettings(int resolution, float distance, float depthBias, float normalBias, float softness) override;
	void setRTReflection(float intensity, float maxDist, int bounces, float roughCutoff) override;
	uint64_t createReflectionCube(int resolution) override;
	void beginCubeFace(uint64_t cube, int face, const float pos[3], float nearZ, float farZ) override;
	void endCubeFace(uint64_t cube, int face) override;
	void setReflectionProbe(uint64_t cube, const float pos[3], float intensity, float farZ, const float boxHalf[3]) override;
	void beginGBufferPass(const NukeCameraDesc& cam) override;
	void renderGBufferObject(Mesh* mesh, Material* mat, const float pos[3], const float quat[4], const float scale[3],
	                         const float prevPos[3] = nullptr, const float prevQuat[4] = nullptr, const float prevScale[3] = nullptr) override;
	void endGBufferPass() override;
	bool rtAvailable() override;
	void beginRTScene() override;
	void addRTInstance(Mesh* mesh, Material* mat, const float pos[3], const float quat[4], const float scale[3], bool inReflections = true) override;
	void setCameraTAA(bool enabled) override;
	void requestClose() override;
	void drawDebugLine(const float a[3], const float b[3], const float color[4]) override;
	void buildRTScene() override;
	uint64_t createPostPipeline(const char* name, const char* ps) override;
	void     setPostChain(const NukePostStage* stages, int count) override;
	void setOnFileDrop(bst::function<void(const char*)> cb) override;
	int  shadowPassCount() override;
	void beginShadowPass(int pass) override;
	void renderShadowObject(Mesh* mesh, const float pos[3], const float quat[4], const float scale[3], Material* mat) override;
	void endShadowPass() override;
	void loop() override;
	void deinit() override;
	void update() override;
	char* getEngine() override;
	char* getVersion() override;
	void setOnGUI(bst::function<void(void)> cb) override;
	void setOnRender(bst::function<void(void)> cb) override;
	void setOnClose(bst::function<void()> cb) override;

	// Neutral UI seam (generic 2D draw; no ImGui types here).
	uint64_t createTexture2D(const void* rgbaPixels, int width, int height) override;
	void     destroyTexture2D(uint64_t handle) override;
	void     renderDrawLists(const NukeUIDrawData& data) override;
	void*    nativeWindow() override;   // GLFWwindow* (UI platform backend mounts on it)
	void     uiViewportRender(void* nativeHandle, int w, int h, const NukeUIDrawData& data) override;
	void     uiViewportDestroy(void* nativeHandle) override;
	void     getFrameStats(int& drawCalls, int& triangles) override;
	uint64_t createRenderTarget(int w, int h) override;
	void     resizeRenderTarget(uint64_t id, int w, int h) override;
	uint64_t getRenderTargetTexture(uint64_t id) override;
	void     beginCamera(const NukeCameraDesc& cam) override;
	void     endCamera() override;
	void     getViewProj(float* view16, float* proj16) override;

	void keyboard(int key, int scancode, int action, int mods) override;
	void mouseMove(double xpos, double ypos) override;
	void mouseClick(int button, int action, int mods) override;
	void setCursorMode(int mode) override;
	void rawMouse(double xpos, double ypos) override;
	void mouseEnterLeave(int entered) override;

	// Runtime-GUI input seam (2.5): drained queues filled by the GLFW callbacks.
	int  fetchUIChars(unsigned int* out, int max) override;
	int  fetchUIKeys(int* keys, int* actions, int* mods, int max) override;
	void getScrollDelta(double& x, double& y) override;
	const char* getClipboardText() override;
	void setClipboardText(const char* text) override;

	// Filled from the GLFW callbacks; size-capped so an idle consumer can't leak.
	struct UIInput
	{
		std::mutex m;
		std::deque<unsigned int> chars;
		struct Key { int key, action, mods; };
		std::deque<Key> keys;
		double scrollX = 0.0, scrollY = 0.0;
	} m_uiInput;

private:
	struct Impl;          // PImpl: keeps Diligent types out of this header
	Impl*       m_impl   = nullptr;
	GLFWwindow* m_window = nullptr;
	// Windowed placement remembered when going fullscreen, so returning to windowed restores
	// a sane position/size (glfwSetWindowMonitor to NULL needs an explicit rect). -1 = unset.
	int m_winX = -1, m_winY = -1, m_winW = 0, m_winH = 0;
	int m_windowMode = 0;   // current applied WindowMode (0/1/2)
};
