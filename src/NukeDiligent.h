#pragma once
// Diligent-based renderer implementing the engine's abstract iRender interface.
// Lives in its own DLL (NukeRenderDiligent) and is loaded as a render module.
// Depends ONLY on the iRender interface (no engine singletons), so the engine
// core can stay a static library.

#include <render/irender.h>
#include <boost/function.hpp>

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
	void getCursorPos(double& x, double& y) override;
	bool isMouseButtonDown(int button) override;
	void bindRenderTarget(uint64_t rtId) override;
	void invalidateTexture(Texture* t) override;
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
	void renderGBufferObject(Mesh* mesh, Material* mat, const float pos[3], const float quat[4], const float scale[3]) override;
	void endGBufferPass() override;
	bool rtAvailable() override;
	void beginRTScene() override;
	void addRTInstance(Mesh* mesh, Material* mat, const float pos[3], const float quat[4], const float scale[3]) override;
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

private:
	struct Impl;          // PImpl: keeps Diligent types out of this header
	Impl*       m_impl   = nullptr;
	GLFWwindow* m_window = nullptr;
};
