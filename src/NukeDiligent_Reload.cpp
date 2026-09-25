// Shader hot reload. The engine pushes a changed source (setShaderSource + reloadShader); at the
// next frame boundary the dependents rebuild: the owners recorded under ReloadScope whose sources
// the change reaches through #includes, the world pipes whose text includes it, the RT pipeline
// for rt_* / *.surf, and the modules' reloaders. Draws keep the old pipelines until the new ones
// land (builder thread / warm-up pump), exactly like an MSAA change.
#include "NukeDiligentImpl.h"
#include "../include/NukeDiligentNative.h"
#include <vector>

using namespace Diligent;
using std::cout;
using std::endl;

static thread_local const char* tlsReloadOwner = nullptr;
const char*& NukeDiligent::Impl::ReloadOwner() { return tlsReloadOwner; }

// "rt_rgen.hlsl" -> "rt_rgen" (sources are keyed by stem; includes keep their .hlsli name).
static std::string StemOf(const std::string& file)
{
	if (file.size() > 5 && file.compare(file.size() - 5, 5, ".hlsl") == 0) return file.substr(0, file.size() - 5);
	return file;
}

void NukeDiligent::Impl::NoteShaderUse(const char* file)
{
	if (!file) return;
	boost::mutex::scoped_lock l(shaderLock);
	if (const char* owner = ReloadOwner()) reloadStems[owner].insert(StemOf(file));
}

void NukeDiligent::Impl::ReloadShaderSource(const char* name)
{
	if (!name) return;
	boost::mutex::scoped_lock l(shaderLock);
	if (!reloadChanged.erase(name)) return;   // the text did not change (a re-push): nothing to rebuild
	reloadQueue.insert(name);
}

// Every `#include "x"` of a source, as source keys.
static void IncludesOf(const std::string& src, std::vector<std::string>& out)
{
	out.clear();
	size_t p = 0;
	while ((p = src.find("#include", p)) != std::string::npos)
	{
		p += 8;
		const size_t q0 = src.find('"', p);
		if (q0 == std::string::npos) break;
		const size_t q1 = src.find('"', q0 + 1);
		if (q1 == std::string::npos) break;
		std::string inc = src.substr(q0 + 1, q1 - q0 - 1);
		const size_t slash = inc.find_last_of("/\\");
		if (slash != std::string::npos) inc = inc.substr(slash + 1);
		out.push_back(StemOf(inc));
		p = q1 + 1;
	}
}

void NukeDiligent::Impl::ProcessShaderReloads()
{
	std::set<std::string> affected;
	{
		boost::mutex::scoped_lock l(shaderLock);
		if (reloadQueue.empty()) return;
		affected.swap(reloadQueue);
		// Closure over includes: a source that includes an affected one is affected too.
		std::vector<std::string> incs;
		for (bool grew = true; grew; )
		{
			grew = false;
			for (auto& kv : shaderSrc)
			{
				if (affected.count(kv.first)) continue;
				IncludesOf(kv.second, incs);
				for (const std::string& i : incs)
					if (affected.count(i)) { affected.insert(kv.first); grew = true; break; }
			}
		}
	}
	std::set<std::string> owners;
	{
		boost::mutex::scoped_lock l(shaderLock);
		for (auto& kv : reloadStems)
			for (const std::string& s : kv.second)
				if (affected.count(s)) { owners.insert(kv.first); break; }
	}
	// World pipes: the default one pulls world.vs/ps by name (owner "world"); the material ones
	// carry their text — stale when it includes an affected source.
	bool worldStale = owners.count("world") != 0;
	std::vector<std::string> incs;
	bool rtDirty = false;
	for (const std::string& a : affected)
		if (a.rfind("rt_", 0) == 0 || (a.size() > 5 && a.compare(a.size() - 5, 5, ".surf") == 0)) rtDirty = true;
	{
		std::string names;
		for (const std::string& a : affected) names += (names.empty() ? "" : ", ") + a;
		std::string who;
		for (const std::string& o : owners) who += (who.empty() ? "" : ", ") + o;
		cout << "[NukeDiligent]\tshader reload: " << names << " -> " << (who.empty() ? "no pipeline owner" : who)
		     << (rtDirty ? ", rt" : "") << endl;
	}
	int staleWorld = 0;
	for (auto& kv : worldPipes)
	{
		WorldPipe& wp = kv.second;
		bool stale = worldStale;
		if (!stale)
		{
			for (const std::string* txt : { &wp.vsSrc, &wp.psSrc, &wp.hsSrc, &wp.dsSrc })
			{
				IncludesOf(*txt, incs);
				for (const std::string& i : incs) if (affected.count(i)) { stale = true; break; }
				if (stale) break;
			}
		}
		if (!stale) continue;
		if (kv.first == defaultWorldHandle) { wp.vsSrc = shaderSource("world.vs"); wp.psSrc = shaderSource("world.ps"); }
		if (!wp.tessCustom) { wp.hsSrc.clear(); wp.dsSrc.clear(); }   // the shared pair resolves fresh at build
		wp.builtStages = 0; wp.buildFailed = false;                     // the pump rebuilds base first; the old pipe draws meanwhile
		++staleWorld;
	}
	if (staleWorld) cout << "[NukeDiligent]\tshader reload: " << staleWorld << " world pipeline(s) rebuild" << endl;
	if (rtDirty && rtSupported) rtPipelineDirty = true;

	auto has = [&](const char* o) { return owners.count(o) != 0; };
	auto drop = [&](auto& obj) { Trash(obj); obj.Release(); };
	if (has("world") || has("skin")) CreateSkinCS();
	if (has("world") || has("bend")) CreateBendCS();
	if (has("ui"))
	{
		for (auto& kv : uiSRBCache) Trash(kv.second.srb);
		uiSRBCache.clear();
		CreateUIPipeline(uiBBFmt, uiDSFmt);
	}
	if (has("post")) CreatePostResources();        // also the panini pipes, the G-buffer build and the PostFX built-ins
	else if (has("postfx")) CreatePostFXPipelines();
	if (has("post") || has("postpipe"))
	{
		int n = 0;
		for (auto& kv : postPipes)
		{
			PostPipe& pp = kv.second;
			if (pp.isRTRef || pp.name.empty()) continue;
			const std::string name = pp.name, ps = pp.ps;
			Trash(pp.pso); Trash(pp.srb);
			pp = PostPipe{};
			if (BuildPostPipe(name, ps, pp)) ++n;
			pp.name = name; pp.ps = ps;
		}
		if (n) cout << "[NukeDiligent]\tshader reload: " << n << " post effect pipeline(s) rebuilt" << endl;
	}
	if (has("post") || has("blit"))
	{
		for (auto& kv : blitPipes) { Trash(kv.second.first); Trash(kv.second.second); }
		blitPipes.clear();
	}
	if (has("ao"))        { DropPostPipe(aoPipe); DropPostPipe(aoResolvePipe); aoFailed = false; }
	if (has("atmo"))      { drop(atmoLutPSO); drop(atmoLutSRB); DropPostPipe(atmoApplyPipe); atmoFailed = false; }
	if (has("clouds"))
	{
		drop(cloudGenPSO); drop(cloudMarchPSO); drop(cloudTemporalPSO); drop(cloudShadowPSO);
		drop(cloudGenSRB); drop(cloudMarchSRB); drop(cloudTemporalSRB); drop(cloudShadowSRB);
		DropPostPipe(cloudApplyPipe); cloudsFailed = false; cloudNoiseReady = false;
	}
	if (has("cloudprobe")) { drop(cloudProbePSO); drop(cloudProbeSRB); cloudProbeVar = nullptr; }
	if (has("decal"))   decalStamp = PipeStamp{};
	if (has("sky"))     skyStamp = PipeStamp{};
	if (has("sprite"))  spriteStamp = PipeStamp{};
	if (has("outline")) outlineStamp = PipeStamp{};
	if (has("debug"))   debugStamp = PipeStamp{};
	if (has("giprobe")) giProbeSamples = -1;
	if (has("gi"))      { drop(giTracePSO); drop(giUpdatePSO); drop(giCubePSO); drop(giTraceSRB); drop(giUpdateSRB); drop(giCubeSRB); giFailed = false; }
	if (has("lens"))    { drop(lensFilmPSO); drop(lensDropsPSO); drop(lensFilmSRB); drop(lensDropsSRB); lensFailed = false; }
	if (has("panini") && !has("post")) { drop(paniniPSO); drop(paniniPSOBB); drop(paniniSRB); drop(paniniSRBBB); paniniSrcVar = paniniSrcVarBB = nullptr; paniniFailed = false; }
	if (has("skymap"))  { drop(skyMapPSO); drop(skyMapSRB); skyMapFailed = false; }
	if (has("trails"))  { drop(trailShiftPSO); drop(trailStampPSO); drop(trailShiftSRB); drop(trailStampSRB); trailFailed = false; }
	if (has("occl"))    CreateOcclResources();
	if (has("boot"))    BuildBootPipe();
	if (has("shadow"))  CreateShadowResources();
	if (has("ssgi"))    { DropPostPipe(ssgiPipe); DropPostPipe(ssgiResolvePipe); ssgiFailed = false; }
	if (has("gbuffer") && !has("post"))
		if (!gbufBuilding.exchange(true))
			EnqueueBuild([this] { BuildGBufferPipe(); }, [this] { gbufBuilding = false; }, kPrioGBuffer, "G-buffer pipelines");
	if (has("grid"))       gridSamples = -1;
	if (has("cost"))       costStamp = PipeStamp{};
	if (has("cursor"))     cursorFmt = TEX_FORMAT_UNKNOWN;
	if (has("depthdebug")) debugDepthSamples = -1;
	if (has("vol"))       { drop(volInjectPSO); drop(volTemporalPSO); drop(volIntegratePSO); drop(volInjectSRB); drop(volTemporalSRB); drop(volIntegrateSRB); DropPostPipe(volApplyPipe); volFailed = false; }
	if (has("fluid"))     { drop(fluidPSO); drop(fluidSRB); fluidFailed = false; }
	if (has("sunshafts")) { drop(sunShaftPSO); drop(sunShaftSRB); ssFailed = false; }
	if (has("rt") && rtSupported) rtPipelineDirty = true;
	for (const ModuleReloader& m : moduleReloaders)
		if (has(m.owner.c_str()) && m.fn) m.fn(m.user);
	for (WarmEntry& e : warmups) e.done = false;   // every builder gets another look
}

// ---- iRender ----
void NukeDiligent::reloadShader(const char* name) { m_impl->ReloadShaderSource(name); }

// ---- native seam (modules) ----
namespace nukediligent {

void AddShaderReloader(const char* owner, ShaderReloadFn fn, void* user)
{
	NukeDiligent::Impl* d = NukeDiligent::nativeImpl;
	if (!d || !owner || !fn) return;
	for (auto& m : d->moduleReloaders) if (m.user == user && m.fn == fn) return;   // idempotent
	NukeDiligent::Impl::ModuleReloader m; m.owner = owner; m.fn = fn; m.user = user;
	d->moduleReloaders.push_back(m);
}

void RemoveShaderReloader(void* user)
{
	NukeDiligent::Impl* d = NukeDiligent::nativeImpl;
	if (!d) return;
	for (size_t i = 0; i < d->moduleReloaders.size(); )
	{
		if (d->moduleReloaders[i].user == user) d->moduleReloaders.erase(d->moduleReloaders.begin() + i);
		else ++i;
	}
}

static thread_local const char* gPrevOwner = nullptr;
void BeginShaderOwner(const char* owner) { gPrevOwner = NukeDiligent::Impl::ReloadOwner(); NukeDiligent::Impl::ReloadOwner() = owner; }
void EndShaderOwner()                    { NukeDiligent::Impl::ReloadOwner() = gPrevOwner; gPrevOwner = nullptr; }

}  // namespace nukediligent
