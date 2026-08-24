// DirectStorage provider (Fast loading 4, D3D12 only): pak entries are read through the NVMe
// request queue, GDeflate blocks inflate on the GPU, cooked textures land straight in VRAM and
// everything else (meshes, clips, documents) lands in system memory for the engine's decoders.
// A waiter thread turns fence completions into finished jobs; the render thread adopts landed
// textures once per frame. Every failure path falls back to a CPU read of the same blocks.
#include "NukeDiligentImpl.h"
#include "API/Model/Storage.h"
#include "API/Model/Texture.h"
#include "API/Model/Jobs.h"
#ifdef _WIN32
#include <config.h>
#include <dstorage.h>
#include "RenderDeviceD3D12.h"
#include "DeviceContextD3D12.h"
#include "TextureD3D12.h"
#include "BufferD3D12.h"
#include <dxgi1_4.h>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <thread>

using nuke::Texture;
using nuke::Package;

namespace {

inline uint64_t BlockFileOffset(const Package::Entry& e, size_t block)
{
	uint64_t off = e.offset;
	for (size_t b = 0; b < block; ++b) off += e.blocks[b].packSize;
	return off;
}
inline bool DirectDecodable(uint8_t method) { return method == Package::M_GDeflate || method == Package::M_Store; }
// Every block of the entry must be something DirectStorage inflates itself (GDeflate or stored).
inline bool DirectDecodable(const Package::Entry& e)
{
	if (e.blocks.empty()) return false;   // a v1 pak: the CPU path knows it
	for (const Package::Block& b : e.blocks) if (!DirectDecodable(b.method)) return false;
	return true;
}
inline DSTORAGE_COMPRESSION_FORMAT Format(const Package::Block& b)
{ return b.method == Package::M_GDeflate ? DSTORAGE_COMPRESSION_FORMAT_GDEFLATE : DSTORAGE_COMPRESSION_FORMAT_NONE; }
inline uint64_t Align(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

}  // namespace

// Texture job: one GPU texture being filled from its pak blocks (DirectStorage requests and/or
// CPU-read blocks the adopter uploads). `t` is only a key — it may be invalidated before the
// job lands. Heap-allocated: request destinations point into cpuBlocks.
struct NukeDiligent::Impl::StorTexJob
{
	Texture* t = nullptr;
	int      base = 0;
	RefCntAutoPtr<ITexture> tex;
	RefCntAutoPtr<IBuffer>  staging;                       // VRAM landing buffer (DirectStorage path)
	std::vector<std::pair<int, uint64_t>> placed;          // (block index, offset in staging)
	std::vector<std::pair<int, std::string>> cpuBlocks;   // (block index, inflated pitched bytes) — CPU path
	uint64_t fence = 0;
	uint64_t bytes = 0;
	uint32_t status = 0;         // slot in the status array (per-job HRESULT)
	bool     failed = false;
};

struct NukeDiligent::Impl::StorMemJob
{
	Package::Location loc;
	std::vector<std::string> blocks;
	uint64_t fence = 0;
	uint64_t bytes = 0;
	uint32_t status = 0;
	nuke::Storage::ReadDone done;
};

// One DirectStorage queue + its own fence (fence values are only monotonic per queue).
struct NukeDiligent::Impl::StorQueue
{
	IDStorageQueue* q = nullptr;
	ID3D12Fence*    fence = nullptr;
	HANDLE          event = nullptr;
	uint64_t        next = 1;
	int             inflight = 0;      // jobs signalled, not completed (guarded by DStor::lock)
	bool            dirty = false;     // enqueued since the last Submit
	std::deque<std::unique_ptr<StorTexJob>> tex;   // pending texture jobs (fence order)
	std::deque<std::unique_ptr<StorMemJob>> mem;   // pending memory jobs (fence order)
};

struct NukeDiligent::Impl::DStor : public nuke::Storage::Provider
{
	Impl* impl = nullptr;
	HMODULE dll = nullptr;
	IDStorageFactory* factory = nullptr;
	ID3D12Device* d3d = nullptr;
	StorQueue qTex, qLow, qMem;        // textures (draw demand), textures (prefetch), memory reads
	// One status slot per job (round-robin; far more slots than jobs can be in flight): the
	// per-job HRESULT — the queue's error record only says "something failed".
	static const uint32_t kStatusSlots = 4096;
	IDStorageStatusArray* status = nullptr;
	uint32_t nextStatus = 0;           // guarded by lock
	uint32_t TakeStatus() { const uint32_t s = nextStatus; nextStatus = (nextStatus + 1) % kStatusSlots; return s; }
	bool gpuDecomp = false;
	bool verify = false;               // NUKE_DSTORAGE_VERIFY=1: read every landed texture back and compare

	std::mutex lock;
	std::map<std::string, IDStorageFile*> files;
	std::vector<std::unique_ptr<StorTexJob>> texDone;   // landed: the render thread adopts
	std::vector<Texture*> prefetch;                      // scan-registered, awaiting a LOW request

	std::thread waiter;
	HANDLE kick = nullptr, quit = nullptr;
	std::atomic<uint64_t> statReq{ 0 }, statBytes{ 0 };

	// ---- Provider ----
	const char* Name() const override { return "DirectStorage"; }
	bool GpuTextures() const override { return true; }
	bool ReadAsync(const Package::Location& loc, int priority, const nuke::Storage::ReadDone& done) override;
	void Flush() override { SubmitAll(); }
	void Stats(uint64_t& requests, uint64_t& bytes, bool& gpu) const override
	{ requests = statReq; bytes = statBytes; gpu = gpuDecomp; }
	void PrefetchTexture(Texture* t) override { std::lock_guard<std::mutex> l(lock); prefetch.push_back(t); }

	IDStorageFile* File(const std::string& path)   // lock held
	{
		auto it = files.find(path);
		if (it != files.end()) return it->second;
		std::wstring w;
		const int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
		if (n > 0) { w.resize(n - 1); MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &w[0], n); }
		IDStorageFile* f = nullptr;
		if (FAILED(factory->OpenFile(w.c_str(), IID_PPV_ARGS(&f)))) f = nullptr;
		files[path] = f;
		return f;
	}
	void SubmitAll()
	{
		std::lock_guard<std::mutex> l(lock);
		for (StorQueue* q : { &qTex, &qLow, &qMem })
			if (q->dirty) { q->q->Submit(); q->dirty = false; }
	}
	bool MakeQueue(StorQueue& q, DSTORAGE_PRIORITY prio, const char* name)
	{
		DSTORAGE_QUEUE_DESC d{};
		d.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
		d.Capacity   = DSTORAGE_MAX_QUEUE_CAPACITY;
		d.Priority   = prio;
		d.Name       = name;
		d.Device     = d3d;
		if (FAILED(factory->CreateQueue(&d, IID_PPV_ARGS(&q.q)))) return false;
		if (FAILED(d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&q.fence)))) return false;
		q.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
		return q.event != nullptr;
	}
	void DestroyQueue(StorQueue& q)
	{
		if (q.q) { q.q->Close(); q.q->Release(); q.q = nullptr; }
		if (q.fence) { q.fence->Release(); q.fence = nullptr; }
		if (q.event) { CloseHandle(q.event); q.event = nullptr; }
	}

	// IO mode ladder, walked down by the init self-test: GPU decompression with the normal
	// unbuffered file path -> GPU decompression with buffered file IO -> CPU inflate.
	enum { kModeGpu = 0, kModeGpuBuffered = 1, kModeCpu = 2 };
	int ioMode = kModeGpu;
	bool CreateRuntime(int mode, int stagingMB);
	void DestroyRuntime();
	bool SelfTestGpuDecompression();
	// The self-test verdict is remembered per adapter + driver + runtime build in
	// config/dstorage.json, so an affected machine skips the doomed GPU init on later boots.
	std::string MachineKey() const;
	static boost::filesystem::path VerdictPath();
	void WaiterLoop();
	void Complete(StorQueue& q);       // a queue's fence moved: finish the jobs it covers
	void CpuFallbackTex(StorTexJob& j);
};

// ---- init / shutdown ---------------------------------------------------------------------------

// Factory + queues + status array under the current process configuration. False = nothing
// created (everything already released).
bool NukeDiligent::Impl::DStor::CreateRuntime(int mode, int stagingMB)
{
	typedef HRESULT(WINAPI* GetFactoryFn)(REFIID, void**);
	typedef HRESULT(WINAPI* SetConfigFn)(DSTORAGE_CONFIGURATION1 const*);
	auto getFactory = (GetFactoryFn)GetProcAddress(dll, "DStorageGetFactory");
	auto setConfig  = (SetConfigFn)GetProcAddress(dll, "DStorageSetConfiguration1");
	if (!getFactory) { std::cout << "[DStorage]\tDStorageGetFactory missing — CPU pak reads" << std::endl; return false; }
	ioMode = mode;
	if (setConfig)
	{
		// Process-wide, read when the factory is created (so a re-create after a failed
		// self-test picks the changed mode up).
		DSTORAGE_CONFIGURATION1 c{};
		c.DisableTelemetry = TRUE;   // an engine does not phone home
		c.DisableGpuDecompression = mode >= kModeCpu ? TRUE : FALSE;
		if (mode == kModeGpuBuffered)
		{
			// The self-test found the UNBUFFERED file path feeding the GPU decompressor garbage
			// (seen on Win10 + DirectStorage 1.3): file reads go through the OS cache instead.
			c.ForceFileBuffering = TRUE;
			c.DisableBypassIO    = TRUE;
		}
		if (std::getenv("NUKE_DSTORAGE_NOBYPASS")) c.DisableBypassIO = TRUE;
		if (std::getenv("NUKE_DSTORAGE_NOMETA")) c.DisableGpuDecompressionMetacommand = TRUE;   // built-in shader instead of the driver's
		setConfig(&c);
	}
	if (FAILED(getFactory(IID_PPV_ARGS(&factory)))) { std::cout << "[DStorage]\tfactory creation failed — CPU pak reads" << std::endl; factory = nullptr; return false; }
	factory->SetStagingBufferSize((UINT32)stagingMB << 20);
#ifdef _DEBUG
	factory->SetDebugFlags(DSTORAGE_DEBUG_SHOW_ERRORS);
#endif
	if (FAILED(factory->CreateStatusArray(kStatusSlots, "nuke jobs", IID_PPV_ARGS(&status)))
	    || !MakeQueue(qTex, DSTORAGE_PRIORITY_NORMAL, "nuke textures")
	    || !MakeQueue(qLow, DSTORAGE_PRIORITY_LOW, "nuke prefetch")
	    || !MakeQueue(qMem, DSTORAGE_PRIORITY_NORMAL, "nuke memory"))
	{
		std::cout << "[DStorage]\tqueue creation failed — CPU pak reads" << std::endl;
		DestroyRuntime();
		return false;
	}
	IDStorageQueue2* q2 = nullptr;
	gpuDecomp = false;
	if (SUCCEEDED(qTex.q->QueryInterface(IID_PPV_ARGS(&q2))) && q2)
	{
		const DSTORAGE_COMPRESSION_SUPPORT sup = q2->GetCompressionSupport(DSTORAGE_COMPRESSION_FORMAT_GDEFLATE);
		gpuDecomp = (sup & (DSTORAGE_COMPRESSION_SUPPORT_GPU_OPTIMIZED | DSTORAGE_COMPRESSION_SUPPORT_GPU_FALLBACK)) != 0;
		std::cout << "[DStorage]\tGDeflate: "
		          << ((sup & DSTORAGE_COMPRESSION_SUPPORT_GPU_OPTIMIZED) ? "GPU (driver metacommand)" : (sup & DSTORAGE_COMPRESSION_SUPPORT_GPU_FALLBACK) ? "GPU (built-in shader)" : "CPU threads")
		          << ((sup & DSTORAGE_COMPRESSION_SUPPORT_USES_COMPUTE_QUEUE) ? ", compute queue" : "")
		          << ((sup & DSTORAGE_COMPRESSION_SUPPORT_USES_COPY_QUEUE) ? ", copy queue" : "") << std::endl;
		q2->Release();
	}
	return true;
}

void NukeDiligent::Impl::DStor::DestroyRuntime()
{
	{
		std::lock_guard<std::mutex> l(lock);
		for (auto& f : files) if (f.second) f.second->Release();
		files.clear();
	}
	DestroyQueue(qTex); DestroyQueue(qLow); DestroyQueue(qMem);
	if (status) { status->Release(); status = nullptr; }
	if (factory) { factory->Release(); factory = nullptr; }
}

// GPU decompression self-test: a GDeflate stream of known content (compressed right here by the
// engine's encoder) from MEMORY into a GPU buffer, read back and compared. Some runtime/driver
// pairs inflate to ZEROS with S_OK (seen: DirectStorage 1.3 + GeForce RTX 5090 on Windows 10) —
// the only defence is to look. False = the GPU path is broken on this machine.
bool NukeDiligent::Impl::DStor::SelfTestGpuDecompression()
{
	// The REAL path: a GDeflate stream in a FILE (the engine's own encoder), read through a
	// DirectStorage file queue into a GPU buffer. A memory-source test is NOT enough — the
	// observed defect lives in the unbuffered file staging feeding the GPU decompressor
	// (memory-source requests decode fine while file-source ones deliver zeros).
	const size_t kBytes = 256 * 1024;
	std::string raw(kBytes, '\0');
	for (size_t i = 0; i < kBytes; ++i) raw[i] = (char)(((i * 7) ^ (i >> 9)) & 0xFF);
	std::string packed;
	if (!Package::Compress(Package::M_GDeflate, 6, raw.data(), raw.size(), packed)) { std::cout << "[DStorage]\tself-test: encoder refused the pattern" << std::endl; return true; }
	const boost::filesystem::path testFile = VerdictPath().parent_path() / "dstorage-selftest.bin";
	{
		boost::system::error_code ec;
		boost::filesystem::create_directories(testFile.parent_path(), ec);
		boost::filesystem::ofstream o(testFile, std::ios::binary | std::ios::trunc);
		if (!o) { std::cout << "[DStorage]\tself-test: cannot write " << testFile.string() << std::endl; return true; }
		o.write(packed.data(), (std::streamsize)packed.size());
	}

	bool ok = false;
	ID3D12Resource* buf = nullptr; ID3D12Resource* rb = nullptr;
	ID3D12CommandQueue* cq = nullptr; ID3D12CommandAllocator* ca = nullptr; ID3D12GraphicsCommandList* cl = nullptr;
	ID3D12Fence* fence = nullptr; IDStorageQueue* q = nullptr; IDStorageFile* file = nullptr;
	HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
	do
	{
		D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
		D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = kBytes; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		HRESULT hr;
		if (FAILED(hr = d3d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buf)))) { std::cout << "[DStorage]	self-test: buffer hr 0x" << std::hex << (unsigned)hr << std::dec << std::endl; break; }
		D3D12_HEAP_PROPERTIES rp{}; rp.Type = D3D12_HEAP_TYPE_READBACK;
		if (FAILED(hr = d3d->CreateCommittedResource(&rp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb)))) { std::cout << "[DStorage]	self-test: readback buffer hr 0x" << std::hex << (unsigned)hr << std::dec << std::endl; break; }
		DSTORAGE_QUEUE_DESC qd{}; qd.SourceType = DSTORAGE_REQUEST_SOURCE_FILE; qd.Capacity = DSTORAGE_MIN_QUEUE_CAPACITY; qd.Priority = DSTORAGE_PRIORITY_REALTIME; qd.Name = "nuke self-test"; qd.Device = d3d;
		if (FAILED(hr = factory->CreateQueue(&qd, IID_PPV_ARGS(&q)))) { std::cout << "[DStorage]	self-test: queue hr 0x" << std::hex << (unsigned)hr << std::dec << std::endl; break; }
		if (FAILED(hr = d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) { std::cout << "[DStorage]	self-test: fence hr 0x" << std::hex << (unsigned)hr << std::dec << std::endl; break; }
		if (FAILED(hr = factory->OpenFile(testFile.wstring().c_str(), IID_PPV_ARGS(&file)))) { std::cout << "[DStorage]	self-test: OpenFile hr 0x" << std::hex << (unsigned)hr << std::dec << " " << testFile.string() << std::endl; break; }
		DSTORAGE_REQUEST r{};
		r.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE; r.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_GDEFLATE;
		r.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
		r.Source.File.Source = file; r.Source.File.Offset = 0; r.Source.File.Size = (UINT32)packed.size();
		r.Destination.Buffer.Resource = buf; r.Destination.Buffer.Offset = 0; r.Destination.Buffer.Size = (UINT32)kBytes;
		r.UncompressedSize = (UINT32)kBytes;
		q->EnqueueRequest(&r);
		fence->SetEventOnCompletion(1, ev);
		q->EnqueueSignal(fence, 1);
		q->Submit();
		if (WaitForSingleObject(ev, 5000) != WAIT_OBJECT_0) { std::cout << "[DStorage]\tself-test: request did not complete" << std::endl; break; }
		DSTORAGE_ERROR_RECORD rec{}; q->RetrieveErrorRecord(&rec);
		if (rec.FailureCount) { std::cout << "[DStorage]\tself-test: request failed hr 0x" << std::hex << (unsigned)rec.FirstFailure.HResult << std::dec << std::endl; break; }
		// Read it back on our own queue.
		D3D12_COMMAND_QUEUE_DESC cqd{};
		if (FAILED(d3d->CreateCommandQueue(&cqd, IID_PPV_ARGS(&cq)))) break;
		if (FAILED(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&ca)))) break;
		if (FAILED(d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, ca, nullptr, IID_PPV_ARGS(&cl)))) break;
		cl->CopyBufferRegion(rb, 0, buf, 0, kBytes);
		cl->Close();
		ID3D12CommandList* lists[] = { cl };
		cq->ExecuteCommandLists(1, lists);
		fence->SetEventOnCompletion(2, ev);
		cq->Signal(fence, 2);
		if (WaitForSingleObject(ev, 5000) != WAIT_OBJECT_0) { std::cout << "[DStorage]	self-test: readback copy did not complete" << std::endl; break; }
		void* mp = nullptr;
		if (FAILED(rb->Map(0, nullptr, &mp)) || !mp) { std::cout << "[DStorage]	self-test: map failed" << std::endl; break; }
		ok = memcmp(mp, raw.data(), kBytes) == 0;
		if (!ok)
		{
			size_t zeros = 0;
			for (size_t i = 0; i < kBytes; ++i) if (!((const unsigned char*)mp)[i]) ++zeros;
			std::cout << "[DStorage]\tself-test readback: " << zeros << "/" << kBytes << " zero bytes" << std::endl;
		}
		D3D12_RANGE none{ 0, 0 }; rb->Unmap(0, &none);
	} while (false);
	if (cl) cl->Release(); if (ca) ca->Release(); if (cq) cq->Release();
	if (fence) fence->Release(); if (file) { file->Close(); file->Release(); } if (q) { q->Close(); q->Release(); }
	if (rb) rb->Release(); if (buf) buf->Release();
	CloseHandle(ev);
	{
		boost::system::error_code ec;
		boost::filesystem::remove(testFile, ec);
	}
	return ok;
}

std::string NukeDiligent::Impl::DStor::MachineKey() const
{
	std::string key;
	IDXGIFactory4* fac = nullptr;
	if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&fac))) && fac)
	{
		IDXGIAdapter1* ad = nullptr;
		if (SUCCEEDED(fac->EnumAdapterByLuid(d3d->GetAdapterLuid(), IID_PPV_ARGS(&ad))) && ad)
		{
			DXGI_ADAPTER_DESC1 d{}; ad->GetDesc1(&d);
			char desc[256] = { 0 }; WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, desc, sizeof(desc), nullptr, nullptr);
			LARGE_INTEGER umd{}; ad->CheckInterfaceSupport(__uuidof(IDXGIDevice), &umd);
			char buf[512];
			snprintf(buf, sizeof(buf), "%s|%04x:%04x|%llu", desc, d.VendorId, d.DeviceId, (unsigned long long)umd.QuadPart);
			key = buf;
			ad->Release();
		}
		fac->Release();
	}
	// + the DirectStorage runtime build (the file version of the dll we loaded).
	wchar_t path[MAX_PATH]; DWORD n = GetModuleFileNameW(dll, path, MAX_PATH);
	DWORD h = 0, sz = n ? GetFileVersionInfoSizeW(path, &h) : 0;
	if (sz)
	{
		std::vector<char> data(sz);
		VS_FIXEDFILEINFO* ffi = nullptr; UINT len = 0;
		if (GetFileVersionInfoW(path, 0, sz, data.data()) && VerQueryValueW(data.data(), L"\\", (void**)&ffi, &len) && ffi)
		{
			char buf[64];
			snprintf(buf, sizeof(buf), "|ds %u.%u.%u.%u", HIWORD(ffi->dwFileVersionMS), LOWORD(ffi->dwFileVersionMS), HIWORD(ffi->dwFileVersionLS), LOWORD(ffi->dwFileVersionLS));
			key += buf;
		}
	}
	return key;
}

boost::filesystem::path NukeDiligent::Impl::DStor::VerdictPath()
{
	return boost::filesystem::path(nuke::Config::writableDir()) / "config" / "dstorage.json";
}

void NukeDiligent::Impl::StorageInit()
{
	if (!useD3D12 || dstor) return;
	const nuke::Config* cfg = nuke::Config::getSingleton();
	if (!cfg->ioDirectStorage || std::getenv("NUKE_NO_DSTORAGE"))
	{
		std::cout << "[DStorage]\tdisabled (" << (cfg->ioDirectStorage ? "NUKE_NO_DSTORAGE" : "io.directStorage=false") << ")" << std::endl;
		return;
	}
	RefCntAutoPtr<IRenderDeviceD3D12> d12(device, IID_RenderDeviceD3D12);
	if (!d12 || !d12->GetD3D12Device()) return;

	std::unique_ptr<DStor> ds(new DStor());
	ds->impl = this;
	ds->d3d  = d12->GetD3D12Device();
	// Loaded by name, never imported: a dist without dstorage.dll (or a machine where it fails)
	// keeps the CPU path instead of failing to load the renderer.
	ds->dll = LoadLibraryW(L"dstorage.dll");
	if (!ds->dll) { std::cout << "[DStorage]\tdstorage.dll not found — CPU pak reads" << std::endl; return; }
	const int stagingMB = cfg->ioStagingMB < 1 ? 32 : cfg->ioStagingMB;
	const std::string machine = ds->MachineKey();
	int startMode = DStor::kModeGpu;
	bool verdictKnown = false;
	if (!cfg->ioGpuDecompression || std::getenv("NUKE_DSTORAGE_CPU")) { startMode = DStor::kModeCpu; verdictKnown = true; }
	else
	{
		// A remembered verdict for THIS adapter/driver/runtime skips the doomed attempts on
		// later boots; a driver or runtime update changes the key and re-tests.
		boost::filesystem::ifstream f(DStor::VerdictPath());
		if (f)
		{
			nlohmann::json j = nlohmann::json::parse(f, nullptr, false);
			if (j.is_object() && j.value("machine", std::string()) == machine && j.contains("ioMode"))
			{
				verdictKnown = true;
				startMode = j.value("ioMode", (int)DStor::kModeGpu);
				if (startMode != DStor::kModeGpu)
					std::cout << "[DStorage]\tremembered verdict (config/dstorage.json): "
					          << (startMode == DStor::kModeGpuBuffered ? "GPU decompression + buffered file IO" : "CPU inflate") << std::endl;
			}
		}
	}
	if (!ds->CreateRuntime(startMode, stagingMB)) { FreeLibrary(ds->dll); return; }
	if (!verdictKnown)
	{
		// Walk the ladder on the REAL file path: unbuffered GPU -> buffered GPU -> CPU.
		// (Seen live: Win10 19045 + DirectStorage 1.3 + RTX 5090 — the unbuffered file staging
		// feeds the GPU decompressor garbage with S_OK; buffered file IO decodes perfectly.)
		while (ds->gpuDecomp && ds->ioMode < DStor::kModeCpu && !ds->SelfTestGpuDecompression())
		{
			const int next = ds->ioMode + 1;
			std::cout << "[DStorage]\tGPU decompression self-test FAILED ("
			          << (ds->ioMode == DStor::kModeGpu ? "unbuffered file path" : "buffered file path")
			          << " inflated to garbage with S_OK) — trying "
			          << (next == DStor::kModeGpuBuffered ? "buffered file IO" : "CPU inflate") << std::endl;
			ds->DestroyRuntime();
			if (!ds->CreateRuntime(next, stagingMB)) { FreeLibrary(ds->dll); return; }
		}
		if (ds->gpuDecomp && ds->ioMode < DStor::kModeCpu) std::cout << "[DStorage]\tGPU decompression self-test passed" << std::endl;
		nlohmann::json j; j["machine"] = machine; j["ioMode"] = ds->ioMode;
		boost::system::error_code ec;
		boost::filesystem::create_directories(DStor::VerdictPath().parent_path(), ec);
		boost::filesystem::ofstream o(DStor::VerdictPath(), std::ios::trunc);
		if (o) o << j.dump(2);
	}
	ds->verify = std::getenv("NUKE_DSTORAGE_VERIFY") != nullptr;
	ds->kick = CreateEventW(nullptr, FALSE, FALSE, nullptr);
	ds->quit = CreateEventW(nullptr, TRUE, FALSE, nullptr);
	DStor* raw = ds.release();
	raw->waiter = std::thread([raw]() { raw->WaiterLoop(); });
	dstor = raw;
	nuke::Storage::SetProvider(raw);
	std::cout << "[DStorage]\tready: staging " << stagingMB << " MB, GDeflate on "
	          << (!raw->gpuDecomp || raw->ioMode == DStor::kModeCpu ? "CPU"
	              : raw->ioMode == DStor::kModeGpuBuffered ? "GPU (buffered file IO)" : "GPU") << std::endl;
}

void NukeDiligent::Impl::StorageShutdown()
{
	if (!dstor) return;
	nuke::Storage::SetProvider(nullptr);
	DStor* ds = dstor;
	// Let every issued request land (the destinations are live GPU resources), then tear down.
	ds->SubmitAll();
	for (StorQueue* q : { &ds->qTex, &ds->qLow, &ds->qMem })
	{
		const uint64_t last = q->next - 1;
		if (last && q->fence->GetCompletedValue() < last)
		{
			q->fence->SetEventOnCompletion(last, q->event);
			WaitForSingleObject(q->event, 10000);
		}
	}
	SetEvent(ds->quit);
	if (ds->waiter.joinable()) ds->waiter.join();
	{
		std::lock_guard<std::mutex> l(ds->lock);
		for (auto& j : ds->texDone) { if (j->tex) Trash(j->tex); if (j->staging) Trash(j->staging); }
		ds->texDone.clear();
		for (StorQueue* q : { &ds->qTex, &ds->qLow, &ds->qMem })
		{
			for (auto& m : q->mem) { std::string none; m->done(false, none); }
			q->mem.clear();
			for (auto& j : q->tex) { if (j->tex) Trash(j->tex); if (j->staging) Trash(j->staging); }
			q->tex.clear();
		}
	}
	ds->DestroyRuntime();
	CloseHandle(ds->kick); CloseHandle(ds->quit);
	FreeLibrary(ds->dll);
	std::cout << "[DStorage]\tserved " << ds->statReq.load() << " requests, " << (ds->statBytes.load() >> 20) << " MB" << std::endl;
	delete ds;
	dstor = nullptr;
}

// ---- memory reads (the content scan, documents) ---------------------------------------------

bool NukeDiligent::Impl::DStor::ReadAsync(const Package::Location& loc, int priority, const nuke::Storage::ReadDone& done)
{
	(void)priority;   // memory reads share one NORMAL queue; the scan issues them in bulk
	const Package::Entry& e = loc.entry;
	if (!DirectDecodable(e)) return false;   // zstd/zlib blocks or a v1 pak: CPU
	std::lock_guard<std::mutex> l(lock);
	IDStorageFile* f = File(loc.pakPath);
	if (!f) return false;
	std::unique_ptr<StorMemJob> job(new StorMemJob());
	job->loc = loc;
	job->done = done;
	job->blocks.resize(e.blocks.size());
	for (size_t b = 0; b < e.blocks.size(); ++b) job->blocks[b].resize(e.blocks[b].rawSize);
	for (size_t b = 0; b < e.blocks.size(); ++b)
	{
		const Package::Block& blk = e.blocks[b];
		if (blk.rawSize == 0) continue;
		DSTORAGE_REQUEST r{};
		r.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
		r.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_MEMORY;
		r.Options.CompressionFormat = Format(blk);
		r.Source.File.Source = f;
		r.Source.File.Offset = BlockFileOffset(e, b);
		r.Source.File.Size   = blk.packSize;
		r.Destination.Memory.Buffer = &job->blocks[b][0];
		r.Destination.Memory.Size   = blk.rawSize;
		r.UncompressedSize = blk.rawSize;
		qMem.q->EnqueueRequest(&r);
		job->bytes += blk.rawSize;
	}
	job->status = TakeStatus();
	qMem.q->EnqueueStatus(status, job->status);
	job->fence = qMem.next++;
	qMem.q->EnqueueSignal(qMem.fence, job->fence);
	qMem.dirty = true;
	++qMem.inflight;
	qMem.mem.push_back(std::move(job));
	SetEvent(kick);
	return true;
}

// ---- textures ----------------------------------------------------------------------------------

// Queue the mips [base..] of a pak-resident texture into a fresh GPU texture. False = nothing
// queued (the caller takes the CPU path).
bool NukeDiligent::Impl::StorageRequestTex(Texture* t, int base, bool low)
{
	if (!dstor || !t->pakSource) return false;
	const Package::Entry& e = t->pakSource->entry;
	if (e.layout != Texture::kPakLayout || e.blocks.size() < 2) return false;
	const bool bc = (t->format == Texture::FMT_BC1 || t->format == Texture::FMT_BC3 || t->format == Texture::FMT_BC5);
	const int mips = bc ? std::max(1, t->mipCount) : 1;
	if (base < 0) base = 0;
	if (base > mips - 1) base = mips - 1;
	base = AlignedBase(t, base);

	std::unique_ptr<StorTexJob> job(new StorTexJob());
	job->t = t; job->base = base;
	{
		TextureDesc td; td.Type = RESOURCE_DIM_TEX_2D;
		td.Width  = (Uint32)std::max(1, t->width  >> base);
		td.Height = (Uint32)std::max(1, t->height >> base);
		td.MipLevels = (Uint32)(mips - base); td.BindFlags = BIND_SHADER_RESOURCE; td.Usage = USAGE_DEFAULT;
		td.Format = !bc ? TEX_FORMAT_RGBA8_UNORM
		          : (t->format == Texture::FMT_BC1) ? TEX_FORMAT_BC1_UNORM
		          : (t->format == Texture::FMT_BC5) ? TEX_FORMAT_BC5_UNORM : TEX_FORMAT_BC3_UNORM;
		device->CreateTexture(td, nullptr, &job->tex);
		if (!job->tex) return false;
	}
	DStor* ds = dstor;

	// Entries DirectStorage cannot inflate (zstd/zlib paks) read on the pool; same adopter.
	if (!DirectDecodable(e))
	{
		std::shared_ptr<StorTexJob> sj(job.release());
		nuke::Jobs::Schedule([ds, sj]()
		{
			ds->CpuFallbackTex(*sj);
			std::lock_guard<std::mutex> l2(ds->lock);
			ds->texDone.push_back(std::unique_ptr<StorTexJob>(new StorTexJob(std::move(*sj))));
		});
		return true;
	}

	// The blocks that carry mips >= base land in ONE VRAM staging buffer (each block at a
	// 512-aligned offset, in the footprint layout the cook wrote); the adopter copies buffer ->
	// texture on the graphics queue. DirectStorage never touches the texture itself: no
	// cross-queue state hazards, and the GPU-inflate-into-texture path of the runtime (which
	// silently delivers zeros on some driver/runtime pairs) is never used.
	uint64_t total = 0;
	for (size_t b = 1; b < e.blocks.size(); ++b)
	{
		const Package::Block& blk = e.blocks[b];
		if (blk.meta[0] == 0xFFFFFFFFu || blk.rawSize == 0) continue;
		if (blk.meta[1] == 0) { if ((int)blk.meta[0] < base) continue; }
		else if ((int)(blk.meta[0] + blk.meta[1]) <= base) continue;
		const uint64_t off = Align(total, Texture::kPlaceAlign);
		job->placed.push_back({ (int)b, off });
		total = off + blk.rawSize;
	}
	if (job->placed.empty()) return false;
	{
		BufferDesc bd; bd.Name = "dstorage staging"; bd.Size = total; bd.Usage = USAGE_DEFAULT; bd.BindFlags = BIND_NONE;
		device->CreateBuffer(bd, nullptr, &job->staging);
		if (!job->staging) return false;
	}
	ID3D12Resource* res = nullptr;
	{
		RefCntAutoPtr<IBufferD3D12> b12(job->staging, IID_BufferD3D12);
		Uint64 dataOff = 0;
		if (b12) res = b12->GetD3D12Buffer(dataOff, nullptr);
		if (!res || dataOff != 0) return false;
	}

	std::lock_guard<std::mutex> l(ds->lock);
	IDStorageFile* f = ds->File(t->pakSource->pakPath);
	if (!f) return false;
	StorQueue& q = low ? ds->qLow : ds->qTex;
	for (auto& pl : job->placed)
	{
		const Package::Block& blk = e.blocks[pl.first];
		DSTORAGE_REQUEST r{};
		r.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
		r.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_BUFFER;
		r.Options.CompressionFormat = Format(blk);
		r.Source.File.Source = f;
		r.Source.File.Offset = BlockFileOffset(e, pl.first);
		r.Source.File.Size   = blk.packSize;
		r.Destination.Buffer.Resource = res;
		r.Destination.Buffer.Offset   = pl.second;
		r.Destination.Buffer.Size     = blk.rawSize;
		r.UncompressedSize = blk.rawSize;
		q.q->EnqueueRequest(&r);
		job->bytes += blk.rawSize;
	}
	job->status = ds->TakeStatus();
	q.q->EnqueueStatus(ds->status, job->status);
	job->fence = q.next++;
	q.q->EnqueueSignal(q.fence, job->fence);
	q.dirty = true;
	++q.inflight;
	q.tex.push_back(std::move(job));
	SetEvent(ds->kick);
	return true;
}


// Whole-texture CPU path: inflate the GPU blocks on this thread (the adopter uploads them).
void NukeDiligent::Impl::DStor::CpuFallbackTex(StorTexJob& j)
{
	j.cpuBlocks.clear();
	Package::Location loc; loc.pakPath = j.t->pakSource->pakPath; loc.entry = j.t->pakSource->entry;
	std::vector<std::string> blocks;
	if (!Package::ReadBlocks(loc, blocks)) { j.failed = true; return; }
	for (size_t b = 1; b < blocks.size(); ++b)
		j.cpuBlocks.push_back({ (int)b, std::move(blocks[b]) });
	j.failed = false;
}

// ---- completion thread -------------------------------------------------------------------------

void NukeDiligent::Impl::DStor::Complete(StorQueue& q)
{
	const uint64_t done = q.fence->GetCompletedValue();
	std::vector<std::unique_ptr<StorMemJob>> memOut;
	std::vector<std::unique_ptr<StorTexJob>> texOut;
	{
		std::lock_guard<std::mutex> l(lock);
		while (!q.mem.empty() && q.mem.front()->fence <= done) { memOut.push_back(std::move(q.mem.front())); q.mem.pop_front(); --q.inflight; }
		while (!q.tex.empty() && q.tex.front()->fence <= done) { texOut.push_back(std::move(q.tex.front())); q.tex.pop_front(); --q.inflight; }
	}
	if (memOut.empty() && texOut.empty()) return;
	// Failures are recorded per queue, not per request: everything that landed in this batch
	// re-reads on the CPU (the same blocks — idempotent), the rest of the pipeline never notices.
	DSTORAGE_ERROR_RECORD rec{};
	q.q->RetrieveErrorRecord(&rec);
	if (rec.FailureCount > 0)
		std::cout << "[DStorage]\t" << rec.FailureCount << " request(s) failed (hr 0x" << std::hex
		          << (unsigned)rec.FirstFailure.HResult << std::dec << ", first: command " << (int)rec.FirstFailure.CommandType << ")" << std::endl;
	auto jobOk = [&](uint32_t slot, const char* what)
	{
		const HRESULT hr = status->GetHResult(slot);
		if (SUCCEEDED(hr)) return true;
		std::cout << "[DStorage]	" << what << " failed: hr 0x" << std::hex << (unsigned)hr << std::dec << " — CPU re-read" << std::endl;
		return false;
	};
	for (auto& m : memOut)
	{
		std::string out;
		bool ok = jobOk(m->status, m->loc.entry.path.c_str()) && Package::Uncook(m->loc.entry, m->blocks, out)
		          && Package::Crc32(out.data(), out.size()) == m->loc.entry.crc;
		if (ok) { ++statReq; statBytes += m->bytes; }
		else
		{
			std::vector<std::string> blocks;
			ok = Package::ReadBlocks(m->loc, blocks) && Package::Uncook(m->loc.entry, blocks, out)
			     && Package::Crc32(out.data(), out.size()) == m->loc.entry.crc;
		}
		m->done(ok, out);
	}
	for (auto& j : texOut)
	{
		if (!jobOk(j->status, j->t->pakSource->entry.path.c_str())) { j->placed.clear(); CpuFallbackTex(*j); }
		else { ++statReq; statBytes += j->bytes; }
		std::lock_guard<std::mutex> l(lock);
		texDone.push_back(std::move(j));
	}
}

void NukeDiligent::Impl::DStor::WaiterLoop()
{
	for (;;)
	{
		// Arm each queue's fence at its oldest outstanding value; wake on quit / new work too.
		HANDLE waits[5] = { quit, kick, nullptr, nullptr, nullptr };
		DWORD n = 2;
		{
			std::lock_guard<std::mutex> l(lock);
			for (StorQueue* q : { &qMem, &qTex, &qLow })
			{
				uint64_t oldest = 0;
				if (!q->mem.empty()) oldest = q->mem.front()->fence;
				if (!q->tex.empty() && (!oldest || q->tex.front()->fence < oldest)) oldest = q->tex.front()->fence;
				if (!oldest) continue;
				if (q->fence->GetCompletedValue() >= oldest) SetEvent(q->event);
				else q->fence->SetEventOnCompletion(oldest, q->event);
				waits[n++] = q->event;
			}
		}
		const DWORD w = WaitForMultipleObjects(n, waits, FALSE, 250);
		if (w == WAIT_OBJECT_0) return;                 // quit
		if (w == WAIT_OBJECT_0 + 1) continue;           // kick: re-arm with the new jobs
		Complete(qMem); Complete(qTex); Complete(qLow);
	}
}

// ---- render-thread adoption --------------------------------------------------------------------

void NukeDiligent::Impl::StoragePump()
{
	if (!dstor) return;
	DStor* ds = dstor;
	std::vector<std::unique_ptr<StorTexJob>> landed;
	std::vector<Texture*> pre;
	{
		std::lock_guard<std::mutex> l(ds->lock);
		landed.swap(ds->texDone);
		pre.swap(ds->prefetch);
	}
	for (auto& jp : landed)
	{
		StorTexJob& j = *jp;
		Texture* t = j.t;
		if (!storPendingTex.count(t)) { if (j.tex) Trash(j.tex); if (j.staging) Trash(j.staging); continue; }   // invalidated meanwhile
		storPendingTex.erase(t);
		if (j.failed || !j.tex)
		{
			std::cout << "[DStorage]\ttexture '" << t->guid << "' could not be read from its pak — CPU decode on the next draw" << std::endl;
			if (j.tex) Trash(j.tex);
			if (j.staging) Trash(j.staging);
			storFailed.insert(t);
			continue;
		}
		const bool bc = (t->format == Texture::FMT_BC1 || t->format == Texture::FMT_BC3 || t->format == Texture::FMT_BC5);
		const TextureDesc& td = j.tex->GetDesc();
		// Per block: the mips (or the row range) it holds -> copy regions into the texture.
		struct Region { int mip; uint32_t y0, rows; uint64_t off; };   // rows in block rows (BC) / texel rows
		auto regions = [&](const Package::Block& blk, uint64_t blockBase, std::vector<Region>& out)
		{
			if (blk.meta[1] == 0)
			{
				const int m = (int)blk.meta[0];
				if (m >= j.base) out.push_back({ m, blk.meta[2] >> 16, blk.meta[2] & 0xFFFFu, blockBase });
				return;
			}
			uint64_t off = 0;
			for (int m = (int)blk.meta[0]; m < (int)(blk.meta[0] + blk.meta[1]); ++m)
			{
				Texture::MipGeom g = t->MipGeometry(m);
				const uint64_t pitch = Align(g.rowBytes, Texture::kPitchAlign);
				off = Align(off, Texture::kPlaceAlign);
				if (m >= j.base) out.push_back({ m, 0, g.rows, blockBase + off });
				off += pitch * (g.rows - 1) + g.rowBytes;
			}
		};
		if (j.staging && !j.placed.empty())
		{
			// Buffer -> texture on the graphics queue with placed footprints (native D3D12: Diligent
			// has no buffer->texture copy). States are declared through Diligent so its tracking
			// stays truthful; a copy changes no pipeline state, so nothing to invalidate.
			StateTransitionDesc barriers[2] = {
				StateTransitionDesc(j.tex,     RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_COPY_DEST,   STATE_TRANSITION_FLAG_UPDATE_STATE),
				StateTransitionDesc(j.staging, RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_COPY_SOURCE, STATE_TRANSITION_FLAG_UPDATE_STATE) };
			context->TransitionResourceStates(2, barriers);
			RefCntAutoPtr<IDeviceContextD3D12> c12(context, IID_DeviceContextD3D12);
			RefCntAutoPtr<ITextureD3D12> t12(j.tex, IID_TextureD3D12);
			RefCntAutoPtr<IBufferD3D12>  b12(j.staging, IID_BufferD3D12);
			Uint64 dataOff = 0;
			ID3D12GraphicsCommandList* cl = c12 ? c12->GetD3D12CommandList() : nullptr;
			ID3D12Resource* dst = t12 ? t12->GetD3D12Texture() : nullptr;
			ID3D12Resource* src = b12 ? b12->GetD3D12Buffer(dataOff, nullptr) : nullptr;
			if (!cl || !dst || !src) { Trash(j.tex); Trash(j.staging); storFailed.insert(t); continue; }
			const DXGI_FORMAT fmt = td.Format == TEX_FORMAT_BC1_UNORM ? DXGI_FORMAT_BC1_UNORM
			                      : td.Format == TEX_FORMAT_BC3_UNORM ? DXGI_FORMAT_BC3_UNORM
			                      : td.Format == TEX_FORMAT_BC5_UNORM ? DXGI_FORMAT_BC5_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM;
			for (auto& pl : j.placed)
			{
				std::vector<Region> regs;
				regions(t->pakSource->entry.blocks[pl.first], pl.second, regs);
				for (const Region& rg : regs)
				{
					Texture::MipGeom g = t->MipGeometry(rg.mip);
					const uint32_t y0 = bc ? rg.y0 * 4 : rg.y0;
					// BC footprints are block-aligned (a 1x1 mip is described as 4x4); the copy clips
					// to the destination subresource.
					const uint32_t rows = std::min<uint32_t>(rg.rows, g.rows - rg.y0);
					const uint32_t fw = bc ? (uint32_t)((g.w + 3) / 4) * 4 : (uint32_t)g.w;
					const uint32_t fh = bc ? rows * 4 : rows;
					D3D12_TEXTURE_COPY_LOCATION dl{}; dl.pResource = dst; dl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
					dl.SubresourceIndex = (UINT)(rg.mip - j.base);
					D3D12_TEXTURE_COPY_LOCATION sl{}; sl.pResource = src; sl.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
					sl.PlacedFootprint.Offset = dataOff + rg.off;
					sl.PlacedFootprint.Footprint.Format   = fmt;
					sl.PlacedFootprint.Footprint.Width    = fw;
					sl.PlacedFootprint.Footprint.Height   = fh;
					sl.PlacedFootprint.Footprint.Depth    = 1;
					sl.PlacedFootprint.Footprint.RowPitch = (UINT)Align(g.rowBytes, Texture::kPitchAlign);
					cl->CopyTextureRegion(&dl, 0, y0, 0, &sl, nullptr);
				}
			}
		}
		// Blocks that came to memory (the CPU path): upload the mips >= base from the pitched
		// bytes — Stride carries the pitch, no repacking.
		for (auto& cb : j.cpuBlocks)
		{
			const Package::Block& blk = t->pakSource->entry.blocks[cb.first];
			const std::string& data = cb.second;
			if (data.size() < t->PakBlockPitchedSize(blk)) { j.failed = true; break; }
			std::vector<Region> regs;
			regions(blk, 0, regs);
			for (const Region& rg : regs)
			{
				Texture::MipGeom g = t->MipGeometry(rg.mip);
				Box box; box.MinX = 0; box.MaxX = (Uint32)g.w;
				box.MinY = bc ? rg.y0 * 4 : rg.y0; box.MaxY = std::min<Uint32>((Uint32)g.h, bc ? (rg.y0 + rg.rows) * 4 : rg.y0 + rg.rows);
				TextureSubResData sd; sd.pData = data.data() + rg.off; sd.Stride = Align(g.rowBytes, Texture::kPitchAlign);
				context->UpdateTexture(j.tex, (Uint32)(rg.mip - j.base), 0, box, sd, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			}
		}
		if (j.staging) Trash(j.staging);
		if (j.failed) { Trash(j.tex); storFailed.insert(t); continue; }
		if (ds->verify) StorageVerify(t, j.tex, j.base);
		auto it = texCache.find(t);
		if (it != texCache.end()) { Trash(it->second); it->second = j.tex; }
		else texCache.emplace(t, j.tex);
		auto st = streamTex.find(t);
		if (st != streamTex.end()) st->second.residentBase = j.base;
		++storLanded;
	}
	// Scan-registered textures: LOW-priority bulk load (tail mips only when streaming is on),
	// bounded by what the queue holds without blocking this thread.
	for (size_t i = 0; i < pre.size(); ++i)
	{
		Texture* t = pre[i];
		{
			std::lock_guard<std::mutex> l(ds->lock);
			if (ds->qLow.inflight >= 2048)
			{
				ds->prefetch.insert(ds->prefetch.begin(), pre.begin() + i, pre.end());
				break;
			}
		}
		if (!t->pakSource || texCache.count(t) || storPendingTex.count(t) || storFailed.count(t)) continue;
		int base = 0;
		if (streamBudget > 0 && StreamEligible(t))
		{
			base = StreamTailBase(t);
			streamTex[t].residentBase = base;
		}
		if (StorageRequestTex(t, base, true)) storPendingTex.insert(t);
	}
	ds->SubmitAll();
}


// NUKE_DSTORAGE_VERIFY=1: copy the landed texture back and compare every mip with the CPU decode
// of the same pak bytes — proves the cook's footprint layout matches what the GPU received.
void NukeDiligent::Impl::StorageVerify(Texture* t, ITexture* tex, int base)
{
	if (!t->EnsurePixels()) { std::cout << "[DStorage]\tverify: no CPU pixels for '" << t->guid << "'" << std::endl; return; }
	const TextureDesc& td = tex->GetDesc();
	TextureDesc sd = td; sd.Usage = USAGE_STAGING; sd.CPUAccessFlags = CPU_ACCESS_READ; sd.BindFlags = BIND_NONE;
	RefCntAutoPtr<ITexture> staging;
	device->CreateTexture(sd, nullptr, &staging);
	if (!staging) return;
	for (Uint32 m = 0; m < td.MipLevels; ++m)
	{
		CopyTextureAttribs c(tex, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, staging, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		c.SrcMipLevel = m; c.DstMipLevel = m;
		context->CopyTexture(c);
	}
	context->WaitForIdle();
	size_t bad = 0, total = 0;
	for (Uint32 m = 0; m < td.MipLevels; ++m)
	{
		Texture::MipGeom g = t->MipGeometry((int)m + base);
		MappedTextureSubresource mp;
		context->MapTextureSubresource(staging, m, 0, MAP_READ, MAP_FLAG_DO_NOT_WAIT, nullptr, mp);
		if (!mp.pData) { std::cout << "[DStorage]\tverify: map failed mip " << m << std::endl; continue; }
		const unsigned char* cpu = t->pixels.data() + t->MipOffset((int)m + base);
		size_t badHere = 0;
		for (uint32_t r = 0; r < g.rows; ++r)
		{
			const unsigned char* gpuRow = (const unsigned char*)mp.pData + r * mp.Stride;
			if (memcmp(gpuRow, cpu + (size_t)r * g.rowBytes, g.rowBytes) != 0) { ++bad; ++badHere; }
			++total;
		}
		if (badHere)
		{
			// Diagnostics: what the GPU holds vs what the pak says, and whether the CPU row exists anywhere in the mip.
			const unsigned char* g0 = (const unsigned char*)mp.pData;
			bool zero = true;
			for (uint32_t r = 0; r < g.rows && zero; ++r) for (uint32_t x = 0; x < g.rowBytes; ++x) if (g0[r * mp.Stride + x]) { zero = false; break; }
			char hex[2][64];
			for (int k = 0; k < 2; ++k) { const unsigned char* src = k ? cpu : g0; for (int x = 0; x < 16; ++x) snprintf(hex[k] + x * 3, 4, "%02x ", src[x]); }
			long foundRow = -1;
			for (uint32_t r = 0; r < g.rows && foundRow < 0; ++r)
				if (memcmp(g0 + r * mp.Stride, cpu, std::min<uint32_t>(g.rowBytes, 64)) == 0) foundRow = (long)r;
			std::cout << "[DStorage]	  mip " << m << " (" << g.w << "x" << g.h << ", stride " << mp.Stride << "): " << badHere << "/" << g.rows
			          << " rows differ; gpu " << (zero ? "ALL ZERO" : "has data") << "; gpu[0]=" << hex[0] << " cpu[0]=" << hex[1]
			          << "; cpu row0 found at gpu row " << foundRow << std::endl;
		}
		context->UnmapTextureSubresource(staging, m, 0);
	}
	std::cout << "[DStorage]\tverify '" << t->guid << "' base " << base << ": " << (total - bad) << "/" << total
	          << " rows identical" << (bad ? "  <-- MISMATCH" : "") << std::endl;
	Trash(staging);
	std::vector<unsigned char>().swap(t->pixels);   // back to pak-resident
}

#else
void NukeDiligent::Impl::StorageInit() {}
void NukeDiligent::Impl::StorageShutdown() {}
void NukeDiligent::Impl::StoragePump() {}
bool NukeDiligent::Impl::StorageRequestTex(nuke::Texture*, int, bool) { return false; }
void NukeDiligent::Impl::StorageVerify(nuke::Texture*, ITexture*, int) {}
#endif
