#include "NukeDiligentImpl.h"
#include <API/Model/Profiler.h>
#include <map>

// GPU pass timings. A duration query is two timestamps written into the command stream, so the
// cost is negligible, and the result only exists after the GPU has run that far — it is read
// from a ring several frames back and never blocks. Passes are flat, not nested: opening one
// closes the previous, which is exactly how the frame is structured (shadows, then the camera,
// then post, then UI). Slices of the same pass (shadow cascades, several cameras) SUM into one
// number. Everything lands in the engine profiler as "gpu.<name>", next to the CPU phases.

void NukeDiligent::Impl::GpuFrame()
{
	if (!device) return;
	static bool probed = false;
	if (!probed)
	{
		probed = true;
		gpuTimers = device->GetDeviceInfo().Features.DurationQueries == DEVICE_FEATURE_STATE_ENABLED;
		if (!gpuTimers)
			std::cout << "[NukeDiligent]\tGPU timings unavailable (no duration queries on this device)" << std::endl;
	}
	if (!gpuTimers) return;

	GpuPassEnd();   // a pass that forgot to close must not leak into the next frame

	// Read the OLDEST ring entry: the GPU is done with it by now. Not ready = drop the sample;
	// a missing number is better than a stall.
	const int old = (gpuCur + 1) % kGpuRing;
	std::map<std::string, double> sum;
	for (GpuScope& s : gpuRing[old])
	{
		if (!s.query || s.name.empty()) continue;
		QueryDataDuration data;
		if (s.query->GetData(&data, sizeof(data), false) && data.Frequency > 0)
			sum[s.name] += (double)data.Duration * 1000.0 / (double)data.Frequency;
		s.name.clear();
	}
	for (const auto& kv : sum) nuke::Profiler::Report("gpu." + kv.first, kv.second);
	gpuCur  = old;
	gpuUsed = 0;
}

void NukeDiligent::Impl::GpuPass(const char* name)
{
	if (!gpuTimers || !context || !name || !*name) return;
	GpuPassEnd();
	std::vector<GpuScope>& ring = gpuRing[gpuCur];
	if (gpuUsed >= ring.size())
	{
		if (ring.size() >= 64) return;   // more passes than any frame should have: drop the sample
		GpuScope s;
		QueryDesc qd;
		qd.Name = "GPU pass duration";
		qd.Type = QUERY_TYPE_DURATION;
		device->CreateQuery(qd, &s.query);
		if (!s.query) { gpuTimers = false; return; }
		ring.push_back(std::move(s));
	}
	GpuScope& s = ring[gpuUsed];
	s.name = name;
	s.open = true;
	context->BeginQuery(s.query);
	gpuOpen = (int)gpuUsed;
	++gpuUsed;
}

void NukeDiligent::Impl::GpuPassEnd()
{
	if (!gpuTimers || gpuOpen < 0 || !context) return;
	std::vector<GpuScope>& ring = gpuRing[gpuCur];
	const int idx = gpuOpen;
	gpuOpen = -1;
	if (idx >= (int)ring.size()) return;
	GpuScope& s = ring[idx];
	if (!s.open || !s.query) return;
	s.open = false;
	context->EndQuery(s.query);
}
