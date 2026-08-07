// macOS window plumbing for the Vulkan (MoltenVK) backend: Diligent's MacOSNativeWindow
// carries an NSView whose backing layer must be a CAMetalLayer (GetLayer() hands it to
// vkCreateMetalSurfaceEXT). GLFW creates the window with GLFW_NO_API and no layer, so this
// shim attaches the CAMetalLayer to the content view exactly once and returns that view.
#ifdef __APPLE__

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

// AppKit only key-focuses TITLED windows on click; the editor's detached imgui viewports
// are borderless (imgui draws its own chrome) and would NEVER receive focus — imgui then
// treats the viewport as inactive and every click and drag in it is dead. One local event
// monitor process-wide: a click into any of our key-capable borderless windows focuses it,
// exactly what Win32 does for free.
static void EnsureClickFocusMonitor()
{
    static bool installed = false;
    if (installed) return;
    installed = true;
    [NSEvent addLocalMonitorForEventsMatchingMask:(NSEventMaskLeftMouseDown | NSEventMaskRightMouseDown)
        handler:^NSEvent*(NSEvent* e)
        {
            NSWindow* w = e.window;
            if (w && !w.keyWindow && w.canBecomeKeyWindow
                && (w.styleMask & NSWindowStyleMaskTitled) == 0)
                [w makeKeyAndOrderFront:nil];
            return e;   // never swallow — the click still lands in the window
        }];
}

// Shared body: give the window's content view a CAMetalLayer (idempotent) and return it.
// contentsScale = backingScaleFactor is LOAD-BEARING: a fresh CAMetalLayer defaults to
// scale 1 — MoltenVK then sizes drawables at half resolution on Retina.
static void* MetalViewFor(NSWindow* nswin)
{
    EnsureClickFocusMonitor();
    if (!nswin) return nullptr;
    NSView* view = [nswin contentView];
    if (![view.layer isKindOfClass:[CAMetalLayer class]])
    {
        CAMetalLayer* layer = [CAMetalLayer layer];
        layer.contentsScale = nswin.backingScaleFactor;   // retina: framebuffer at native pixels
        [view setLayer:layer];
        [view setWantsLayer:YES];
    }
    return (__bridge void*)view;
}

extern "C" void* NukeCocoaMetalView(GLFWwindow* wnd)
{
    return wnd ? MetalViewFor(glfwGetCocoaWindow(wnd)) : nullptr;
}

// Secondary imgui viewports: imgui_impl_glfw exposes the raw NSWindow (PlatformHandleRaw).
extern "C" void* NukeCocoaMetalViewForNSWindow(void* nswindow)
{
    return MetalViewFor((__bridge NSWindow*)nswindow);
}

#endif // __APPLE__
