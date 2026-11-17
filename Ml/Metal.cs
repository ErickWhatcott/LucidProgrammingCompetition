using System.Runtime.InteropServices;

namespace Lpc.Ml;

[StructLayout(LayoutKind.Sequential)]
public readonly struct MTLSize(long width, long height, long depth)
{
    public readonly nuint Width = (nuint)width, Height = (nuint)height, Depth = (nuint)depth;
}

public sealed class MetalException(string message) : Exception(message);

/// <summary>
/// Minimal Objective-C / Metal interop (no packages): P/Invoke into Metal.framework and libobjc,
/// with objc_msgSend called through typed unmanaged function pointers. Buffers use shared storage, so
/// their <c>contents</c> pointer is read and written directly from C# (unified memory, no copies).
/// Ownership: everything returned by new*/Create* is +1 and must be <see cref="Release"/>d; command
/// buffers and encoders are autoreleased, so encode inside an <see cref="AutoreleasePool"/>.
/// </summary>
public static unsafe class Metal
{
    const string MetalPath = "/System/Library/Frameworks/Metal.framework/Metal";
    const string ObjcPath = "/usr/lib/libobjc.A.dylib";
    const int CommandBufferStatusError = 5;

    [DllImport(MetalPath)] static extern IntPtr MTLCreateSystemDefaultDevice();
    [DllImport(ObjcPath)] static extern IntPtr objc_getClass(string name);
    [DllImport(ObjcPath)] static extern IntPtr sel_registerName(string name);
    [DllImport(ObjcPath)] static extern IntPtr objc_autoreleasePoolPush();
    [DllImport(ObjcPath)] static extern void objc_autoreleasePoolPop(IntPtr pool);

    static IntPtr msgSend;
    static IntPtr device, queue;
    static string? deviceName;
    static readonly bool available = Probe();

    /// <summary>True on macOS when a Metal device and command queue could be created. Never throws.</summary>
    public static bool Available => available;
    public static IntPtr Device => available ? device : throw new MetalException("Metal is not available.");
    public static IntPtr Queue => available ? queue : throw new MetalException("Metal is not available.");
    public static string DeviceName => deviceName ?? "none";

    // selectors
    static IntPtr selRelease, selNewCommandQueue, selNewBufferWithLength, selContents, selStringWithUtf8, selUtf8String,
        selLocalizedDescription, selNewLibraryWithSource, selNewFunctionWithName, selNewPipeline, selCommandBuffer,
        selComputeCommandEncoder, selSetPipeline, selSetBuffer, selSetBytes, selDispatch, selEndEncoding, selCommit,
        selWait, selStatus, selError, selName;

    static bool Probe()
    {
        try
        {
            if (!OperatingSystem.IsMacOS()) return false;
            // Some macOS versions only enumerate GPUs for command-line processes once CoreGraphics is loaded.
            NativeLibrary.TryLoad("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics", out _);
            var objc = NativeLibrary.Load(ObjcPath);
            msgSend = NativeLibrary.GetExport(objc, "objc_msgSend");
            selRelease = sel_registerName("release");
            selNewCommandQueue = sel_registerName("newCommandQueue");
            selNewBufferWithLength = sel_registerName("newBufferWithLength:options:");
            selContents = sel_registerName("contents");
            selStringWithUtf8 = sel_registerName("stringWithUTF8String:");
            selUtf8String = sel_registerName("UTF8String");
            selLocalizedDescription = sel_registerName("localizedDescription");
            selNewLibraryWithSource = sel_registerName("newLibraryWithSource:options:error:");
            selNewFunctionWithName = sel_registerName("newFunctionWithName:");
            selNewPipeline = sel_registerName("newComputePipelineStateWithFunction:error:");
            selCommandBuffer = sel_registerName("commandBuffer");
            selComputeCommandEncoder = sel_registerName("computeCommandEncoder");
            selSetPipeline = sel_registerName("setComputePipelineState:");
            selSetBuffer = sel_registerName("setBuffer:offset:atIndex:");
            selSetBytes = sel_registerName("setBytes:length:atIndex:");
            selDispatch = sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:");
            selEndEncoding = sel_registerName("endEncoding");
            selCommit = sel_registerName("commit");
            selWait = sel_registerName("waitUntilCompleted");
            selStatus = sel_registerName("status");
            selError = sel_registerName("error");
            selName = sel_registerName("name");

            device = MTLCreateSystemDefaultDevice();
            if (device == IntPtr.Zero) return false;
            queue = Send(device, selNewCommandQueue);
            if (queue == IntPtr.Zero) return false;
            var pool = objc_autoreleasePoolPush();
            try { deviceName = NsString(Send(device, selName)); }
            finally { objc_autoreleasePoolPop(pool); }
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    // ---------------------------------------------------------------- objc_msgSend shapes

    static IntPtr Send(IntPtr o, IntPtr sel) => ((delegate* unmanaged<IntPtr, IntPtr, IntPtr>)msgSend)(o, sel);
    static IntPtr Send(IntPtr o, IntPtr sel, IntPtr a) => ((delegate* unmanaged<IntPtr, IntPtr, IntPtr, IntPtr>)msgSend)(o, sel, a);
    static void SendV(IntPtr o, IntPtr sel) => ((delegate* unmanaged<IntPtr, IntPtr, void>)msgSend)(o, sel);
    static void SendV(IntPtr o, IntPtr sel, IntPtr a) => ((delegate* unmanaged<IntPtr, IntPtr, IntPtr, void>)msgSend)(o, sel, a);

    public static void Release(IntPtr obj)
    {
        if (obj != IntPtr.Zero) SendV(obj, selRelease);
    }

    static string? NsString(IntPtr nsString)
    {
        if (nsString == IntPtr.Zero) return null;
        var utf8 = Send(nsString, selUtf8String);
        return utf8 == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(utf8);
    }

    /// <summary>Autoreleased NSString (call inside a pool).</summary>
    static IntPtr MakeNsString(string s)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(s + "\0");
        fixed (byte* p = bytes)
            return Send(objc_getClass("NSString"), selStringWithUtf8, (IntPtr)p);
    }

    static string ErrorText(IntPtr nsError) =>
        nsError == IntPtr.Zero ? "unknown error" : NsString(Send(nsError, selLocalizedDescription)) ?? "unknown error";

    public readonly struct AutoreleasePool : IDisposable
    {
        readonly IntPtr token;
        AutoreleasePool(IntPtr t) { token = t; }
        public static AutoreleasePool Push() => new(objc_autoreleasePoolPush());
        public void Dispose() { if (token != IntPtr.Zero) objc_autoreleasePoolPop(token); }
    }

    // ---------------------------------------------------------------- resources

    /// <summary>Shared-storage MTLBuffer whose contents pointer is directly usable from C#.</summary>
    public sealed class Buffer : IDisposable
    {
        public IntPtr Handle { get; private set; }
        public void* Contents { get; }
        public long Length { get; }

        public Buffer(long bytes)
        {
            bytes = Math.Max(16, (bytes + 15) & ~15L);
            // MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache = 0
            Handle = ((delegate* unmanaged<IntPtr, IntPtr, nuint, nuint, IntPtr>)msgSend)(Device, selNewBufferWithLength, (nuint)bytes, 0);
            if (Handle == IntPtr.Zero) throw new MetalException($"newBufferWithLength({bytes}) failed (out of GPU memory?).");
            Contents = (void*)Send(Handle, selContents);
            Length = bytes;
        }

        public Span<T> AsSpan<T>(long count) where T : unmanaged
        {
            if (count * sizeof(T) > Length) throw new ArgumentOutOfRangeException(nameof(count));
            return new Span<T>(Contents, checked((int)count));
        }

        public void Dispose()
        {
            if (Handle != IntPtr.Zero) { Release(Handle); Handle = IntPtr.Zero; }
        }
    }

    /// <summary>Compiles MSL source and returns +1 compute pipeline states for the named kernels.</summary>
    public static IntPtr[] CompilePipelines(string source, params string[] kernels)
    {
        using var pool = AutoreleasePool.Push();
        IntPtr err = IntPtr.Zero;
        var lib = ((delegate* unmanaged<IntPtr, IntPtr, IntPtr, IntPtr, IntPtr*, IntPtr>)msgSend)(
            Device, selNewLibraryWithSource, MakeNsString(source), IntPtr.Zero, &err);
        if (lib == IntPtr.Zero) throw new MetalException("MSL compile failed: " + ErrorText(err));
        var result = new IntPtr[kernels.Length];
        try
        {
            for (int i = 0; i < kernels.Length; i++)
            {
                var fn = Send(lib, selNewFunctionWithName, MakeNsString(kernels[i]));
                if (fn == IntPtr.Zero) throw new MetalException($"kernel '{kernels[i]}' not found in library.");
                err = IntPtr.Zero;
                var ps = ((delegate* unmanaged<IntPtr, IntPtr, IntPtr, IntPtr*, IntPtr>)msgSend)(Device, selNewPipeline, fn, &err);
                Release(fn);
                if (ps == IntPtr.Zero) throw new MetalException($"pipeline '{kernels[i]}' failed: " + ErrorText(err));
                result[i] = ps;
            }
        }
        catch
        {
            foreach (var p in result) Release(p);
            throw;
        }
        finally { Release(lib); }
        return result;
    }

    // ---------------------------------------------------------------- command encoding

    /// <summary>Autoreleased command buffer (create inside a pool).</summary>
    public static IntPtr NewCommandBuffer()
    {
        var cb = Send(Queue, selCommandBuffer);
        if (cb == IntPtr.Zero) throw new MetalException("commandBuffer returned nil.");
        return cb;
    }

    public static void Commit(IntPtr commandBuffer) => SendV(commandBuffer, selCommit);
    public static void WaitUntilCompleted(IntPtr commandBuffer) => SendV(commandBuffer, selWait);

    /// <summary>Throws if a completed command buffer reported an error.</summary>
    public static void CheckCompleted(IntPtr commandBuffer)
    {
        long status = (long)Send(commandBuffer, selStatus);
        if (status == CommandBufferStatusError)
            throw new MetalException("GPU command buffer failed: " + ErrorText(Send(commandBuffer, selError)));
    }

    public readonly struct ComputeEncoder
    {
        public readonly IntPtr Handle;
        ComputeEncoder(IntPtr h) { Handle = h; }

        /// <summary>Serial (dispatches run in order) compute encoder; autoreleased.</summary>
        public static ComputeEncoder Begin(IntPtr commandBuffer)
        {
            var h = Send(commandBuffer, selComputeCommandEncoder);
            if (h == IntPtr.Zero) throw new MetalException("computeCommandEncoder returned nil.");
            return new ComputeEncoder(h);
        }

        public void SetPipeline(IntPtr pipeline) => SendV(Handle, selSetPipeline, pipeline);

        public void SetBuffer(Buffer buffer, long offsetBytes, int index) =>
            ((delegate* unmanaged<IntPtr, IntPtr, IntPtr, nuint, nuint, void>)msgSend)(Handle, selSetBuffer, buffer.Handle, (nuint)offsetBytes, (nuint)index);

        public void SetBytes<T>(in T value, int index) where T : unmanaged
        {
            fixed (T* p = &value)
                ((delegate* unmanaged<IntPtr, IntPtr, void*, nuint, nuint, void>)msgSend)(Handle, selSetBytes, p, (nuint)sizeof(T), (nuint)index);
        }

        public void Dispatch(MTLSize threadgroups, MTLSize threadsPerThreadgroup) =>
            ((delegate* unmanaged<IntPtr, IntPtr, MTLSize, MTLSize, void>)msgSend)(Handle, selDispatch, threadgroups, threadsPerThreadgroup);

        /// <summary>1-D dispatch covering <paramref name="threads"/> threads in groups of <paramref name="group"/>.</summary>
        public void Dispatch1D(long threads, int group = 256) =>
            Dispatch(new MTLSize((threads + group - 1) / group, 1, 1), new MTLSize(group, 1, 1));

        public void End() => SendV(Handle, selEndEncoding);
    }
}
