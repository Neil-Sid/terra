// D3DCompile timer for tools/fxc-time.cjs: compiles HLSL files with Chrome's own d3dcompiler_47.dll, the way ANGLE's
// D3D11 backend does (entry "main", macro ANGLE_ENABLE_LOOP_FLATTEN=1), and prints one line per file: ms, ok, bytes.
using System;
using System.IO;
using System.Diagnostics;
using System.Runtime.InteropServices;

public static class FxcTime {
  [ComImport, Guid("8BA5FB08-5195-40e2-AC58-0D989C3A0102"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  public interface ID3DBlob { [PreserveSig] IntPtr GetBufferPointer(); [PreserveSig] UIntPtr GetBufferSize(); }

  [StructLayout(LayoutKind.Sequential)]
  public struct Macro { public IntPtr Name; public IntPtr Definition; }

  [UnmanagedFunctionPointer(CallingConvention.StdCall)]
  delegate int D3DCompileFn(byte[] src, UIntPtr size, [MarshalAs(UnmanagedType.LPStr)] string name, Macro[] defines, IntPtr include,
    [MarshalAs(UnmanagedType.LPStr)] string entry, [MarshalAs(UnmanagedType.LPStr)] string target, uint flags1, uint flags2,
    out ID3DBlob code, out ID3DBlob errors);

  [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)] static extern IntPtr LoadLibrary(string path);
  [DllImport("kernel32", CharSet = CharSet.Ansi)] static extern IntPtr GetProcAddress(IntPtr module, string name);

  static string Text(ID3DBlob b) {
    if (b == null) return "";
    int n = (int)b.GetBufferSize();
    byte[] a = new byte[n];
    Marshal.Copy(b.GetBufferPointer(), a, 0, n);
    return System.Text.Encoding.ASCII.GetString(a).TrimEnd('\0');
  }

  // flags: D3DCOMPILE_* bits (ANGLE: OPTIMIZATION_LEVEL2 = 0xC000)
  public static void Run(string dll, string target, uint flags, string[] files) {
    IntPtr mod = LoadLibrary(dll);
    if (mod == IntPtr.Zero) throw new Exception("cannot load " + dll);
    var compile = (D3DCompileFn)Marshal.GetDelegateForFunctionPointer(GetProcAddress(mod, "D3DCompile"), typeof(D3DCompileFn));
    var macros = new Macro[] {
      new Macro { Name = Marshal.StringToHGlobalAnsi("ANGLE_ENABLE_LOOP_FLATTEN"), Definition = Marshal.StringToHGlobalAnsi("1") },
      new Macro { Name = IntPtr.Zero, Definition = IntPtr.Zero } };
    foreach (string f in files) {
      byte[] src = File.ReadAllBytes(f);
      ID3DBlob code, errors;
      var sw = Stopwatch.StartNew();
      int hr = compile(src, (UIntPtr)src.Length, "C:\\fakepath", macros, IntPtr.Zero, "main", target, flags, 0, out code, out errors);
      sw.Stop();
      long bytes = code == null ? 0 : (long)code.GetBufferSize();
      string err = Text(errors).Replace("\r", "").Replace("\n", " | ");
      if (err.Length > 600) err = err.Substring(0, 600);
      Console.WriteLine(String.Format("{0}\t{1}\t{2}\t{3}\t{4}", (int)sw.ElapsedMilliseconds, hr == 0 ? "ok" : "FAIL 0x" + hr.ToString("X8"), bytes, f, hr == 0 ? "" : err));
      if (code != null) Marshal.ReleaseComObject(code);
      if (errors != null) Marshal.ReleaseComObject(errors);
    }
  }
}
