// Times fxc (D3DCompile from Chrome's own d3dcompiler_47.dll) on the HLSL ANGLE hands it, outside Chrome: no GPU
// process, no WebGL, one compile at a time, and the DXBC size as a deterministic second measure (notes/compile.md).
//   node tools/fxc-time.cjs <dump.fs.hlsl> [more dumps] [--repeat 2] [--flags 0xC000] [--no-warm] [--mode pass|zero]
//                           [--stub name1,name2 | --stub-each name1,name2]
// ~25 s per terrain FS compile plus a warm-up: keep a run to ~9 compiles under `timeout 300`.
// Input: a --dump file of tools/compile-bench.cjs (the "COMPILER INPUT HLSL" section is used) or plain HLSL.
// --stub: replace those functions' bodies (HLSL names, e.g. f_snoised, or GLSL names: every f_<name>[_types]
// overload) with a trivial one. --stub-each: one compile per name (plus the unstubbed file) in one run.
// --mode pass (default): the stub returns the sum of its scalar arguments (x of vectors) cast to the return type and
// sets its out and inout arguments to it, so the callers' code stays data-dependent: the saving is what the
// function's instances cost (with callees reached only through it). --mode zero: "return (T)0;" and out args 0
// (inout untouched), which also lets fxc fold whatever depends on it (an upper bound).
// Target from the file name (.vs. -> vs_5_0, else ps_5_0); flags default to ANGLE's D3DCOMPILE_OPTIMIZATION_LEVEL2.
const fs = require('fs')
const os = require('os')
const path = require('path')
const { spawnSync } = require('child_process')

const a = process.argv.slice(2), opt = k => a.includes(k) ? a[a.indexOf(k) + 1] : null
const files = a.filter((x, i) => !x.startsWith('--') && !(i > 0 && a[i - 1].startsWith('--')))
const repeat = +(opt('--repeat') || 1), flags = opt('--flags') || '0xC000'
const mode = opt('--mode') || 'pass'
const stub = opt('--stub') ? opt('--stub').split(',') : [], each = opt('--stub-each') ? opt('--stub-each').split(',') : null

const chromeDir = 'C:/Program Files/Google/Chrome/Application'
const ver = fs.readdirSync(chromeDir).filter(d => /^\d+\./.test(d)).sort().pop()
const dll = path.join(chromeDir, ver, 'd3dcompiler_47.dll')
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'fxc-time-'))

const hlslOf = f => { const t = fs.readFileSync(f, 'utf8'), m = t.match(/\/\/ COMPILER INPUT HLSL BEGIN([\s\S]*?)\/\/ COMPILER INPUT HLSL END/); return m ? m[1] : t }
// body of every definition whose name is `n` or `f_n` / `f_n_<types>` -> return (T)0; (void: empty)
function stubbed(src, names) {
  let out = src, hit = []
  for (const n of names) {
    const re = new RegExp(`^([A-Za-z_]\\w*)[ \\t]+(${n.startsWith('f_') ? n : `f_${n}(?:_\\w+)?`})[ \\t]*\\(([^;{}]*)\\)[ \\t]*\\n?[ \\t]*\\{`, 'gm')
    let m
    while ((m = re.exec(out))) {
      const open = m.index + m[0].length - 1
      let d = 0, i = open
      for (; i < out.length; i++) { if (out[i] === '{') d++; else if (out[i] === '}' && --d === 0) break }
      const params = m[3].split(',').map(x => x.trim().match(/^(?:(in|out|inout|const)\s+)?(\w+)\s+(\w+)$/)).filter(Boolean)
      const term = ([, q, t, n]) => q === 'out' ? null : t === 'float' ? n : /^float[234]$/.test(t) ? `${n}.x`
        : /^(int|uint|bool)$/.test(t) ? `(float)${n}` : /^(int|uint|bool)[234]$/.test(t) ? `(float)${n}.x` : null
      const v = mode === 'zero' ? '0' : '_stub_s'
      const head = mode === 'zero' ? '' : `float _stub_s = ${params.map(term).filter(Boolean).join(' + ') || '0.0'};\n`
      // out params are set; in pass mode inout ones too (left alone, a struct the caller filled with constants would fold)
      const outs = params.filter(q => q[1] === 'out' || (q[1] === 'inout' && mode === 'pass')).map(([, , t, n]) => `${n} = (${t})${v};\n`).join('')
      const body = `{\n${head}${outs}${m[1] === 'void' ? '' : `return (${m[1]})${v};\n`}}`
      out = out.slice(0, open) + body + out.slice(i + 1)
      re.lastIndex = open + body.length
      hit.push(m[2])
    }
  }
  return { src: out, hit }
}

const jobs = []
for (const f of files) {
  const base = hlslOf(f), target = /\.vs\./.test(f) ? 'vs_5_0' : 'ps_5_0', tag = path.basename(f).replace(/\.hlsl$/, '')
  const sets = each ? [[], ...each.map(n => [n])] : [stub]
  for (const s of sets) {
    const { src, hit } = s.length ? stubbed(base, s) : { src: base, hit: [] }
    if (s.length && !hit.length) { console.log(`${tag}: no function matches ${s.join(',')}`); continue }
    const p = path.join(tmp, `${jobs.length}-${tag}${s.length ? '-stub-' + s.join('+') : ''}.hlsl`)
    fs.writeFileSync(p, src)
    jobs.push({ p, target, label: `${tag}${s.length ? ` ${mode} ` + s.join('+') + ` (${hit.length} fn)` : ''}` })
  }
}
// One process per target runs everything: first an untimed warm-up compile of the first file (the first big
// D3DCompile in a process is ~5 s slower: 29.8 s, then 24.30 / 24.36 / 24.31 s for the same file), then the jobs
// `repeat` times in order, then the first job once more (drift check: the machine is shared).
const cs = path.join(__dirname, 'fxc-time.cs').replace(/'/g, "''")
const results = new Map()
for (const t of ['ps_5_0', 'vs_5_0']) {
  const list = jobs.filter(j => j.target === t)
  if (!list.length) continue
  const order = [...(a.includes('--no-warm') ? [] : [list[0]]), ...Array.from({ length: repeat }, () => list).flat(), ...(list.length > 1 ? [list[0]] : [])]
  const cmd = `Add-Type -Path '${cs}'; [FxcTime]::Run('${dll}', '${t}', ${flags}, @(${order.map(j => `'${j.p}'`).join(', ')}))`
  const res = spawnSync('powershell', ['-NoProfile', '-NonInteractive', '-Command', cmd], { encoding: 'utf8', maxBuffer: 1 << 26 })
  if (res.status !== 0) { console.error(res.stderr || res.stdout); process.exit(1) }
  res.stdout.trim().split(/\r?\n/).forEach((line, i) => {
    const [ms, ok, bytes, p, err] = line.split('\t'), j = jobs.find(x => x.p === p)
    if (!j) return
    const warm = i === 0 && !a.includes('--no-warm')
    if (!warm) { const e = results.get(j) || { ms: [], bytes, ok, err }; e.ms.push(+ms); results.set(j, e) }
    console.log(`${(warm ? '(warm-up) ' : '') + j.label.padEnd(48)} ${String(ms).padStart(7)} ms  ${ok}  dxbc ${bytes} B${err ? '  ' + err : ''}`)
  })
}
if (repeat > 1 || jobs.length > 1) {
  console.log('\nsummary (min ms over repeats)')
  const ref = results.get(jobs[0])
  for (const j of jobs) {
    const e = results.get(j), mn = Math.min(...e.ms), d = j === jobs[0] ? '' : `${mn - Math.min(...ref.ms) > 0 ? '+' : ''}${mn - Math.min(...ref.ms)} ms, dxbc ${e.bytes - ref.bytes > 0 ? '+' : ''}${e.bytes - ref.bytes} B`
    console.log(`${j.label.padEnd(48)} ${String(mn).padStart(7)} ms  dxbc ${String(e.bytes).padStart(7)} B  ${d}`)
  }
}
fs.rmSync(tmp, { recursive: true, force: true })
