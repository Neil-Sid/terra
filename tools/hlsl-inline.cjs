// Inlining estimate for an ANGLE-translated shader (a --dump file of tools/compile-bench.cjs). fxc inlines every call
// and unrolls constant-bound loops, so a function's weight in the compile is (times it gets instantiated in the fully
// inlined main) x (its own body size). Prints the heaviest functions with their source file (notes/compile.md).
//   node tools/hlsl-inline.cjs <file.fs.hlsl> [--top 15] [--src <root>]
const fs = require('fs')
const path = require('path')

const a = process.argv.slice(2)
const file = a[0], top = +(a[a.indexOf('--top') + 1] || 15) || 15
const root = a.includes('--src') ? path.resolve(a[a.indexOf('--src') + 1]) : path.resolve(__dirname, '..')
let text = fs.readFileSync(file, 'utf8')
const m = text.match(/\/\/ COMPILER INPUT HLSL BEGIN([\s\S]*?)\/\/ COMPILER INPUT HLSL END/)
if (m) text = m[1]

// function definitions: "type name(args)" then a "{" (same or next line); body by brace matching
const DEF = /^([A-Za-z_]\w*)[ \t]+([A-Za-z_]\w*)[ \t]*\(([^;{}]*)\)[ \t]*\n?[ \t]*\{/gm
const KEYWORDS = new Set(['return', 'if', 'else', 'for', 'while', 'switch', 'struct', 'do'])
const fns = new Map()
for (const d of text.matchAll(DEF)) {
  if (KEYWORDS.has(d[1]) || KEYWORDS.has(d[2])) continue
  const open = d.index + d[0].length - 1
  let depth = 0, i = open
  for (; i < text.length; i++) { const c = text[i]; if (c === '{') depth++; else if (c === '}' && --depth === 0) break }
  fns.set(d[2], { name: d[2], body: text.slice(open, i + 1), start: open })
}

// constant trip counts of "for(int _i = {a}; (_i < N); (_i++))" loops; data-dependent bounds count once (not unrolled)
const FOR = /for\((?:int|uint) (\w+) = \{(-?\d+)\}; \(\1 (<=?) (-?\d+)\); \(\1(\+\+| \+= (\d+))\)\)/
function loopSpans(body) {
  const spans = []
  for (let i = body.indexOf('for('); i >= 0; i = body.indexOf('for(', i + 4)) {
    let depth = 0, j = i + 3
    for (; j < body.length; j++) { if (body[j] === '(') depth++; else if (body[j] === ')' && --depth === 0) break }
    const header = body.slice(i, j + 1), f = header.match(FOR)
    let trip = 1
    if (f) {
      const a0 = +f[2], n = +f[4], step = f[6] ? +f[6] : 1
      trip = Math.max(0, f[3] === '<' ? Math.ceil((n - a0) / step) : Math.floor((n - a0) / step) + 1)
    }
    const b0 = body.indexOf('{', j)
    let d = 0, k = b0
    for (; k < body.length; k++) { if (body[k] === '{') d++; else if (body[k] === '}' && --d === 0) break }
    spans.push({ from: b0, to: k, trip, header })
  }
  return spans
}

for (const f of fns.values()) {
  const spans = loopSpans(f.body)
  f.calls = new Map()        // callee -> instances per instance of f (loop trip counts included)
  f.sites = new Map()        // callee -> textual call sites
  for (const c of f.body.matchAll(/\b([A-Za-z_]\w*)\s*\(/g)) {
    const g = c[1]
    if (g === f.name || !fns.has(g)) continue
    const mult = spans.filter(s => c.index > s.from && c.index < s.to).reduce((p, s) => p * s.trip, 1)
    f.calls.set(g, (f.calls.get(g) || 0) + mult)
    f.sites.set(g, (f.sites.get(g) || 0) + 1)
  }
  f.size = f.body.length
}

// instances in the fully inlined entry point (propagate in topological order: GLSL has no recursion)
const entry = fns.get('main') ? 'main' : [...fns.keys()].pop()
const order = [], seen = new Set()
const visit = n => { if (seen.has(n)) return; seen.add(n); for (const g of fns.get(n).calls.keys()) visit(g); order.push(n) }
visit(entry)
order.reverse()
for (const f of fns.values()) { f.n = 0; f.siteN = 0 }
fns.get(entry).n = 1
for (const n of order) { const f = fns.get(n); for (const [g, k] of f.calls) { fns.get(g).n += f.n * k; fns.get(g).siteN += f.sites.get(g) } }

// source names and files: HLSL "f_name[_argtypes]" -> the GLSL function "name" defined in js/**/*.js
const srcDefs = new Map()
const walk = d => fs.readdirSync(d, { withFileTypes: true }).forEach(e => e.isDirectory() ? walk(path.join(d, e.name)) : e.name.endsWith('.js') && scan(path.join(d, e.name)))
const scan = p => {
  const s = fs.readFileSync(p, 'utf8')
  for (const d of s.matchAll(/^\s*(?:[A-Z]\w*|float|int|uint|bool|void|[iub]?vec[234]|mat[234])\s+([a-zA-Z_]\w*)\s*\([^;{]*\)\s*\{/gm)) {
    const list = srcDefs.get(d[1]) || []
    const rel = path.relative(root, p).replace(/\\/g, '/')
    if (!list.includes(rel)) list.push(rel)
    srcDefs.set(d[1], list)
  }
}
walk(path.join(root, 'js'))
const source = h => {
  if (!h.startsWith('f_')) return { name: h, file: '(ANGLE helper / entry)' }
  const s = h.slice(2)
  let best = null
  for (const k of srcDefs.keys()) if ((s === k || s.startsWith(k + '_')) && (!best || k.length > best.length)) best = k
  return best ? { name: best, file: srcDefs.get(best).join(', ') } : { name: s, file: '?' }
}

const live = [...fns.values()].filter(f => f.n > 0)
const total = live.reduce((t, f) => t + f.n * f.size, 0)
console.log(`${path.basename(file)}: ${fns.size} functions, ${live.length} reachable; inlined size ~${(total / 1e6).toFixed(2)} M chars (own body x instances, constant loops unrolled)`)
console.log('rank  share   instances  sites  body chars  function (source file)')
const ranked = live.sort((x, y) => y.n * y.size - x.n * x.size).slice(0, top)
ranked.forEach((f, i) => {
  const s = source(f.name)
  console.log(`${String(i + 1).padStart(4)}  ${(100 * f.n * f.size / total).toFixed(1).padStart(5)} %  ${String(f.n).padStart(9)}  ${String(f.siteN).padStart(5)}  ${String(f.size).padStart(10)}  ${s.name} (${s.file})  [${f.name}]`)
})
// where the instances come from: caller (its instances x calls per instance)
if (a.includes('--callers')) {
  console.log('\ninstances by caller')
  for (const f of ranked) {
    const by = live.filter(g => g.calls.has(f.name)).map(g => [source(g.name).name, g.n * g.calls.get(f.name), g.n, g.calls.get(f.name)]).sort((x, y) => y[1] - x[1])
    console.log(`${source(f.name).name}: ` + by.slice(0, 8).map(([n, k, gn, c]) => `${n} ${k} (${gn}x${c})`).join(', ') + (by.length > 8 ? ', ...' : ''))
  }
}
