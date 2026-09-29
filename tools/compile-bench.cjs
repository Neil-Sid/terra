// Shader compile-time bench (notes/compile.md). Headless Chrome on ANGLE D3D11 (same launch as tools/check.cjs), one
// http server per root; each root's page (tools/compile-bench.page.js, served at /__bench/) builds the material from
// that root's own js/ modules, compiles it with a fresh nonce comment (no program-cache hit) and times the link until
// COMPLETION_STATUS_KHR. Roots are interleaved inside every repeat (A B A B ...), so contention hits them alike.
//   node tools/compile-bench.cjs --root . --root ../base --variants full,NO_FARM --repeat 2
// Options:
//   --root <dir>        (repeatable; default: this tool's parent dir)
//   --variants a,b      full | NO_X | NO_X+NO_Y | KEY=VAL   (defines added to the material; default full)
//   --program name      terrain | preview | sampler | probe (comma list; default terrain)
//   --part p            program (both stages, default) | fs (real FS, stub VS) | vs (real VS, stub FS)
//   --repeat n          (default 1)
//   --budget s          start no compile after s seconds (default 280: fits `timeout 300`)
//   --dump dir          write the translated HLSL and the GLSL of each (root, program, variant, part) once
//   --out file.jsonl    append one JSON line per compile
//   --alongside a,b     also compile these programs (untimed) at the same time, like the app's start-up
//   --no-nonce          same source every compile (shows the in-memory program cache at work; not for timing)
const http = require('http')
const fs = require('fs')
const path = require('path')
const puppeteer = require('F:/personal projects/silent-front/node_modules/puppeteer-core')

const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png' }
const PAGE_JS = path.join(__dirname, 'compile-bench.page.js')

function args() {
  const o = { roots: [], variants: ['full'], programs: ['terrain'], part: 'program', repeat: 1, budget: 280, dump: null, out: null, nonce: true, alongside: [] }
  const a = process.argv.slice(2)
  for (let i = 0; i < a.length; i++) {
    const k = a[i], v = a[i + 1]
    if (k === '--root') { o.roots.push(path.resolve(v)); i++ }
    else if (k === '--variants') { o.variants = v.split(',').filter(Boolean); i++ }
    else if (k === '--program') { o.programs = v.split(',').filter(Boolean); i++ }
    else if (k === '--part') { o.part = v; i++ }
    else if (k === '--repeat') { o.repeat = +v; i++ }
    else if (k === '--budget') { o.budget = +v; i++ }
    else if (k === '--dump') { o.dump = path.resolve(v); i++ }
    else if (k === '--out') { o.out = path.resolve(v); i++ }
    else if (k === '--no-nonce') o.nonce = false
    else if (k === '--alongside') { o.alongside = v.split(',').filter(Boolean); i++ }
    else throw new Error('unknown option ' + k)
  }
  if (!o.roots.length) o.roots.push(path.resolve(__dirname, '..'))
  return o
}

// full -> {}, NO_FARM+NO_FOREST -> { NO_FARM: 1, NO_FOREST: 1 }, K=V -> { K: V }
const defines = v => v === 'full' ? {} : Object.fromEntries(v.split('+').map(d => d.includes('=') ? d.split('=') : [d, 1]))

function serve(root) {
  const importMap = (fs.readFileSync(path.join(root, 'index.html'), 'utf8').match(/<script type="importmap">[\s\S]*?<\/script>/) || [''])[0]
  const benchHtml = `<!doctype html><html><head><meta charset="utf-8"><link rel="icon" href="data:,">${importMap}</head>
<body><canvas width="64" height="64"></canvas><script type="module" src="/__bench/page.js"></script></body></html>`
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(req.url.split('?')[0])
    const send = (type, body) => { res.writeHead(200, { 'Content-Type': type, 'Cache-Control': 'no-store' }); res.end(body) }
    if (url === '/__bench/bench.html') return send('text/html', benchHtml)
    if (url === '/__bench/page.js') return send('text/javascript', fs.readFileSync(PAGE_JS))
    const file = path.join(root, url === '/' ? 'index.html' : url)
    if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); return res.end() }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-store' })
    fs.createReadStream(file).pipe(res)
  })
  return new Promise(r => server.listen(0, '127.0.0.1', () => r(server)))
}

const median = a => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[n >> 1] : (s[n / 2 - 1] + s[n / 2]) / 2 }

let browser = null
async function main() {
  const o = args(), t0 = Date.now()
  const labels = o.roots.map((_, i) => String.fromCharCode(65 + i))
  o.roots.forEach((r, i) => console.log(`root ${labels[i]} = ${r}`))
  if (o.alongside.length) console.log(`compiling alongside each: ${o.alongside.join(', ')}`)
  // fresh temporary profile (puppeteer's default): no shader disk cache carried over from any earlier run
  browser = await puppeteer.launch({
    executablePath: CHROME, headless: true, protocolTimeout: 600000,
    args: ['--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist', '--disable-background-timer-throttling', '--disable-renderer-backgrounding'],
  })
  const errors = [], servers = [], pages = []
  for (const [i, root] of o.roots.entries()) {
    const server = await serve(root)
    const page = await browser.newPage()
    page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errors.push(`${labels[i]} [${m.type()}] ${m.text()}`) })
    page.on('pageerror', e => errors.push(`${labels[i]} [pageerror] ${e.message}`))
    await page.goto(`http://127.0.0.1:${server.address().port}/__bench/bench.html`, { timeout: 120000 })
    await page.waitForFunction(() => window.benchReady, { timeout: 120000 })
    const warm = await page.evaluate(() => bench.warm())
    if (i === 0) { const info = await page.evaluate(() => bench.info()); console.log(`gpu: ${info.gpu} | KHR_parallel_shader_compile ${info.khr} | WEBGL_debug_shaders ${info.debugShaders}`) }
    console.log(`root ${labels[i]} ready (warm-up program ${warm} ms)`)
    servers.push(server); pages.push(page)
  }
  if (o.dump) fs.mkdirSync(o.dump, { recursive: true })
  const results = [], dumped = new Set()
  let stopped = false
  outer:
  for (let rep = 1; rep <= o.repeat; rep++) for (const program of o.programs) for (const variant of o.variants) for (const [i, page] of pages.entries()) {
    if ((Date.now() - t0) / 1000 > o.budget) { stopped = true; break outer }
    const key = `${labels[i]}-${program}-${variant}-${o.part}`
    const hlsl = !!o.dump && !dumped.has(key)
    const r = await page.evaluate(q => bench.compile(q), { program, defines: defines(variant), part: o.part, hlsl, nonce: o.nonce, alongside: o.alongside })
    if (hlsl) {
      dumped.add(key)
      for (const [ext, k] of [['vs.hlsl', 'vsText'], ['fs.hlsl', 'fsText'], ['vs.glsl', 'vsGlsl'], ['fs.glsl', 'fsGlsl']]) fs.writeFileSync(path.join(o.dump, `${key}.${ext}`), r[k])
      delete r.vsText; delete r.fsText; delete r.vsGlsl; delete r.fsGlsl
    }
    const row = { root: labels[i], dir: o.roots[i], variant, rep, ...r }
    results.push(row)
    console.log(`${labels[i]} ${program} ${o.part} ${variant.padEnd(12)} rep ${rep}  ${String(r.ms).padStart(6)} ms  vs hlsl ${r.vsHlsl}  fs hlsl ${r.fsHlsl} chars${r.ok ? '' : '  LINK FAILED\n' + r.log}`)
    if (o.out) fs.appendFileSync(o.out, JSON.stringify({ at: new Date().toISOString(), ...row }) + '\n')
  }
  if (stopped) console.log(`budget of ${o.budget} s reached: stopped early`)

  // summary: median per (root, program, variant), and its difference from that root's full
  console.log('\nroot program  variant       n  median ms  min ms  vs full  fs hlsl')
  for (const program of o.programs) for (const [i, l] of labels.entries()) {
    const of = v => results.filter(r => r.root === l && r.program === program && r.variant === v && r.ok).map(r => r.ms)
    const full = of('full').length ? median(of('full')) : null
    for (const v of o.variants) {
      const ms = of(v)
      if (!ms.length) continue
      const m = median(ms), fsH = results.find(r => r.root === l && r.program === program && r.variant === v).fsHlsl
      const d = full !== null && v !== 'full' ? `${m - full > 0 ? '+' : ''}${Math.round(m - full)}` : ''
      console.log(`${l.padEnd(4)} ${program.padEnd(8)} ${v.padEnd(12)} ${String(ms.length).padStart(2)}  ${String(Math.round(m)).padStart(9)}  ${String(Math.min(...ms)).padStart(6)}  ${d.padStart(7)}  ${fsH}`)
    }
  }
  console.log(errors.length ? '\n' + errors.slice(0, 30).join('\n') : '\nno console errors')
  console.log(`total ${Math.round((Date.now() - t0) / 1000)} s`)
  await browser.close()
  servers.forEach(s => s.close())
  process.exit(errors.some(e => e.includes('[pageerror]')) ? 1 : 0)
}
main().catch(async e => { console.error(e); if (browser) await browser.close().catch(() => {}); process.exit(1) })
