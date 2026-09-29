// Headless check: serves terra/, opens it in Chrome on the real GPU, runs steps, prints console errors.
//   node tools/check.cjs "js:terra.geo.flyTo({agl:2000,pitch:-0.2}); terra.geo.agl=2000" "wait:2" "shot:low.png" "js:terra.terrain.count"
// Steps: wait:<s>   js:<code; value printed>   shot:<file in shots/>
// Env: CHECK_SIZE=WxH (default 1280x720)
const http = require('http')
const fs = require('fs')
const path = require('path')
const puppeteer = require('F:/personal projects/silent-front/node_modules/puppeteer-core')

const ROOT = path.resolve(__dirname, '..')
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png' }

function serve() {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(req.url.split('?')[0])
    const file = path.join(ROOT, url === '/' ? 'index.html' : url)
    if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); return res.end() }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-store' })
    fs.createReadStream(file).pipe(res)
  })
  return new Promise(r => server.listen(0, '127.0.0.1', () => r(server)))
}

async function main() {
  const steps = process.argv.slice(2)
  const server = await serve()
  const browser = await puppeteer.launch({
    executablePath: CHROME, headless: true, protocolTimeout: 180000,
    args: ['--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist', '--disable-background-timer-throttling', '--disable-renderer-backgrounding'],
  })
  const errors = []
  const page = await browser.newPage()
  const [vw, vh] = (process.env.CHECK_SIZE || '1280x720').split('x').map(Number)
  await page.setViewport({ width: vw, height: vh })
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errors.push(`[${m.type()}] ${m.text()}`); else if (m.type() === 'info') console.log(`info> ${m.text()}`) })
  page.on('pageerror', e => errors.push(`[pageerror] ${e.message}`))
  await page.goto(`http://127.0.0.1:${server.address().port}/`, { timeout: 240000 })
  try { await page.waitForFunction(() => window.terra && window.terra.ready, { timeout: 120000 }) }
  catch { console.log('terra did not become ready within 120 s') }
  await new Promise(r => setTimeout(r, 1500))
  fs.mkdirSync(path.join(ROOT, 'shots'), { recursive: true })
  for (const step of steps) {
    const i = step.indexOf(':'), kind = step.slice(0, i), arg = step.slice(i + 1)
    if (kind === 'wait') await new Promise(r => setTimeout(r, parseFloat(arg) * 1000))
    else if (kind === 'js') {
      try {
        const body = arg.includes('return') || arg.includes(';') ? arg : 'return (' + arg + ')'
        const value = await page.evaluate(`(async () => { ${body} })()`)
        console.log(`js> ${JSON.stringify(value)}`)
      } catch (e) { console.log(`js error: ${e.message}`) }
    } else if (kind === 'shot') {
      await page.screenshot({ path: path.join(ROOT, 'shots', arg) })
      console.log(`shot> shots/${arg}`)
    }
  }
  console.log(errors.length ? errors.slice(0, 30).join('\n') : 'no console errors')
  await browser.close()
  server.close()
  process.exit(errors.some(e => e.startsWith('[pageerror]')) ? 1 : 0)
}
main()
