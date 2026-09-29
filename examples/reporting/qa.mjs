// Browser QA: npm install --no-save playwright; npx playwright install chromium
// node examples/reporting/qa.mjs --report output/report-demo/report.html --out output/report-qa --synthetic
// Optional: PLAYWRIGHT_MODULE=/absolute/module/path CHROME_EXECUTABLE=/path/to/chrome
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const args=process.argv.slice(2);
const arg=(key, fallback)=>args.includes(key)?args[args.indexOf(key)+1]:fallback;
const report=path.resolve(arg('--report','output/report-demo/report.html'));
const output=path.resolve(arg('--out','output/report-qa'));
await fs.mkdir(output,{recursive:true});
const browser=await chromium.launch({headless:true,...(process.env.CHROME_EXECUTABLE?{executablePath:process.env.CHROME_EXECUTABLE}:{})});
try {
  const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
  const errors=[], requests=[];
  page.on('pageerror',error=>errors.push(error.message));
  page.on('request',request=>{if(/^https?:/.test(request.url()))requests.push(request.url())});
  await page.goto(pathToFileURL(report).href);
  assert.equal(await page.locator('.stage').count(),5);
  assert.equal(await page.locator('script').count(),0);
  const visible=await page.locator('main').innerText();
  const words=visible.trim().split(/\s+/).length;
  assert(words<650,`Customer view grew to ${words} words`);
  assert.match(visible,/Real-task transfer is not established/);
  if(args.includes('--synthetic')) assert.match(visible,/SYNTHETIC EXAMPLE/);
  await page.screenshot({path:path.join(output,'desktop.png'),fullPage:true});
  if(args.includes('--preview')) {
    assert(args.includes('--synthetic'),'Only publish a synthetic preview');
    const preview=path.resolve(arg('--preview'));
    await fs.mkdir(path.dirname(preview),{recursive:true});
    await page.screenshot({path:preview,fullPage:true});
  }
  const sizes=[];
  for(const width of [1440,1024,768,390,320]) {
    await page.setViewportSize({width,height:900});
    const check=async()=>{
      const size=await page.evaluate(()=>({viewport:innerWidth,scroll:document.documentElement.scrollWidth}));
      assert(size.scroll<=width,`Overflow ${JSON.stringify(size)}`);
      return size;
    };
    sizes.push(await check());
    for(let i=0;i<5;i++) {
      const detail=page.locator('.stage details').nth(i);
      await detail.locator('summary').click();
      assert(await detail.getAttribute('open')!==null);
      await check();
      if(width===1440) await page.screenshot({path:path.join(output,`detail-${i+1}.png`),fullPage:true});
      await detail.locator('summary').click();
    }
  }
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
  await page.locator('.audit summary').focus();
  await page.keyboard.press('Enter');
  assert(await page.locator('.audit').getAttribute('open')!==null);
  const downloadWait=page.waitForEvent('download');
  await page.getByRole('link',{name:'Download records ↓'}).click();
  const download=await downloadWait;
  await download.saveAs(path.join(output,'run-records.zip'));
  assert.equal(await page.locator('img').evaluateAll(images=>images.every(i=>i.complete&&i.naturalWidth>0)),true);
  assert.deepEqual(errors,[]); assert.deepEqual(requests,[]);
  const result={passed:true,customerWords:words,viewports:sizes,errors,externalRequests:requests};
  await fs.writeFile(path.join(output,'result.json'),JSON.stringify(result,null,2));
  console.log(JSON.stringify(result));
} finally { await browser.close(); }
