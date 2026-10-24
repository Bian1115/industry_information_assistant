/* 使用真实聊天页，埋点由独立Vite配置注入；不修改生产业务流程。 */
const fs = require('node:fs'); const path = require('node:path'); const crypto = require('node:crypto');
const { chromium } = require('/Users/bxy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const OUT = process.env.SSE_EVAL_OUTPUT || fs.readFileSync('/tmp/SSE浏览器评测目录.txt','utf8').trim();
const WEB = 'http://127.0.0.1:5187'; const API = 'http://127.0.0.1:18765';
const labels = { normal:'正常回放',weak:'弱网回放',duplicate:'重复帧',gap:'缺口补齐',graph_control:'图执行对照',disconnect:'真实断网',reload:'页面刷新',fresh:'首次发起研究',fresh_duplicate:'首跑重复帧',fresh_gap:'首跑缺口补齐' };
const save = (p,v) => {fs.mkdirSync(path.dirname(p),{recursive:true}); fs.writeFileSync(p,JSON.stringify(v,null,2)+'\n')};
const sha = s => crypto.createHash('sha256').update(s).digest('hex');
const delay = ms => new Promise(r=>setTimeout(r,ms));
async function api(url, body) { const res=await fetch(API+url,body?{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)}:{}); const text=await res.text(); if(!res.ok)throw Error(`${res.status} ${text.slice(0,600)}`);return JSON.parse(text) }
function installProbe(job) {
  localStorage.setItem('auth',JSON.stringify({token:job.auth_token,isLoggedIn:true,user:{id:job.user_id,username:'浏览器隔离评测',email:'sse-eval@example.invalid',is_active:false,created_at:new Date().toISOString()}}));
  const now=()=>({mono_ms:performance.now(),epoch_ms:performance.timeOrigin+performance.now()});
  const probe = window.__SSEProbe = {page_id:crypto.randomUUID(),time_origin:performance.timeOrigin,received:[],applied:[],rejected_events:[],requests:[],dom:[],latest:null,first_body:null,first_progress:null,last_seq:0,action_start:null};
  probe.request=(run_id,after_seq)=>probe.requests.push({...now(),run_id,after_seq});
  probe.receive=e=>probe.received.push({...now(),seq:e.seq,type:e.type,wire:e.measurement||null});
  probe.generation=0;probe.reset=()=>{probe.generation++;};
  probe.initialAccepted=(e,state)=>{probe.last_seq=e.seq;probe.applied.push({...now(),generation:probe.generation,seq:e.seq,type:e.type,status:state.status,report_length:null,source:'首次发起协议接受'})};
  probe.render=p=>{probe.latest={run_id:job.run_id,status:p.status,report:p.report}};
  probe.apply=(e,p)=>{probe.last_seq=e.seq;probe.applied.push({...now(),generation:probe.generation,seq:e.seq,type:e.type,status:p.status,report_length:p.streaming_report.length,body_marker_count:p.streaming_report.split(job.body_marker).length-1})};
  probe.rejected=(e,reason)=>probe.rejected_events.push({...now(),seq:e.seq,type:e.type,reason});
  probe.projection=(checkpoint,p)=>{probe.latest={run_id:checkpoint.run_id,status:p.status,report:p.streaming_report};};
  function visibleText(needle,root=document.body) {
    if(!root)return false; const w=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);let n;
    while(n=w.nextNode())if(n.textContent.includes(needle)) {const range=document.createRange();range.selectNodeContents(n);const r=range.getBoundingClientRect();const s=getComputedStyle(n.parentElement);if(r.width&&r.height&&r.bottom>0&&r.top<innerHeight&&r.right>0&&r.left<innerWidth&&s.visibility!=='hidden'&&s.display!=='none')return true;}
    return false;
  }
  let queued=false,last='';
  function observe() {
    queued=false; if(!document.body)return;
    const root=document.querySelector('[aria-label="研究结果"]');
    const report=root?.querySelector('.com-markdown');const text=report?.textContent||'';
    const markerCount=text.split(job.body_marker).length-1;
    const signature=`${probe.last_seq}|${text.length}|${markerCount}|${probe.latest?.status}`;
    if(signature===last)return;last=signature;
    const sample={...now(),seq:probe.last_seq,status:probe.latest?.status||null,report_text_length:text.length,body_marker_count:markerCount,body_visible:visibleText(job.body_marker,report),progress_visible:visibleText(job.progress_marker),assistant_cards:document.querySelectorAll('[class*="chat-message-item--assistant"]').length};
    probe.dom.push(sample);
    if(sample.body_visible&&!probe.first_body){probe.first_body=sample;requestAnimationFrame(()=>{probe.first_body.raf_epoch_ms=performance.timeOrigin+performance.now()})}
    if(sample.progress_visible&&!probe.first_progress){probe.first_progress=sample;requestAnimationFrame(()=>{probe.first_progress.raf_epoch_ms=performance.timeOrigin+performance.now()})}
  }
  new MutationObserver(()=>{if(!queued){queued=true;queueMicrotask(observe)}}).observe(document,{childList:true,subtree:true,characterData:true});
}
async function probe(page) {return page.evaluate(()=>window.__SSEProbe)}
async function clockCalibration(page) {
  const samples=[];for(let i=0;i<7;i++)samples.push(await page.evaluate(async()=>{const start=performance.timeOrigin+performance.now();const {epoch_ms}=await(await fetch('/evaluation/clock')).json();const end=performance.timeOrigin+performance.now();return {start,end,server:epoch_ms,rtt_ms:end-start,offset_ms:epoch_ms-(start+end)/2}}));
  const best=[...samples].sort((a,b)=>a.rtt_ms-b.rtt_ms)[0];return {samples,offset_ms:best.offset_ms,error_bound_ms:best.rtt_ms/2};
}
function quantile(values,p){if(!values.length)return null;const a=[...values].sort((x,y)=>x-y);const i=(a.length-1)*p;return a[Math.floor(i)]+(a[Math.ceil(i)]-a[Math.floor(i)])*(i%1)}
function summarize(job,pages,result,clock,fault,error,network) {
  const received=pages.flatMap(p=>p.received.map(e=>({...e,page_id:p.page_id})));
  const applied=pages.flatMap(p=>p.applied.map(e=>({...e,page_id:p.page_id})));
  const expected=result?.ledger.map(e=>e.seq)||[]; const seen=new Set(applied.map(e=>e.seq));
  const duplicates=pages.flatMap(p=>{const count={};for(const e of p.applied){const key=`${e.generation}:${e.seq}`;count[key]=(count[key]||0)+1;}return Object.entries(count).filter(([_,n])=>n>1).map(([key,n])=>({page_id:p.page_id,projection_generation:Number(key.split(':')[0]),seq:Number(key.split(':')[1]),count:n}))});
  const latency=received.filter(e=>e.wire).map(e=>e.epoch_ms+clock.offset_ms-e.wire.sent_epoch_ms);
  const receiveToApply=applied.map(a=>{const r=received.find(r=>r.seq===a.seq&&r.page_id===a.page_id&&r.epoch_ms<=a.epoch_ms);return r?a.epoch_ms-r.epoch_ms:null}).filter(x=>x!==null);
  const firstRequest=pages.flatMap(p=>p.requests)[0];const start=pages[0]?.action_start;
  const firstBody=pages.find(p=>p.first_body)?.first_body;const firstProgress=pages.find(p=>p.first_progress)?.first_progress;
  const report=pages.at(-1)?.latest?.report||'';const same=report===result?.expected_report;
  const missing=expected.filter(x=>!seen.has(x));const complete=result?.delivery?.complete_events===1&&result?.delivery?.assistant_messages===1&&result?.delivery?.status==='completed';
  const dom=pages.flatMap(p=>p.dom);const lastDom=dom.at(-1);
  const streamingMax=Math.max(0,...dom.filter(x=>x.status!=='completed').map(x=>x.body_marker_count));
  const deliverySuccess=!error&&missing.length===0&&duplicates.length===0&&same&&complete&&lastDom?.assistant_cards===1;
  return {label:path.basename(job.folder),source_id:job.source_id,scenario:job.scenario,mode:job.mode,run_id:job.run_id,error:error||null,
    success:deliverySuccess&&streamingMax<=1,delivery_success:deliverySuccess,transient_section_duplicate:streamingMax>1,
    event_ledger_count:expected.length,received_sse_frames:received.length,unique_applied_sequences:seen.size,missing_sequences:missing,duplicate_applied_within_page:duplicates,
    rejected_duplicates:pages.reduce((a,p)=>a+p.rejected_events.filter(e=>e.reason==='duplicate').length,0),gaps_detected:pages.reduce((a,p)=>a+p.rejected_events.filter(e=>e.reason==='gap').length,0),
    network_latency_ms:{count:latency.length,p50:quantile(latency,.5),p95:quantile(latency,.95),min:Math.min(...latency),max:Math.max(...latency)},
    receive_to_projection_ms:{count:receiveToApply.length,p50:quantile(receiveToApply,.5),p95:quantile(receiveToApply,.95)},
    first_sse_from_request_ms:firstRequest&&received.length?received[0].epoch_ms-firstRequest.epoch_ms:null,
    first_progress_dom_from_click_ms:firstProgress&&start?firstProgress.epoch_ms-start:null,
    first_body_dom_from_click_ms:firstBody&&start?firstBody.epoch_ms-start:null,
    first_body_raf_from_click_ms:firstBody?.raf_epoch_ms&&start?firstBody.raf_epoch_ms-start:null,
    first_body_visible_before_complete:!!firstBody&&firstBody.status!=='completed',
    first_section_receive_to_body_ms:firstBody&&received.find(e=>e.type==='section_content')?firstBody.epoch_ms-received.find(e=>e.type==='section_content').epoch_ms:null,
    total_from_click_ms:start&&lastDom?lastDom.epoch_ms-start:null,
    fault,catchup_to_first_new_receive_ms:fault?.restored_epoch_ms?received.find(e=>e.epoch_ms>=fault.restored_epoch_ms)?.epoch_ms-fault.restored_epoch_ms:null,
    catchup_to_completed_dom_ms:fault?.restored_epoch_ms&&lastDom?lastDom.epoch_ms-fault.restored_epoch_ms:null,
    final_projection_matches: same,final_projection_sha256:sha(report),expected_sha256:sha(result?.expected_report||''),
    final_dom:lastDom,streaming_dom_max_body_marker_count:streamingMax,
    delivery:result?.delivery,external_http_attempts:result?.external_http_attempts,network_failures:network.filter(e=>e.kind==='requestfailed'),page_count:pages.length};
}
async function runCase(browser,sid,scenario,prefix='') {
  const label=`${prefix}题目${sid}-${labels[scenario]}`;
  const job=await api('/evaluation/create',{source_id:sid,scenario,label});
  const folder=job.folder; const context=await browser.newContext({viewport:{width:1800,height:1100},locale:'zh-CN',timezoneId:'Asia/Singapore',deviceScaleFactor:1});
  const network=[],errors=[],pages=[];let page,clock={offset_ms:0,error_bound_ms:null},fault=null,error=null,result=null;
  await context.route('**/*',route=>{const url=new URL(route.request().url());if(['127.0.0.1','localhost'].includes(url.hostname))return route.continue();network.push({kind:'external_blocked',url:url.href,epoch_ms:Date.now()});return route.abort()});
  await context.addInitScript(installProbe,job);
  await context.tracing.start({screenshots:true,snapshots:true,sources:true});
  try {
    page=await context.newPage();page.on('pageerror',e=>errors.push(String(e)));
    page.on('request',r=>{if(r.url().includes('/research/'))network.push({kind:'request',url:r.url(),epoch_ms:Date.now()})});
    page.on('requestfailed',r=>network.push({kind:'requestfailed',url:r.url(),reason:r.failure()?.errorText,epoch_ms:Date.now()}));
    page.on('response',r=>{if(r.url().includes('/research/'))network.push({kind:'response',url:r.url(),status:r.status(),epoch_ms:Date.now()})});
    await page.goto(WEB+'/chat/'+job.session_id,{waitUntil:'domcontentloaded',timeout:60000});
    const fresh=scenario.startsWith('fresh');
    const button=page.getByRole('button',{name:'继续本次研究',exact:true});
    if(fresh){await page.getByPlaceholder('按 Enter 发送，Shift + Enter 换行').fill(job.query)}
    else {await button.waitFor({timeout:45000});await button.waitFor({state:'visible'});await page.waitForFunction(()=>!Array.from(document.querySelectorAll('button')).find(b=>b.textContent==='继续本次研究')?.disabled);}
    clock=await clockCalibration(page);save(path.join(folder,'时钟校准.json'),clock);
    await page.screenshot({path:path.join(folder,'请求前页面.png')});
    if(scenario==='weak') {const cdp=await context.newCDPSession(page);await cdp.send('Network.enable');await cdp.send('Network.emulateNetworkConditions',{offline:false,latency:150,downloadThroughput:128*1024,uploadThroughput:64*1024});}
    await page.evaluate(()=>window.__SSEProbe.action_start=performance.timeOrigin+performance.now());
    if(fresh)await page.getByPlaceholder('按 Enter 发送，Shift + Enter 换行').press('Enter');else await button.click();
    if(['disconnect','reload'].includes(scenario)) {
      await page.waitForFunction(()=>!!window.__SSEProbe.first_body,undefined,{timeout:25000});
      await page.screenshot({path:path.join(folder,'中断前正文.png')});
      fault={kind:scenario,trigger:'首章已进入可见DOM，write节点尚未返回',trigger_epoch_ms:Date.now()};
      if(scenario==='disconnect') {await context.setOffline(true);fault.offline_epoch_ms=Date.now();fault.socket_cut=await(await fetch(WEB+'/evaluation/cut-stream/'+job.run_id,{method:'POST'})).json();if(!fault.socket_cut.closed_connections)throw Error('未找到可中断的真实SSE连接');await delay(3000);await context.setOffline(false);fault.restored_epoch_ms=Date.now();}
      else {pages.push(await probe(page));await page.reload({waitUntil:'domcontentloaded'});fault.restored_epoch_ms=Date.now();await page.getByRole('button',{name:'继续本次研究',exact:true}).waitFor({timeout:20000});await page.getByRole('button',{name:'继续本次研究',exact:true}).click();fault.manual_resume_click_epoch_ms=Date.now();}
    }
    await page.waitForFunction(()=>window.__SSEProbe.latest?.status==='completed',undefined,{timeout:60000});
    await page.waitForFunction(()=>document.querySelector('[aria-label="研究结果"]')?.textContent.includes('研究完成'),undefined,{timeout:10000});
    await delay(100);pages.push(await probe(page));
    await page.screenshot({path:path.join(folder,'完成页面.png')});
    save(path.join(folder,'最终页面DOM.json'),await page.evaluate(()=>({report_html:document.querySelector('[aria-label="研究结果"] .com-markdown')?.innerHTML,report_text:document.querySelector('[aria-label="研究结果"] .com-markdown')?.textContent,assistant_cards:document.querySelectorAll('[class*="chat-message-item--assistant"]').length,body_text:document.body.innerText})));
  } catch(e) {error=String(e);if(page){try{pages.push(await probe(page));await page.screenshot({path:path.join(folder,'失败页面.png')});save(path.join(folder,'失败页面内容.json'),{body:await page.locator('body').innerText()})}catch(_){}}}
  finally {
    if(scenario.startsWith('fresh'))job.run_id=pages[0]?.requests[0]?.run_id||job.run_id;
    try{result=await api('/evaluation/result/'+job.run_id)}catch(e){error=(error||'')+';服务端读取:'+e}
    save(path.join(folder,'浏览器测量日志.json'),{pages,clock,fault,network,errors});
    const metrics=summarize(job,pages,result,clock,fault,error,network);save(path.join(folder,'测量结果.json'),metrics);
    await context.tracing.stop({path:path.join(folder,'浏览器轨迹.zip')});await context.close();
    console.log(JSON.stringify({label,success:metrics.success,error,events:metrics.event_ledger_count,applied:metrics.unique_applied_sequences,first_body_ms:metrics.first_body_dom_from_click_ms}));return metrics;
  }
}
async function main() {
  for(let i=0;i<50;i++){try{if((await api('/evaluation/health')).ready)break}catch(_){}if(i===49)throw Error('隔离服务未就绪');await delay(200)}
  const browser=await chromium.launch({channel:'chrome',headless:true});
  save(path.join(OUT,'浏览器环境.json'),{browser_version:browser.version(),headless:true,viewport:{width:1800,height:1100},node:process.version,playwright:require('/Users/bxy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/package.json').version,weak_network:{latency_ms:150,download_bytes_per_sec:131072,upload_bytes_per_sec:65536}});
  const specs=process.argv.slice(2);const cases=specs.length?specs.map(s=>{const [id,scene]=s.split(':');return [+id,scene]}):[8,25,26,27,28].flatMap(id=>Object.keys(labels).filter(s=>!s.startsWith('fresh')).map(scene=>[id,scene]));
  const metrics=[];try{for(const [id,scene]of cases){metrics.push(await runCase(browser,id,scene,process.env.SSE_EVAL_PREFIX||''));save(path.join(OUT,process.env.SSE_EVAL_SUMMARY||(process.env.SSE_EVAL_PREFIX?'准备验证汇总.json':'评测汇总.json')),metrics)}}finally{await browser.close()}
}
main().catch(e=>{console.error(e);process.exitCode=1});
