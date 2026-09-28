const cameraNames={"CAM-A":"东入口","CAM-B":"检修段","CAM-C":"西出口"};
let timer=null,last={detections:{},events:[]};
const $=id=>document.getElementById(id);
function boxStyle(box){if(!box)return'';return `left:${box[0]/6.4}%;top:${box[1]/3.6}%;width:${(box[2]-box[0])/6.4}%;height:${(box[3]-box[1])/3.6}%`}
function entities(d={}){let html='';(d.persons||[]).forEach(x=>html+=`<div class="entity" data-label="PERSON · ${x.helmet==='wearing'?'HELMET OK':'NO HELMET'}" style="${boxStyle(x.bbox)}"></div>`);(d.objects||[]).forEach(x=>html+=`<div class="entity object" data-label="OBJECT · ${x.track_id}" style="${boxStyle(x.bbox)}"></div>`);(d.fire||[]).forEach(x=>html+=`<div class="entity fire" data-label="FIRE CANDIDATE" style="${boxStyle(x.bbox)}"></div>`);return html}
function render(s){last=s;$('step').textContent=String(Math.min(s.step,35)).padStart(2,'0');$('phase').textContent=s.phase;$('description').textContent=s.description;$('progress').style.width=`${Math.min(100,s.step/35*100)}%`;$('frames').textContent=s.metrics.frames;$('llm').textContent=s.metrics.llm_calls;$('eventCount').textContent=s.metrics.events;$('algorithm').textContent=s.metrics.algorithm;
  const session=s.corridor.session;$('session').textContent=session?session.session_id:'未建立会话';$('sessionState').textContent=session?session.state:'idle';
  const prior=last.detections||{};$('cameras').innerHTML=Object.entries(cameraNames).map(([id,name])=>{const d=(s.detections&&s.detections[id])||prior[id]||{};const c=s.corridor.cameras[id]||{};return `<article class="camera ${d.changed?'changed':''}"><div class="camera-head"><span>${id} · ${name}</span><span>${c.healthy?'● ONLINE':'○ WAIT'}</span></div><div class="viewport">${entities(d)}</div><div class="camera-foot"><span>BASE v${d.baseline_version||c.baseline_version||0} · ${c.baseline_state||'calibrating'}</span><span class="ratio">Δ ${((d.changed_ratio||0)*100).toFixed(1)}%</span></div></article>`}).join('');
  const active=s.step<5?0:s.step<11?1:s.step<16?2:s.step<22?3:4;document.querySelectorAll('#trace li').forEach((el,i)=>{el.className=i<active?'done':i===active?'active':''});
  if(s.events.length){$('events').className='';$('events').innerHTML=s.events.slice().reverse().map(e=>`<div class="event"><code>${e.event_type}</code><div><strong>${e.event_text}</strong><span>${e.camera_id} · ${e.session_id||'无会话'} · ${e.reason_code}</span></div><b>${Math.round(e.confidence*100)}%</b></div>`).join('')}else{$('events').className='empty';$('events').textContent='当前没有正式事件。人员尚未离开整个廊段时，工具箱只保留为候选。'}
  if(s.complete)stop();}
async function call(path){const r=await fetch(path,{method:'POST'});if(!r.ok)throw new Error('请求失败');render(await r.json())}
async function step(){try{await call('/api/step')}catch(e){stop();$('description').textContent='演示服务连接中断，请重新启动。'}}
function play(){if(timer)return stop();$('toggle').textContent='Ⅱ 暂停';timer=setInterval(step,700);step()}
function stop(){clearInterval(timer);timer=null;$('toggle').textContent='▶ 继续演示'}
$('toggle').onclick=play;$('next').onclick=step;$('reset').onclick=async()=>{stop();await call('/api/reset');$('toggle').textContent='▶ 开始演示'};
fetch('/api/state').then(r=>r.json()).then(render);
