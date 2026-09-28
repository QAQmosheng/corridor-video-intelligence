const $=id=>document.getElementById(id);
const labels={person_intrusion:'人员/安全帽异常',worker_detected:'作业人员',fire_smoke:'火焰或烟雾',waterlogging:'积水',abandoned_object:'遗留物',visual_review_required:'等待视觉复核'};
function time(value){try{return new Date(value).toLocaleTimeString('zh-CN',{hour12:false})}catch{return value||'--'}}
function render(s){
  $('camera').textContent=`${s.camera_id} · ${s.corridor_id}`;$('source').textContent=s.source;
  $('health').className=`health ${s.status==='running'?'':'warning'}`;$('health').querySelector('span').textContent=s.status==='running'?'检测运行中':s.status;
  $('baseline').textContent=`BASE v${s.baseline_version} · ${s.baseline_state}`;$('diff').textContent=`差异 ${(s.changed_ratio*100).toFixed(1)}%`;
  $('frames').textContent=s.frames.toLocaleString();$('fps').textContent=`${s.fps} FPS`;$('targets').textContent=`${s.detections.persons} / ${s.detections.objects}`;$('eventCount').textContent=s.events.length;
  $('videoError').textContent=s.error||'';$('videoError').style.display=s.error?'grid':'none';
  const active=s.active_incidents||[];$('activeCount').textContent=active.length;
  $('activeAlarms').className=active.length?'':'alarm-empty';$('activeAlarms').innerHTML=active.length?active.map(x=>`<div class="alarm-card"><div><strong>${labels[x.event_type]||x.detection_label}</strong><span>${x.detection_label} · ${x.incident_id}</span></div><b>ACTIVE</b><small>首次 ${time(x.first_seen)} · 连续 ${x.hits} 帧</small></div>`).join(''):'当前无活动报警';
  const events=s.events||[];$('events').className=events.length?'':'empty';$('events').innerHTML=events.length?events.map(e=>`<div class="event"><code>${e.event_type}</code><div><strong>${e.event_text}</strong><span>${time(e.captured_at)} · ${e.camera_id} · ${e.metadata.incident_id||e.event_id}</span></div><b>${Math.round(e.confidence*100)}%</b></div>`).join(''):'检测到稳定异常后，事件将在这里出现。';
}
async function refresh(){try{const r=await fetch('/api/live/state',{cache:'no-store'});render(await r.json())}catch{$('health').querySelector('span').textContent='服务连接中断'}}
setInterval(refresh,500);setInterval(()=>{$('clock').textContent=new Date().toLocaleTimeString('zh-CN',{hour12:false})},1000);refresh();
