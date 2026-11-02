"""Isolated browser-only legacy take migration, cross-device playback and reboot test.
Requires Playwright and Chrome; runs a temporary Geo server and recording store.
No synthetic recordings are sent to the user's running service.
"""
import json,sys,tempfile,threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from webgui import server,avp_recordings
from playwright.sync_api import sync_playwright
with tempfile.TemporaryDirectory() as directory, sync_playwright() as p:
 avp_recordings.ROOT=Path(directory)/'takes'
 http=server.GeoGuiServer(('127.0.0.1',0),server.Handler);port=http.server_port
 thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
 url=f'http://127.0.0.1:{port}/#/sandbox/avp_remote'
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
 errors=[]
 def page(context):
  page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.goto(url);page.wait_for_function("document.querySelector('#avp-remote')?.dataset.modelsReady==='true'",timeout=60000);return page
 headset=b.new_context(viewport={'width':1280,'height':800});head=page(headset)
 scene=json.loads(Path('algorithms/avp_remote/outputs/web_scene/scene.json').read_text())
 scene['snapshotPose']={j['child']:scene['snapshotPose'].get(j['child'],0) for j in scene['joints'] if j['type']!='fixed'}
 scene['snapshot']={k:v for k,v in scene['snapshot'].items() if k in ['head','left_arm','right_arm','left_wrist','right_wrist']}
 clip={'format':'avp-motion-clip','version':1,'name':'Yesterday test take','createdAt':'2026-11-01T13:00:00.000Z','urdfSha256':scene['source']['urdfSha256'],'frames':[{'t':i/30,'pose':scene['snapshotPose'],'tracking':scene['snapshot'],'rawTracking':scene['snapshot'],'alignment':None} for i in range(31)]}
 head.evaluate("async clip=>{const m=await import('/api/artifact?path=algorithms/avp_remote/gui/motion-clip.js');await m.clipStorage(clip)}",clip)
 head.reload();head.wait_for_function("document.querySelector('[data-save-note]')?.textContent.includes('已同步服务器')",timeout=30000)
 assert len(avp_recordings.summaries())==1
 saved=avp_recordings.summaries()[0];assert avp_recordings.read(saved['id'])==clip
 mac=b.new_context(viewport={'width':1500,'height':950});desktop=page(mac)
 desktop.wait_for_function("document.querySelector('#avp-remote').dataset.clipFrames==='31'",timeout=30000)
 assert '服务器' in desktop.locator('[data-takes]').text_content()
 desktop.locator('[data-action=play]').click();desktop.wait_for_function("document.querySelector('#avp-remote').dataset.playing==='true'")
 # Recreate the service, retaining only its on-disk recording directory.
 http.shutdown();http.server_close();thread.join()
 http=server.GeoGuiServer(('127.0.0.1',port),server.Handler);thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
 fresh=b.new_context(viewport={'width':390,'height':844});mobile=page(fresh)
 mobile.wait_for_function("document.querySelector('#avp-remote').dataset.clipFrames==='31'",timeout=30000)
 assert mobile.evaluate('document.documentElement.scrollHeight<=innerHeight+1')
 assert avp_recordings.read(saved['id'])==clip
 # Network failure must not delete a local take or fabricate shared success.
 head.route('**/api/avp/recordings',lambda r:r.abort());head.locator('[data-action=sync-takes]').click()
 head.wait_for_function("document.querySelector('[data-save-note]').textContent.includes('服务器同步失败')")
 assert head.evaluate("async()=>{const m=await import('/api/artifact?path=algorithms/avp_remote/gui/motion-clip.js');return (await m.clipSummaries()).length}")==1
 assert not errors,errors
 # Recomputed playback never overwrites the original stored motion.
 assert avp_recordings.read(saved['id'])==clip
 print('PASS: old IndexedDB take auto-uploaded unchanged; independent Mac context loads and plays it; fresh context recovers after server restart; mobile has no vertical overflow; failed sync preserves local data.')
 import subprocess,os
 result=subprocess.run([sys.executable,'algorithms/avp_remote/tests/browser_xr_runtime.py'],env=dict(os.environ,AVP_GUI_URL=url),capture_output=True,text=True,timeout=120)
 print(result.stdout);assert result.returncode==0,result.stderr
 b.close();http.shutdown();http.server_close();thread.join()
