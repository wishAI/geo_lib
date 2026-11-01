"""XR integration regression; requires a running Geo GUI and Playwright.

Uses Three.js's real XR manager, session animation loop and stereo renderer.
Only the device/session and XR framebuffer are simulated. This is not a
Vision Pro/Safari performance test. Run with AVP_GUI_URL to override LAN URL.
"""
import os
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
 page=browser.new_page(ignore_https_errors=True,viewport={'width':1000,'height':700});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 source=Path(__file__).resolve().parents[1].joinpath('gui/viewer.js').read_text().replace('renderer.setPixelRatio','window.qaRenderer=renderer;renderer.getContext().makeXRCompatible=async()=>{};renderer.setPixelRatio',1)
 source=source.replace('liveSolver=new LiveRetargeter','liveSolver=window.qaSolver=new LiveRetargeter')
 page.route('**/api/artifact?path=algorithms/avp_remote/gui/viewer.js',lambda r:r.fulfill(body=source,content_type='text/javascript'))
 page.add_init_script('''
 window.testSession=new EventTarget();Object.assign(testSession,{visibilityState:'visible',renderState:{},environmentBlendMode:'opaque',inputSources:['left','right'].map(handedness=>({handedness,hand:{get:name=>handedness+':'+name}})),requestReferenceSpace:async()=>({}),updateRenderState(value){Object.assign(this.renderState,value)},end:async()=>testSession.dispatchEvent(new Event('end')),requestAnimationFrame(cb){return requestAnimationFrame(t=>{window.xrFrames=(window.xrFrames||0)+1;cb(t,testFrame)})},cancelAnimationFrame:id=>cancelAnimationFrame(id)});
 const matrix=(x=0,y=1.6,z=0)=>[1,0,0,0,0,1,0,0,0,0,1,0,x,y,z,1];
 window.testFrame={session:testSession,getViewerPose:()=>({transform:{matrix:window.headMatrix||matrix()},views:['left','right'].map((eye,i)=>({eye,transform:{matrix:matrix(i===0?-.032:.032)},projectionMatrix:[1,0,0,0,0,1,0,0,0,0,-1.0002,-1,0,0,-.020002,0]}))}),getJointPose(joint){if(window.failJoint){window.failJoint=false;throw new DOMException('Injected temporarily unavailable joint','InvalidStateError');}return {transform:{matrix:matrix((joint.startsWith('left')?-.2:.2)+.06*Math.sin(performance.now()/130),1.2,-.5)}}}};
 window.XRWebGLLayer=class {constructor(session,gl,options){this.framebuffer=null;this.framebufferWidth=1024;this.framebufferHeight=512;this.options=options;window.qaLayer=this;}getViewport(view){return {x:view.eye==='left'?0:512,y:0,width:512,height:512}}};
 Object.defineProperty(navigator,'xr',{value:{isSessionSupported:async()=>true,requestSession:async()=>testSession}});
 ''')
 page.goto(os.environ.get('AVP_GUI_URL','https://192.168.2.31:8443/#/sandbox/avp_remote'));page.wait_for_function("document.querySelector('#avp-remote')?.dataset.modelsReady==='true'",timeout=60000)
 page.locator('[data-action=begin]').click();page.wait_for_timeout(3000)
 def state():return page.evaluate("({frames:window.xrFrames,presenting:qaRenderer.xr.isPresenting,eyes:qaRenderer.xr.getCamera().cameras.length,render:qaRenderer.info.render,root:{...document.querySelector('#avp-remote').dataset},layer:qaLayer.options})")
 before=state();assert before['presenting'] and before['eyes']==2,before
 assert before['layer']['framebufferScaleFactor']==.65 and not before['layer']['antialias']
 page.evaluate('window.failJoint=true');page.wait_for_timeout(1000)
 after=state();assert after['frames']>before['frames']+10,(before,after)
 assert 'temporarily unavailable' in after['root']['xrLastError']
 # A failure after reading tracking (inside the solver) must also keep submitting.
 page.evaluate("()=>{window.originalSolve=qaSolver.solve;qaSolver.solve=function(...args){qaSolver.solve=originalSolve;throw new Error('Injected solver failure')};}")
 page.wait_for_timeout(1000);assert state()['frames']>after['frames']+10
 assert 'Injected solver failure' in state()['root']['xrLastError']
 # Aim at the panel and use genuine XR session select events with transient rays.
 page.evaluate("""async()=>{const THREE=await import('/vendor/three/three.module.js');const eye=new THREE.Vector3(0,1.6,0),target=new THREE.Vector3(-.48*.68,1.95,-1.55);window.headMatrix=new THREE.Matrix4().lookAt(eye,target,new THREE.Vector3(0,1,0)).setPosition(eye).elements;window.pointer={targetRaySpace:{},targetRayMode:'transient-pointer'};window.pinch=()=>{for(const type of ['selectstart','select','selectend']){const e=new Event(type);e.inputSource=pointer;e.frame={getPose:()=>type==='selectstart'?{transform:{matrix:headMatrix}}:null};testSession.dispatchEvent(e);}};}""")
 page.wait_for_function("document.querySelector('#avp-remote').dataset.xrHover==='Record'")
 for take in range(2):
  page.evaluate('pinch()');page.wait_for_function("document.querySelector('#avp-remote').dataset.recording==='true'")
  page.wait_for_timeout(1200)
  page.evaluate('pinch()');page.wait_for_function("document.querySelector('#avp-remote').dataset.recording==='false' && document.querySelector('[data-save-note]').textContent.includes('已保存')")
  assert int(state()['root']['clipFrames'])>10
 page.locator('[data-action=play]').click();page.wait_for_timeout(300)
 assert state()['root']['immersiveHandMarkers']=='50'
 # A blurred/hidden session must not read hand poses or record stale samples.
 page.evaluate("testSession.visibilityState='hidden';testSession.dispatchEvent(new Event('visibilitychange'));window.failJoint=true")
 page.wait_for_timeout(300);assert state()['root']['immersiveHandMarkers']=='0'
 assert page.evaluate('window.failJoint')
 page.evaluate("window.failJoint=false;testSession.visibilityState='visible';testSession.dispatchEvent(new Event('visibilitychange'))")
 page.wait_for_timeout(300);assert state()['root']['immersiveHandMarkers']=='50'
 page.evaluate('testSession.end()');page.wait_for_timeout(300);assert not state()['presenting']
 page.locator('[data-action=xr]').click();page.wait_for_timeout(1000);assert state()['presenting']
 # Sustained rendering after reentry, without replacing the Three animation loop.
 start=state()['frames'];page.wait_for_timeout(5000);assert state()['frames']>start+50
 page.evaluate('testSession.end()');page.wait_for_timeout(300)
 report=page.evaluate("JSON.parse(localStorage.getItem('avp-xr-diagnostics'))")
 assert report['state']=='ended' and report['frames']>50,report
 assert not errors,errors
 print(json.dumps({'before_injection_frames':before['frames'],'after_injection_frames':after['frames'],'last_session':report}))
 print('PASS: stereo XR frames survive joint/solver faults; pinch records twice; playback hands, visibility, session reentry and diagnostic persistence. Simulated device, desktop Chrome.')
 browser.close()
