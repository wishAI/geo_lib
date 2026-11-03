"""Replay an explicitly supplied real take in an isolated browser/store.

Usage: python tests/browser_hand_recording.py recording.json[.gz]
Requires Playwright and Chrome. Never writes to the user's recording store.
"""
import gzip
import json
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from webgui import avp_recordings, server
from playwright.sync_api import sync_playwright

path = Path(sys.argv[1])
original = path.read_bytes()
clip = json.loads(gzip.decompress(original) if path.suffix == '.gz' else original)
with tempfile.TemporaryDirectory() as directory, sync_playwright() as p:
    avp_recordings.ROOT = Path(directory) / 'takes'
    saved = avp_recordings.save(clip)
    http = server.GeoGuiServer(('127.0.0.1', 0), server.Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True,
        args=['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
    page = browser.new_page(viewport={'width': 1280, 'height': 850})
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(f'http://127.0.0.1:{http.server_port}/#/sandbox/avp_remote')
    page.wait_for_function("document.querySelector('#avp-remote')?.dataset.modelsReady==='true'", timeout=60000)
    page.wait_for_function(f"document.querySelector('#avp-remote').dataset.clipFrames==='{len(clip['frames'])}'")
    for fraction in (0, .14, .31, .88, 1, .14):
        t = clip['frames'][round((len(clip['frames']) - 1) * fraction)]['t']
        page.evaluate("t=>{const e=document.querySelector('[data-seek]');e.value=t;e.dispatchEvent(new Event('input'));}", t)
        assert page.locator('#avp-remote').get_attribute('data-hand-rig-bones') == '38'
    def exported_pose(filename):
        with page.expect_download() as download:
            page.evaluate("document.querySelector('[data-action=save]').click()")
        target = Path(directory) / filename
        download.value.save_as(target)
        return json.loads(target.read_text())

    def expected_head(remap):
        return page.evaluate("""async ({clip,remap})=>{
          const {sampleClip}=await import('/api/artifact?path=algorithms/avp_remote/gui/motion-clip.js');
          const {LiveRetargeter}=await import('/api/artifact?path=algorithms/avp_remote/gui/live-retarget.js');
          const data=await (await fetch('/api/artifact?path=algorithms/avp_remote/outputs/web_scene/scene.json')).json();
          const solver=new LiveRetargeter(data),t=Number(document.querySelector('[data-seek]').value);
          return sampleClip(clip,t,remap?i=>({...clip.frames[i].pose,...solver.headPose(clip.frames[i].tracking)}):null).pose;
        }""", {'clip': clip, 'remap': remap})

    # Original old takes have no hand rotations; toggle must restore the exact
    # scalar playback instead of leaving the corrected skin in place.
    page.evaluate("()=>{const e=document.querySelector('[data-remap-playback]');e.checked=false;e.dispatchEvent(new Event('change'));}")
    original_pose, expected = exported_pose('original.json')['pose'], expected_head(False)
    for name in ('neck_x', 'head_x'):
        assert abs(original_pose[name] - expected[name]) < 1e-8, name
    if not clip['frames'][0].get('handPose'):
        assert page.locator('#avp-remote').get_attribute('data-hand-rig-bones') == '0'
    page.evaluate("()=>{const e=document.querySelector('[data-remap-playback]');e.checked=true;e.dispatchEvent(new Event('change'));}")
    with page.expect_download() as download:
        page.evaluate("document.querySelector('[data-action=save]').click()")
    snapshot = Path(directory) / 'pose.json'
    download.value.save_as(snapshot)
    assert len(json.loads(snapshot.read_text())['handPose']) == 38
    corrected_pose, expected = json.loads(snapshot.read_text())['pose'], expected_head(True)
    for name in ('neck_x', 'head_x'):
        assert abs(corrected_pose[name] - expected[name]) < 1e-8, name
    page.locator('input[type=file]').set_input_files(snapshot)
    page.wait_for_function("document.querySelector('.avp-status').textContent.includes('Loaded pose.json')")
    assert page.locator('#avp-remote').get_attribute('data-hand-rig-bones') == '38'
    # New hand data also survives the unchanged immutable server format.
    new_clip = json.loads(json.dumps(clip))
    new_clip['frames'][0]['handPose'] = json.loads(snapshot.read_text())['handPose']
    new_clip['retargetVersion'] = 4
    new_saved = avp_recordings.save(new_clip)
    assert avp_recordings.read(new_saved['id']) == new_clip
    assert avp_recordings.read(saved['id']) == clip
    assert path.read_bytes() == original
    assert not errors, errors
    browser.close()
    http.shutdown()
    http.server_close()
    thread.join()
    print('PASS: real take seeks, corrected/original head toggle, 38-bone snapshot export/import, recording round trip; original unchanged.')
