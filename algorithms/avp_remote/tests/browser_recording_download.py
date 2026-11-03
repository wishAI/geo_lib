"""Reproduce the old download timeout and check the fix at 2 Mbps.

Usage: python tests/browser_recording_download.py recording.json.gz
Uses a temporary store/server and the actual supplied recording, never uploads
test data to the user's service. Requires Playwright and desktop Chrome.
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


class Handler(server.Handler):
    def _recording(self, identifier):
        if self.path.endswith('?legacy=1'):
            self._json(avp_recordings.read(identifier))
        else:
            super()._recording(identifier)


path = Path(sys.argv[1])
original = path.read_bytes()
clip = json.loads(gzip.decompress(original) if path.suffix == '.gz' else original)
with tempfile.TemporaryDirectory() as directory, sync_playwright() as p:
    avp_recordings.ROOT = Path(directory) / 'takes'
    saved = avp_recordings.save(clip)
    http = server.GeoGuiServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True, args=['--no-sandbox'])
    page = browser.new_page()
    page.route('**/download-test', lambda route: route.fulfill(content_type='text/html', body='''
      <script type="importmap">{"imports":{"three":"/vendor/three/three.module.js"}}</script>
    '''))
    page.goto(f'http://127.0.0.1:{http.server_port}/download-test')
    page.evaluate("async()=>{window.downloadClip=(await import('/api/artifact?path=algorithms/avp_remote/gui/motion-clip.js')).sharedClips}")
    client = page.context.new_cdp_session(page)
    client.send('Network.enable')
    client.send('Network.emulateNetworkConditions', dict(offline=False, latency=60, downloadThroughput=250000, uploadThroughput=125000))
    before = page.evaluate('''async id=>{
      const start=performance.now();
      try{await (await fetch('/api/avp/recordings/'+id+'?legacy=1',{signal:AbortSignal.timeout(20000)})).json();return {unexpectedSuccess:true};}
      catch(error){return {error:error.name,seconds:(performance.now()-start)/1000};}
    }''', saved['id'])
    assert before.get('error') in ('AbortError', 'TimeoutError'), before
    after = page.evaluate('''async id=>{
      const progress=[],start=performance.now();
      const take=await downloadClip('server:'+id,{onProgress:p=>progress.push(p)});
      return {frames:take.frames.length,seconds:(performance.now()-start)/1000,progressUpdates:progress.length,first:progress[0],last:progress.at(-1)};
    }''', saved['id'])
    assert after['frames'] == len(clip['frames'])
    assert after['progressUpdates'] > 2
    assert after['last']['loaded'] == after['last']['total'] > 0
    assert path.read_bytes() == original
    print(json.dumps({'before': before, 'after': after}, indent=2))
    browser.close()
    http.shutdown()
    http.server_close()
    thread.join()
