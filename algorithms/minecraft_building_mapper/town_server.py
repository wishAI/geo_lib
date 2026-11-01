"""Loopback-only town viewer. Saves and mesh cache stay on TK2."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs,urlparse

from .town_mesh import Mesher
from .town_bounds import BoundsStore, validate_bounds, KEYS


def serve(root,port=8783):
    root=Path(root).resolve(); mesher=Mesher(root); web=Path(__file__).with_name('web')
    vendor=Path(__file__).resolve().parents[2]/'webgui/static/vendor/three'
    store=BoundsStore(root,mesher.report)
    cache=root/'meshes-v4';cache.mkdir(exist_ok=True); lock=threading.Lock()
    files={'/':'town.html','/town.js':'town.js','/town.css':'town.css','/bounds-editor.js':'bounds-editor.js'}
    artifacts={'town.json','terrain.json','satellite.png','roads.png','construction.png','buildings.png','atlas.png','render_coverage.json','extracts.json'}
    class Handler(BaseHTTPRequestHandler):
        def send(self,payload,kind,status=200,compressed=False):
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(payload)))
            # Even derivative meshes remain in memory only on the viewing Mac.
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            if compressed:self.send_header('Content-Encoding','gzip')
            self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            u=urlparse(self.path)
            try:
                if u.path=='/town.json':
                    self.send(json.dumps(mesher.report).encode(),'application/json');return
                if u.path=='/bounds':
                    self.send(json.dumps(store.read()).encode(),'application/json');return
                if u.path=='/mesh':
                    q=parse_qs(u.query);cx=int(q['x'][0]);cz=int(q['z'][0]);cut=int(q.get('cut',['255'])[0]);b=mesher.report['bounds']
                    if not b['x0']//16<=cx<=(b['x1']-1)//16 or not b['z0']//16<=cz<=(b['z1']-1)//16 or not 0<=cut<=255:raise ValueError('Chunk outside selected region')
                    building=q.get('building',[None])[0]
                    if building is not None and building not in mesher.buildings:raise ValueError('Unknown building')
                    clip=None
                    if 'box' in q:
                        values=q['box'][0].split(',')
                        if len(values)!=6:raise ValueError('Expected six box coordinates')
                        clip=validate_bounds(dict(zip(KEYS,map(int,values))),mesher.report['bounds'])
                        if building is None:raise ValueError('A box requires a selected building')
                    elif building is not None:clip=store.read()['buildings'].get(building)
                    if building is not None:
                        bb=clip or mesher.buildings[building]['bounds']
                        if not bb['x0']//16<=cx<=(bb['x1']-1)//16 or not bb['z0']//16<=cz<=(bb['z1']-1)//16:raise ValueError('Chunk outside selected building')
                    version=hashlib.sha256(json.dumps(clip,sort_keys=True).encode()).hexdigest()[:16]
                    path=cache/f'{building or "town"}.{version}.{cx}.{cz}.{cut}.bin.gz'
                    with lock:
                        if not path.exists():
                            payload=gzip.compress(mesher.mesh(cx,cz,cut,building,clip),compresslevel=3);tmp=path.with_suffix('.tmp');tmp.write_bytes(payload);tmp.replace(path)
                    payload=path.read_bytes()
                    accept='gzip' in self.headers.get('Accept-Encoding','')
                    self.send(payload if accept else gzip.decompress(payload),'application/octet-stream',compressed=accept);return
                if u.path in files:path=web/files[u.path]
                elif u.path.removeprefix('/') in artifacts:path=root/u.path[1:]
                elif u.path=='/vendor/three.module.js':path=vendor/'three.module.js'
                elif u.path=='/vendor/OrbitControls.js':path=vendor/'modules/OrbitControls.js'
                else:self.send(b'{"error":"not found"}','application/json',404);return
                kind={'.html':'text/html; charset=utf-8','.js':'text/javascript','.css':'text/css','.json':'application/json','.png':'image/png'}[path.suffix]
                self.send(path.read_bytes(),kind)
            except (ValueError,KeyError,OSError) as e:self.send(json.dumps({'error':str(e)}).encode(),'application/json',400)
            except BrokenPipeError:pass
        def do_POST(self):
            try:
                if self.path!='/bounds':raise ValueError('Unknown operation')
                if self.headers.get_content_type()!='application/json':raise ValueError('JSON required')
                origin=self.headers.get('Origin')
                if origin and origin!='http://'+self.headers.get('Host',''):raise ValueError('Origin mismatch')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=8192:raise ValueError('Invalid request size')
                result=store.save(json.loads(self.rfile.read(length)))
                self.send(json.dumps(result).encode(),'application/json')
            except (ValueError,KeyError,OSError) as e:self.send(json.dumps({'error':str(e)}).encode(),'application/json',400)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Luna Town viewer: http://127.0.0.1:{port}',flush=True);server.serve_forever()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--port',type=int,default=8783);a=p.parse_args();serve(a.root,a.port)
