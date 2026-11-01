"""Start the TK2 viewer and a loopback SSH tunnel; never transfer world data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--port',type=int,default=8783);a=p.parse_args()
    if not 1024<=a.port<=65535:raise ValueError('Use an unprivileged TCP port')
    repo=Path(__file__).resolve().parents[2];url=f'http://127.0.0.1:{a.port}/'
    def live():
        try:
            with urlopen(url+'town.json',timeout=2) as r: value=json.load(r)
            return value.get('schema')=='geo.minecraft-town/v1' and str(value.get('source',{}).get('world','')).endswith('/overworld')
        except Exception:return False
    if live():print(url);return
    subprocess.run([str(repo/'geo'),'storage','sync-code-tk2'],cwd=repo,check=True,timeout=150)
    remote='''import json,subprocess,time
from pathlib import Path
from urllib.request import urlopen
root=Path(ROOT);base=root.parents[1];port=PORT
if not (root/'town.json').is_file():raise RuntimeError('Extract this region on TK2 before opening the viewer')
try:
 with urlopen('http://127.0.0.1:%d/town.json'%port,timeout=2) as r: current=json.load(r)
 if current.get('source',{}).get('world')!=json.loads((root/'town.json').read_text())['source']['world']:raise RuntimeError('Viewer port is occupied by a different world')
except OSError:
 log=(base/'viewer.log').open('ab')
 subprocess.Popen([str(base/'venv/bin/python'),'-u','-m','algorithms.minecraft_building_mapper.town_server','--root',str(root),'--port',str(port)],cwd='/home/wishai/.cache/geo-lib-webgui/current',stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
'''.replace('ROOT',repr(a.root)).replace('PORT',str(a.port))
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10']
    subprocess.run(ssh+['tk2','python3','-'],input=remote,text=True,check=True,timeout=30)
    subprocess.run(ssh+['-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-f','-N','-L',f'127.0.0.1:{a.port}:127.0.0.1:{a.port}','tk2'],check=True,timeout=20)
    for _ in range(30):
        if live():print(url);return
        time.sleep(1)
    raise RuntimeError('Viewer is still starting; check the TK2 viewer.log')

if __name__=='__main__':main()
