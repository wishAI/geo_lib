import {readFile} from 'node:fs/promises';
const cache=new Map();
export async function moduleURL(name){
  if(cache.has(name))return cache.get(name);
  let code=await readFile(new URL('../gui/'+name,import.meta.url),'utf8');
  code=code.replaceAll("'three'",JSON.stringify(new URL('../../../webgui/static/vendor/three/three.module.js',import.meta.url).href));
  for(const match of [...code.matchAll(/'\/api\/artifact\?path=algorithms\/avp_remote\/gui\/([^']+)'/g)])
    code=code.replaceAll(match[0],JSON.stringify(await moduleURL(match[1])));
  const url='data:text/javascript;base64,'+Buffer.from(code).toString('base64');cache.set(name,url);return url;
}
