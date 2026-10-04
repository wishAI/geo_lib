// Pure polygon fields shared by the editor and the garment deformation path.
export const regionKey=(part,control)=>`${part}:${control}`;
export function edgeDistance(p,a,b){const dx=b[0]-a[0],dy=b[1]-a[1],d=dx*dx+dy*dy,t=d?Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/d)):0;return Math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy);}
export function polygonValid(points){
 if(!Array.isArray(points)||points.length<3||points.length>32||points.some(p=>!Array.isArray(p)||p.length!==2||p.some(v=>!Number.isFinite(v)||v<0||v>1)))return false;
 const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);let area=0;
 for(let i=0;i<points.length;i++){const a=points[i],b=points[(i+1)%points.length];if(Math.hypot(a[0]-b[0],a[1]-b[1])<1e-5)return false;area+=a[0]*b[1]-b[0]*a[1];
  for(let j=i+2;j<points.length;j++){if(i===0&&j===points.length-1)continue;const c=points[j],d=points[(j+1)%points.length];if(cross(a,b,c)*cross(a,b,d)<=0&&cross(c,d,a)*cross(c,d,b)<=0&&Math.max(Math.min(a[0],b[0]),Math.min(c[0],d[0]))<=Math.min(Math.max(a[0],b[0]),Math.max(c[0],d[0]))&&Math.max(Math.min(a[1],b[1]),Math.min(c[1],d[1]))<=Math.min(Math.max(a[1],b[1]),Math.max(c[1],d[1])))return false;}
 }return Math.abs(area)>1e-6;
}
export function regionWeight(region,u,v){
 if(!region||region.enabled===false||!region.points)return 1;const p=[u,v],points=region.points;let inside=false,distance=Infinity;
 for(let i=0,j=points.length-1;i<points.length;j=i++){
  const a=points[i],b=points[j];if((a[1]>v)!==(b[1]>v)&&u<(b[0]-a[0])*(v-a[1])/(b[1]-a[1])+a[0])inside=!inside;
  distance=Math.min(distance,edgeDistance(p,a,b));
 }
 if(inside||distance<1e-9)return 1;if(!region.feather)return 0;
 const t=Math.min(1,distance/region.feather);return 1-t*t*(3-2*t);
}
export function validateRegions(regions={},allowed=()=>true){
 if(!regions||typeof regions!=='object'||Array.isArray(regions)||Object.keys(regions).length>160)throw new Error('Invalid UV regions');
 const out={};for(const [key,r]of Object.entries(regions)){
  if(!allowed(key)||!r||(r.points!==undefined&&!polygonValid(r.points))||(r.feather!==undefined&&(!Number.isFinite(r.feather)||r.feather<0||r.feather>.25))||typeof r.enabled!=='boolean')throw new Error('Invalid UV region: '+key);
  if(r.projection!==undefined&&(!key.startsWith('surface:')||!['front','side'].includes(r.projection)))throw new Error('Invalid surface projection');
  if(r.atlasHash!==undefined&&!/^[a-f0-9]{64}$/.test(r.atlasHash))throw new Error('Invalid editing atlas identity');
  const edits=r.edits||[];if(!Array.isArray(edits)||edits.length>2048)throw new Error('Too many influence edits');
  for(const e of edits){if(!e||typeof e.sheet!=='string'||e.sheet.length>80||!Number.isFinite(e.value)||e.value<0||e.value>1||!Number.isFinite(e.opacity)||e.opacity<0||e.opacity>1)throw new Error('Invalid influence edit');
   if(e.type==='polygon'){if(!polygonValid(e.points)||!Number.isFinite(e.feather)||e.feather<0||e.feather>.25)throw new Error('Invalid influence polygon');}
   else if(e.type==='brush'){if(!Array.isArray(e.center)||e.center.length!==2||e.center.some(v=>!Number.isFinite(v)||v<0||v>1)||!Number.isFinite(e.radius)||e.radius<=0||e.radius>.5)throw new Error('Invalid influence brush');}
   else throw new Error('Unknown influence edit');
  }
  out[key]={...(r.atlasHash?{atlasHash:r.atlasHash}:{}),...(r.projection?{projection:r.projection}:{}),...(r.points?{points:r.points.map(p=>[...p]),feather:r.feather||0}:{}),...(edits.length?{edits:structuredClone(edits)}:{}),enabled:r.enabled};
 }return out;
}
// Welded garment nodes share one mask value across every UV split. The existing
// garment seam constraints run after this field and keep parent/child joins exact.
export function editCoverage(edit,samples){
 let coverage=0;
 for(const p of samples){if(p.sheet!==edit.sheet)continue;
  const w=edit.type==='polygon'?regionWeight(edit,p.u,p.v):(()=>{const t=Math.min(1,Math.hypot(p.u-edit.center[0],p.v-edit.center[1])/edit.radius);return 1-t*t*(3-2*t);})();coverage=Math.max(coverage,w);
 }return coverage*edit.opacity;
}
// The strongest coverage among split UV aliases edits their shared vertex once.
export function editedWeight(region,samples,base=1){
 if(!region?.enabled)return 1;let weight=base;for(const edit of region.edits||[]){const a=editCoverage(edit,samples);weight+=(edit.value-weight)*a;}return weight;
}
export function nodeRegionWeight(region,node){
 if(!region?.enabled)return 1;let weight=region.points?0:1;const samples=[];
 for(const [r,i]of node.refs){const uv=r.g.attributes.uv;if(uv){const u=uv.getX(i),v=uv.getY(i);if(region.points)weight=Math.max(weight,regionWeight(region,u,v));samples.push({sheet:r.part.name,u,v});}}
 return editedWeight(region,samples,weight);
}
