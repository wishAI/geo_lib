// Pure polygon fields shared by the editor and the garment deformation path.
export const regionKey=(part,control)=>`${part}:${control}`;
export function edgeDistance(p,a,b){const dx=b[0]-a[0],dy=b[1]-a[1],d=dx*dx+dy*dy,t=d?Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/d)):0;return Math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy);}
export function polygonValid(points,maxPoints=32){
 if(!Array.isArray(points)||points.length<3||points.length>maxPoints||points.some(p=>!Array.isArray(p)||p.length!==2||p.some(v=>!Number.isFinite(v)||v<0||v>1)))return false;
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
  for(const e of edits){if(!e||(e.symmetry!==undefined&&typeof e.symmetry!=='boolean')||typeof e.sheet!=='string'||e.sheet.length>80||!Number.isFinite(e.value)||e.value<0||e.value>1||!Number.isFinite(e.opacity)||e.opacity<0||e.opacity>1)throw new Error('Invalid influence edit');
   if(e.type==='polygon'){if(!polygonValid(e.points)||!Number.isFinite(e.feather)||e.feather<0||e.feather>.25)throw new Error('Invalid influence polygon');}
   else if(e.type==='brush'){if(!Array.isArray(e.center)||e.center.length!==2||e.center.some(v=>!Number.isFinite(v)||v<0||v>1)||!Number.isFinite(e.radius)||e.radius<=0||e.radius>.5)throw new Error('Invalid influence brush');}
   else throw new Error('Unknown influence edit');
  }
  if(r.volume!==undefined)validateVolume(r.volume);
  out[key]={...(r.volume?{volume:structuredClone(r.volume)}:{}),...(r.atlasHash?{atlasHash:r.atlasHash}:{}),...(r.projection?{projection:r.projection}:{}),...(r.points?{points:r.points.map(p=>[...p]),feather:r.feather||0}:{}),...(edits.length?{edits:structuredClone(edits)}:{}),enabled:r.enabled};
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
export function editedWeight(region,samples,base=1,mirror=[]){
 if(!region?.enabled)return 1;let weight=base;for(const edit of region.edits||[]){const a=Math.max(editCoverage(edit,samples),edit.symmetry?editCoverage(edit,mirror):0);weight+=(edit.value-weight)*a;}return weight;
}
export function nodeRegionWeight(region,node){
 if(!region?.enabled)return 1;if(region.volume)return volumeWeight(region.volume,node.regionPoint);let weight=region.points?0:1;const samples=[];
 for(const [r,i]of node.refs){const uv=r.g.attributes.uv;if(uv){const u=uv.getX(i),v=uv.getY(i);if(region.points)weight=Math.max(weight,regionWeight(region,u,v));samples.push({sheet:r.part.name,u,v});}}
 return editedWeight(region,samples,weight);
}

// Pair rest-space samples across an anatomical plane. UV islands may be rotated,
// offset or split, so mirroring U would select unrelated geometry.
export function mirrorCorrespondence(nodes,center=0,tolerance=.006){
 const list=[...nodes];
 function tree(items,depth=0){if(!items.length)return null;const axis=['x','y','z'][depth%3];items.sort((a,b)=>a.p[axis]-b.p[axis]);const m=items.length>>1;return {n:items[m],axis,left:tree(items.slice(0,m),depth+1),right:tree(items.slice(m+1),depth+1)};}
 const groups=new Map();for(const n of list){const key=n.group||'';if(!groups.has(key))groups.set(key,[]);groups.get(key).push(n);}const roots=new Map([...groups].map(([key,items])=>[key,tree(items)]));
 for(const n of list){const target={x:2*center-n.p.x,y:n.p.y,z:n.p.z};let best=tolerance*tolerance,match=null;
  function visit(t){if(!t)return;const p=t.n.p,d=(p.x-target.x)**2+(p.y-target.y)**2+(p.z-target.z)**2;if(d<best){best=d;match=t.n;}const diff=target[t.axis]-p[t.axis];visit(diff<0?t.left:t.right);if(diff*diff<=best)visit(diff<0?t.right:t.left);}
  visit(roots.get(n.group||''));n.mirror=match;
 }
}

// Geometric selections live in the neutral reference frame. Feather fades
// across a configurable band around the boundary, in local dimensions.
export function validateVolume(v){
 const vec=(a,min,max)=>Array.isArray(a)&&a.length===3&&a.every(n=>Number.isFinite(n)&&n>=min&&n<=max);
 if(!v||(v.version!==undefined&&v.version!==2)||!['sphere','box','polygon'].includes(v.shape)||!vec(v.position,-20,20)||!vec(v.rotation,-1000,1000)||!vec(v.size,.0001,10)||!Number.isFinite(v.feather)||v.feather<0||v.feather>.5||typeof v.symmetry!=='boolean'||!Number.isFinite(v.centerX)||Math.abs(v.centerX)>20||(v.shape==='polygon'&&!polygonValid(v.points,128)))throw new Error('Invalid selection volume');
 if(v.featherBias!==undefined&&(!Number.isFinite(v.featherBias)||v.featherBias<0||v.featherBias>1))throw new Error('Invalid feather position');
 if(v.version===2&&v.symmetry&&(Math.abs(v.position[0]-v.centerX)>1e-8||Math.abs(v.rotation[1])+Math.abs(v.rotation[2])>1e-8||(v.shape==='polygon'&&!v.points.every(([x,y])=>v.points.some(([u,w])=>Math.abs(x+u-1)<1e-7&&Math.abs(y-w)<1e-7)))))throw new Error('Symmetric area must be centered with mirrored polygon points');
 return v;
}
const volumeMatrices=new WeakMap();
export function volumeWeight(v,p){
 let cached=volumeMatrices.get(v),m=cached?.matrix;if(!cached||cached.rotation.some((n,i)=>n!==v.rotation[i])){const [x,y,z]=v.rotation,a=Math.cos(x),b=Math.sin(x),c=Math.cos(y),d=Math.sin(y),e=Math.cos(z),f=Math.sin(z);m=[c*e,a*f+b*d*e,b*f-a*d*e,-c*f,a*e-b*d*f,b*e+a*d*f,d,-b*c,a*c];volumeMatrices.set(v,{matrix:m,rotation:[...v.rotation]});}
 function sample(px){const dx=px-v.position[0],dy=p.y-v.position[1],dz=p.z-v.position[2],x=(m[0]*dx+m[1]*dy+m[2]*dz)/v.size[0],y=(m[3]*dx+m[4]*dy+m[5]*dz)/v.size[1],z=(m[6]*dx+m[7]*dy+m[8]*dz)/v.size[2];let d;
  if(v.shape==='sphere')d=Math.hypot(x,y,z)-.5;
  else if(v.shape==='box')d=Math.max(Math.abs(x),Math.abs(y),Math.abs(z))-.5;
  else {const u=x+.5,w=y+.5;let inside=false,edge=Infinity;for(let i=0,j=v.points.length-1;i<v.points.length;j=i++){const a=v.points[i],b=v.points[j];if((a[1]>w)!==(b[1]>w)&&u<(b[0]-a[0])*(w-a[1])/(b[1]-a[1])+a[0])inside=!inside;edge=Math.min(edge,edgeDistance([u,w],a,b));}d=Math.max(inside?-edge:edge,Math.abs(z)-.5);}
  if(!v.feather)return d<=1e-10?1:0;const t=Math.max(0,Math.min(1,(v.version===2?(v.featherBias??0):0)-d/v.feather));return v.version===2?t*t*t*(t*(t*6-15)+10):t*t*(3-2*t);
 }
 return Math.max(sample(p.x),v.symmetry&&v.version!==2?sample(2*v.centerX-p.x):0);
}

// Clip to the right half, then reflect its boundary into one closed polygon.
export function symmetricPolygon(points){
 if(points.every(([x,y])=>points.some(([u,v])=>Math.abs(u-(1-x))<1e-7&&Math.abs(v-y)<1e-7)))return points.map(p=>[...p]);
 const half=[];for(let i=0;i<points.length;i++){const a=points[i],b=points[(i+1)%points.length],inside=a[0]>=.5,next=b[0]>=.5;if(inside)half.push([...a]);if(inside!==next)half.push([.5,a[1]+(b[1]-a[1])*(.5-a[0])/(b[0]-a[0])]);}
 const clean=half.filter((p,i)=>!i||Math.hypot(p[0]-half[i-1][0],p[1]-half[i-1][1])>1e-7);if(clean.length<3)return [[.15,.15],[.85,.15],[.85,.85],[.15,.85]];
 // Start after the segment on the centerline so the outward chain is contiguous.
 let start=clean.findIndex((p,i)=>Math.abs(p[0]-.5)<1e-7&&Math.abs(clean[(i+clean.length-1)%clean.length][0]-.5)<1e-7);if(start<0)return [[.15,.15],[.85,.15],[.85,.85],[.15,.85]];
 const chain=clean.slice(start).concat(clean.slice(0,start)),out=[...chain,...chain.slice(1,-1).reverse().map(([x,y])=>[1-x,y])];
 return polygonValid(out,128)?out:[[.15,.15],[.85,.15],[.85,.85],[.15,.85]];
}
export function centerVolume(v){
 v.position[0]=v.centerX;v.rotation[1]=v.rotation[2]=0;if(v.shape==='polygon')v.points=symmetricPolygon(v.points);return v;
}

// Smooth continuation of a displacement field beyond its original falloff.
// An affine trend preserves width/depth directions; a continuous residual field
// preserves local variation without nearest-sample jumps. No UVs are involved.
export function displacementField(samples){
 let peak=0;for(const s of samples)peak=Math.max(peak,Math.hypot(...s.d));if(!peak)return ()=>[0,0,0];
 const seeds=samples.filter(s=>Math.hypot(...s.d)>=peak*.35),lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];for(const s of seeds)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],s.p[k]);hi[k]=Math.max(hi[k],s.p[k]);}
 const center=lo.map((n,k)=>(n+hi[k])/2),scale=Math.max(1e-5,...hi.map((n,k)=>n-lo[k])),cells=new Map();
 for(const s of seeds){const key=s.p.map((n,k)=>Math.min(5,Math.floor((n-lo[k])/Math.max(1e-8,hi[k]-lo[k])*6))).join(',');let c=cells.get(key);if(!c){c={p:[0,0,0],d:[0,0,0],n:0};cells.set(key,c);}c.n++;for(let k=0;k<3;k++){c.p[k]+=s.p[k];c.d[k]+=s.d[k];}}
 const list=[...cells.values()];for(const s of list)for(let k=0;k<3;k++){s.p[k]=(s.p[k]/s.n-center[k])/scale;s.d[k]/=s.n;}
 const a=Array.from({length:4},()=>Array(7).fill(0));for(const s of list){const x=[1,...s.p];for(let i=0;i<4;i++){for(let j=0;j<4;j++)a[i][j]+=x[i]*x[j];for(let k=0;k<3;k++)a[i][4+k]+=x[i]*s.d[k];}}for(let i=1;i<4;i++)a[i][i]+=1e-6;
 for(let i=0;i<4;i++){let pivot=i;for(let j=i+1;j<4;j++)if(Math.abs(a[j][i])>Math.abs(a[pivot][i]))pivot=j;[a[i],a[pivot]]=[a[pivot],a[i]];const div=a[i][i]||1e-20;for(let k=i;k<7;k++)a[i][k]/=div;for(let j=0;j<4;j++)if(j!==i){const f=a[j][i];for(let k=i;k<7;k++)a[j][k]-=f*a[i][k];}}
 const trend=p=>[0,1,2].map(k=>a[0][k+4]+p.reduce((n,x,i)=>n+x*a[i+1][k+4],0));for(const s of list){const t=trend(s.p);s.res=s.d.map((n,k)=>n-t[k]);}
 return point=>{const p=point.map((n,k)=>(n-center[k])/scale),d=trend(p),res=[0,0,0];let sum=0;for(const s of list){const w=1/(.01+p.reduce((n,x,k)=>n+(x-s.p[k])**2,0))**2;sum+=w;for(let k=0;k<3;k++)res[k]+=s.res[k]*w;}for(let k=0;k<3;k++)d[k]+=res[k]/sum;const length=Math.hypot(...d),limit=peak*2;if(length>limit)for(let k=0;k<3;k++)d[k]*=limit/length;return d;};
}
