const paths={
 reset:'M4 9a8 8 0 1 1 1 8M4 3v6h6',
 morph:'M12 3 3 12l9 9 9-9Z M3 12h18M12 3v18',
 move:'M12 2v20M2 12h20M8 6l4-4 4 4M8 18l4 4 4-4M6 8l-4 4 4 4M18 8l4 4-4 4',
 scale:'M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5M4 4l6 6M20 20l-6-6',
 rotate:'M4 9a8 8 0 1 1 1 8M4 3v6h6M12 8v4l3 2',
 world:'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0ZM3 12h18M12 3c-5 5-5 13 0 18 5-5 5-13 0-18Z',
 local:'m12 3 9 5v8l-9 5-9-5V8ZM3 8l9 5 9-5M12 13v8',
 live:'m9 5 11 7-11 7ZM3 5v14',
 edit:'m4 16 12-12 4 4L8 20H4Zm9-9 4 4',
 link:'m9 15 6-6M8 16l-1 1a4 4 0 0 1-6-6l5-5a4 4 0 0 1 6 0M16 8l1-1a4 4 0 0 1 6 6l-5 5a4 4 0 0 1-6 0',
 eye:'M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Zm13 0a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z',
 shirt:'m8 3 4 2 4-2 6 5-4 4-2-2v11H8V10l-2 2-4-4Z',
 chevron:'m8 5 7 7-7 7',
};
export const icon=(name,title='')=>`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"${title?`><title>${title}</title>`:'>'}<path d="${paths[name]||paths.morph}"/></svg>`;
export const GARMENT_CARDS=[{id:'vest',label:'Vest',parts:['Vest']},{id:'sleeves',label:'Sleeves',parts:['Sleeve_L','Sleeve_R']},{id:'cuffs',label:'Cuffs',parts:['Cuff_L','Cuff_R']},{id:'trousers',label:'Trousers',parts:['Trousers']},{id:'boots',label:'Boots',parts:['Boot_L','Boot_R']}];
export function widget({label,attrs,min=-1,max=1,step=.01,value=0,reset,life='edit',op='morph',scope='local',axis=''}){
 const semantic=`${life==='live'?'Animation / game':'Character editing'} · ${op==='morph'?'Surface deformation':op} · ${scope} space`;
 return `<div class="char-property ${life} ${op} ${axis?'axis-'+axis:''}"><span class="char-property-kind" title="${semantic}" aria-label="${semantic}">${icon(life)}${icon(op)}</span><label><span>${label}</span><input type="range" aria-label="${label}" ${attrs} min="${min}" max="${max}" step="${step}" value="${value}"><output>${Number(value).toFixed(step===1?0:step<.01?3:2)}</output></label><span class="char-scope" title="${scope} space">${icon(scope)}</span><button class="char-icon-reset" aria-label="Reset ${label}" title="Reset ${label}" ${reset}>${icon('reset')}</button></div>`;
}
