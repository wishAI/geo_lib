const base = 'algorithms/motion_anim_generate/outputs/';
const url = path => `/api/artifact?path=${encodeURIComponent(path)}`;
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const style = `
.motion-page{max-width:1120px;margin:auto;padding:12px 0 40px}.motion-page *{box-sizing:border-box}
.motion-page header{display:flex;align-items:end;justify-content:space-between;gap:20px;margin:22px 0}.motion-page h1{font-size:clamp(28px,4vw,42px);margin:0 0 8px}.motion-page p{color:#66716b;margin:0;line-height:1.6}.motion-page .motion-back{color:#365d50;text-decoration:none;font-size:13px}
.motion-page .motion-player{background:#e7ece9;border:1px solid #d2dcd5;border-radius:18px;overflow:hidden}.motion-page video{display:block;width:100%;max-height:66vh;min-height:220px;background:#18232a;aspect-ratio:16/9}.motion-page .motion-caption{padding:15px 19px;display:flex;align-items:center;justify-content:space-between;gap:18px}.motion-page h2{margin:0 0 3px;font-size:18px}.motion-page .motion-caption p{font-size:13px}.motion-page .motion-caption a{white-space:nowrap;color:#265b48;font-size:13px}
.motion-page .motion-gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px;margin:16px 0 26px}.motion-page .motion-clip{padding:0;text-align:left;border:1px solid #d4ddd7;background:#fff;border-radius:12px;overflow:hidden;color:#273c32;cursor:pointer}.motion-page .motion-clip[aria-pressed=true]{outline:2px solid #397c64;outline-offset:2px}.motion-page .motion-clip img{width:100%;aspect-ratio:16/9;object-fit:cover;display:block;background:#e5ebe7}.motion-page .motion-clip span{display:block;padding:10px 12px;font-size:13px}.motion-page details{border-top:1px solid #d7dfd9;padding:18px 0}.motion-page summary{cursor:pointer;color:#365d50;font-weight:600}.motion-page .motion-details{padding-top:15px;display:flex;gap:10px;flex-wrap:wrap}.motion-page .motion-details p{flex-basis:100%;font-size:13px}.motion-page button.motion-action{border:1px solid #cdd8d1;border-radius:9px;background:#fafcf9;padding:9px 13px;color:#31513f;cursor:pointer}.motion-page button:focus-visible,.motion-page a:focus-visible{outline:3px solid #dc8c40;outline-offset:3px}.motion-page .motion-message{padding:30px;color:#607167}.motion-page .motion-error{color:#a34331;padding:12px 20px}
@media(max-width:600px){.motion-page header{align-items:start}.motion-page header p{max-width:28ch}.motion-page .motion-caption{align-items:start;flex-direction:column;gap:8px}.motion-page video{min-height:150px}.motion-page .motion-gallery{grid-template-columns:repeat(2,minmax(0,1fr))}}
`;

export async function mount(root, { onPreview, onEvolution }) {
  root.innerHTML = `<style>${style}</style><main class="motion-page"><a class="motion-back" href="#/">← All sandboxes</a><header><div><h1>Landau motion</h1><p>Generated animation, ready to watch.</p></div><button class="motion-action" data-refresh>Refresh clips</button></header><section class="motion-player"><video controls playsinline preload="metadata" aria-label="Landau animation"></video><div class="motion-error" hidden></div><div class="motion-caption"><div><h2 data-title>Latest animation</h2><p data-description>Loading available clips…</p></div><a data-open target="_blank" rel="noopener">Open video ↗</a></div></section><nav class="motion-gallery" aria-label="Animation clips"></nav><details><summary>Details &amp; experiment history</summary><div class="motion-details"><p>Quality checks describe smoothness, pose fidelity and visible intersections. They do not require physical balance or robot-control feasibility. Current samples are generated without a text prompt.</p><button class="motion-action" data-comparison>Source comparison</button><button class="motion-action" data-before-after hidden>Before / after</button><button class="motion-action" data-quality>Quality report</button><button class="motion-action" data-history>Experiment history</button><button class="motion-action" data-progress>Backend progress</button><p>The main player shows Landau only. Open the source comparison to inspect the original side-by-side debug video.</p></div></details></main>`;
  const video = root.querySelector('video');
  let current = base + 'proof.mp4';
  let inventory = [];
  const show = (path, title, description, updated = '') => {
    current = path;
    root.querySelector('[data-title]').textContent = title;
    root.querySelector('[data-description]').textContent = description;
    root.querySelector('[data-open]').href = url(path);
    root.querySelector('.motion-error').hidden = true;
    const preview = ['clean_preview.mp4', 'preview.mp4'].map(name => inventory.find(a => a.path === path.replace(/proof\.mp4$/, name) && a.exists)).find(Boolean);
    const displayPath = preview?.path || path;
    root.querySelector('[data-open]').href = url(displayPath);
    video.src = url(displayPath) + '&v=' + encodeURIComponent(updated);
    for (const b of root.querySelectorAll('[data-clip]')) b.setAttribute('aria-pressed', String(b.dataset.clip === path));
  };
  video.addEventListener('error', () => {const e=root.querySelector('.motion-error');e.textContent='This clip is still syncing. Refresh clips to try again.';e.hidden=false;});
  async function refresh() {
    const response = await fetch('/api/artifacts/motion_anim_generate');
    if (!response.ok) throw new Error('Could not load clips');
    inventory = (await response.json()).artifacts;
    if (!root.isConnected) return;
    root.querySelector('[data-before-after]').hidden = !inventory.some(a => a.exists && a.path === base + 'facing_foot_comparison/preview.mp4');
    const available = inventory.filter(a => a.exists && a.kind === 'video' && a.path.endsWith('/proof.mp4'));
    const latest = available.find(a => a.path === base + 'proof.mp4');
    const generated = available.filter(a => a.path.includes('/runs/') && !a.path.includes('/upstream_')).sort((a,b)=>String(b.modifiedAt).localeCompare(String(a.modifiedAt))).slice(0,5);
    const clips = [...(latest ? [latest] : []), ...generated];
    if (!clips.length) {root.querySelector('[data-description]').textContent='No video has synced yet. Check back shortly.';return;}
    root.querySelector('.motion-gallery').innerHTML = clips.map((a,i) => {
      const poster = ['clean_contact_sheet.png', 'contact_sheet.png'].map(name => a.path.replace(/proof\.mp4$/, name)).find(path => inventory.some(x => x.path === path && x.exists));
      const hasPoster = Boolean(poster);
      return `<button class="motion-clip" data-clip="${escape(a.path)}" aria-pressed="false">${hasPoster?`<img src="${url(poster)}" alt="" loading="lazy">`:''}<span>${i===0&&latest?'Latest animation':`Generated take ${i+(latest?0:1)}`}</span></button>`;
    }).join('');
    const description = 'Landau animation · original motion and timing';
    root.querySelectorAll('[data-clip]').forEach((button,i)=>button.addEventListener('click',()=>show(clips[i].path,button.textContent,description,clips[i].modifiedAt)));
    const selected = clips.find(c=>c.path===current) || clips[0];
    const button = [...root.querySelectorAll('[data-clip]')].find(b=>b.dataset.clip===selected.path);
    show(selected.path,button.textContent,description,selected.modifiedAt);
  }
  root.querySelector('[data-refresh]').addEventListener('click',()=>refresh().catch(e=>root.querySelector('[data-description]').textContent=e.message));
  root.querySelector('[data-comparison]').addEventListener('click',()=>onPreview(current,'video'));
  root.querySelector('[data-before-after]').addEventListener('click',()=>onPreview(base+'facing_foot_comparison/preview.mp4','video'));
  root.querySelector('[data-quality]').addEventListener('click',()=>onPreview(current.replace(/proof\.mp4$/, 'validation.json'),'json'));
  root.querySelector('[data-history]').addEventListener('click',onEvolution);
  root.querySelector('[data-progress]').addEventListener('click',()=>onPreview(base+'backend_progress.json','json'));
  await refresh();
}
