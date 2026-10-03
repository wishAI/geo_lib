// Mac supervisor for the dedicated persistent TK2 task. No parent-worktree sync.
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {homedir} from 'node:os';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
const {CodexAgentManager, activeTurnIdFromThread} = await import(join(homedir(), 'vscode/mac-ops-hub/server/codex-agent-manager.mjs'));
const root=dirname(fileURLToPath(import.meta.url));
const statePath=join(root,'outputs/backend.json');
const cwd='/home/wishai/.cache/geo-lib-motion-20260920/source';
const model='gpt-6-astra', effort='high', serviceTier='default';
const manager=new CodexAgentManager();
try {
  const client=await manager.client('tk2');
  const command=process.argv[2] || 'read';
  let info;
  try { info=JSON.parse(await readFile(statePath,'utf8')); } catch(e) {if(e.code!=='ENOENT')throw e;}
  if(command==='start') {
    if(info?.threadId)throw new Error('Task already exists. Read it; never create duplicates.');
    const catalog=await client.request('model/list',{limit:100,includeHidden:true});
    const selected=catalog.data.find(m=>m.id===model || m.model===model);
    if(!selected)throw new Error('Requested gpt-6-astra model unavailable; do not substitute.');
    const started=await client.request('thread/start',{cwd,model,effort,serviceTier,approvalPolicy:'never',sandbox:'workspace-write',developerInstructions:'Work only in the dedicated motion sandbox repository. Commit focused source chunks; preserve other sandboxes and do not push or change the parent developer checkout. Follow algorithms/motion_anim_generate/AGENTS.md. The Mac supervisor handles Git integration and artifact sync.'},60000);
    info={threadId:started.thread.id,cwd,model,effort,serviceTier,branch:'codex/motion-anim-generate',createdAt:new Date().toISOString()};
    await mkdir(join(root,'outputs'),{recursive:true});
    await writeFile(statePath,JSON.stringify(info,null,2)+'\n');
    await client.request('thread/name/set',{threadId:info.threadId,name:'Landau v10 · NVIDIA Kimodo motion feasibility'});
    const prompt=await readFile(join(root,'task_request.txt'),'utf8');
    const turn=await client.request('turn/start',{threadId:info.threadId,cwd,model,effort,serviceTier,approvalPolicy:'never',sandboxPolicy:{type:'workspaceWrite',writableRoots:[cwd,`${cwd}/.git`],networkAccess:true},input:[{type:'text',text:prompt}]},60000);
    console.log(JSON.stringify({...info,turnId:turn.turn.id,status:turn.turn.status}));
  } else {
    if(!info?.threadId)throw new Error('No saved backend task.');
    const result=await manager.read('tk2',info.threadId);
    const thread=result.thread;
    const active=result.activeTurnId || activeTurnIdFromThread(thread);
    if(command==='continue') {
      if(active)throw new Error('Task active; refusing duplicate turn.');
      const prompt=process.argv.slice(3).join(' ');
      if(!prompt)throw new Error('A concrete continuation message is required.');
      await client.request('thread/resume',{threadId:info.threadId,cwd,model,effort,serviceTier,approvalPolicy:'never',sandbox:'workspace-write'},60000);
      const turn=await client.request('turn/start',{threadId:info.threadId,cwd,model,effort,serviceTier,approvalPolicy:'never',sandboxPolicy:{type:'workspaceWrite',writableRoots:[cwd,`${cwd}/.git`],networkAccess:true},input:[{type:'text',text:prompt}]},60000);
      console.log(JSON.stringify({threadId:info.threadId,turnId:turn.turn.id,status:turn.turn.status}));
    } else {
      console.log(JSON.stringify({...info,status:thread.status,activeTurnId:active,turns:(thread.turns||[]).slice(-2).map(t=>({id:t.id,status:t.status,error:t.error,messages:(t.items||[]).filter(i=>i.type==='agentMessage').slice(-4).map(i=>({phase:i.phase,text:String(i.text||'').slice(-5000)}))}))},null,2));
    }
  }
} finally { manager.close(); }
