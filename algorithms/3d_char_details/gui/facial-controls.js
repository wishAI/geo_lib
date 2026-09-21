// Blender/glTF relative morphs add linearly. Clamp the shared blink/squint
// amount, then apply the midpoint clearance corrective to both original lash
// and lid. Saved settings remain the user's independent control values.
export function applyBlink(morphs, settings, animatedBlink=null) {
  for(const side of ['L','R']) {
    const blink=animatedBlink??settings['eyeBlink'+side]??0;
    const amount=Math.max(0,Math.min(1,blink+.45*(settings['eyeSquint'+side]||0)));
    for(const [name,value] of [['eyeBlink'+side,amount],['eyeSquint'+side,0],['_blinkArc'+side,4*amount*(1-amount)]])
      for(const [mesh,index] of morphs.get(name)||[])mesh.morphTargetInfluences[index]=value;
  }
}

// These were the raised white supports, now replaced by the smooth eye shells.
// Keep immutable history snapshots intact; ignore their obsolete part entries
// when applying a known-compatible preset to the revised model.
export function migrateFacePreset(preset,report) {
  if(!preset||!report?.facial_repair)return preset;
  const retired=new Set(report.facial_repair.removed_objects||[]);
  return {...preset,parts:Object.fromEntries(Object.entries(preset.parts||{}).filter(([name])=>!retired.has(name)))};
}
