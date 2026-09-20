"""Re-render retained targets with identical cameras; preserve original debug evidence."""
import argparse
import json
import shutil
import subprocess
from PIL import Image
from directions import diagnose
from render import render
from state import OUT, sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--before',required=True);p.add_argument('--after',required=True)
    args=p.parse_args();runs=[OUT/'runs'/n for n in (args.before,args.after)]
    if sha256(runs[0]/'source.npz')!=sha256(runs[1]/'source.npz'):raise ValueError('Comparison requires identical source bytes')
    reports=[]
    for title,run in zip(['BEFORE: original retarget','AFTER: facing and foot orientation'],runs):
        for name in ('proof.mp4','contact_sheet.png','video.json'):
            saved=run/('original_'+name)
            if not saved.exists():shutil.copyfile(run/name,saved)
        render(run,title+' | '+run.name)
        reports.append(diagnose(run))
    out=OUT/'facing_foot_comparison';out.mkdir(exist_ok=True)
    commands=[]
    for src,dst in [('proof.mp4','proof.mp4'),('clean_preview.mp4','preview.mp4')]:
        command=['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(runs[0]/src),'-i',str(runs[1]/src),
                 '-filter_complex','hstack=inputs=2','-c:v','libx264','-crf','24','-pix_fmt','yuv420p',str(out/dst)]
        subprocess.run(command,check=True);commands.append(command)
    images=[Image.open(r/'clean_contact_sheet.png') for r in runs]
    sheet=Image.new('RGB',(images[0].width,images[0].height*2))
    for i,im in enumerate(images):sheet.paste(im,(0,i*im.height))
    sheet.save(out/'contact_sheet.png')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','quiet','-print_format','json','-show_streams','-show_format',str(out/'proof.mp4')]))
    write_json(out/'comparison.json',{'before':args.before,'after':args.after,'layout':'before left, after right; sheet before top, after bottom',
        'identical_source_sha256':reports[0]['source_sha256'],'before_direction_metrics':reports[0]['summary'],
        'after_direction_metrics':reports[1]['summary'],'camera':'Identical fixed world cameras in both renders; no video rotation',
        'caveat':'Same motion source; combined frame, anchor, orientation, shin-DOF and temporal-contact changes, not an isolated single-variable ablation.',
        'commands':commands,'ffprobe':probe,'video_sha256':sha256(out/'proof.mp4')})


if __name__=='__main__':main()
