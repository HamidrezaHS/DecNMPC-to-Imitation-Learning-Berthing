"""Render a scientific MP4 replay of archived Dec-NMPC and IL berthing trajectories.

The images are generated from recorded ship/tug state histories, rather than
new simulator runs. Smooth image motion uses linear interpolation between
recorded telemetry samples (including heading unwrapping). Frame-time is
compressed relative to simulation time. No physical experiment is depicted.

Usage from repository root (ffmpeg required in PATH):
  python -m reproducibility.make_supplementary_video --out supplementary/Video_S1_Berthing_Comparison.mp4
"""
from __future__ import annotations
import argparse, math, subprocess, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from visualization.marine_renderer import get_hull
from plot_marl import _load_tug_outline

ROOT=Path(__file__).resolve().parents[1]
NAVY='#102a44'; BLUE='#2778a9'; TEAL='#1c9888'; GRAY='#a0b7c7'; ORANGE='#de9850'; VIOLET='#9062ac'; WHITE='#ffffff'; MUTED='#d5e2eb'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTS={}
def font(s,bold=False):
    key=(s,bold)
    if key not in FONTS:
        try:
            path=FONT.replace('DejaVuSans.ttf','DejaVuSans-Bold.ttf') if bold else FONT
            FONTS[key]=ImageFont.truetype(path,s)
        except OSError:
            FONTS[key]=ImageFont.load_default()
    return FONTS[key]

SHIP=np.asarray(get_hull(),float)[::6]
TUG=np.asarray(_load_tug_outline(),float)[::7]
COLS=[BLUE,TEAL,ORANGE,VIOLET]

def line(draw,xy,fill,width=1):
    draw.line([(round(x),round(y)) for x,y in xy],fill=fill,width=width,joint='curve')

def rounded(draw,box,fill,outline=None,r=12,width=1):
    draw.rounded_rectangle(box,radius=r,fill=fill,outline=outline,width=width)

def txt(draw,x,y,s,sz=16,fill=WHITE,bold=False):
    draw.text((x,y),s,font=font(sz,bold),fill=fill)

def rot(poly,x,y,psi_deg):
    a=np.deg2rad(psi_deg); c,s=np.cos(a),np.sin(a)
    return np.column_stack((poly[:,0]*c-poly[:,1]*s+x,poly[:,0]*s+poly[:,1]*c+y))

def sample(df,t):
    ts=df['time_s'].to_numpy(float)
    t=max(ts[0],min(t,ts[-1]))
    r={}
    for c in ['x_m','y_m','psi_deg','position_error_m']+[f'tug{i}_{k}' for i in range(1,5) for k in ['x_m','y_m','psi_deg','contact_x_m','contact_y_m','rope_anchor_x_m','rope_anchor_y_m']]:
        if c not in df.columns:continue
        v=df[c].to_numpy(float)
        if c.endswith('psi_deg'):
            v=np.rad2deg(np.unwrap(np.deg2rad(v)))
        r[c]=float(np.interp(t,ts,v))
    r['time_s']=t
    return r

def back(draw):
    draw.rectangle((0,0,1280,720),fill='#edf4f8')
    draw.rectangle((0,0,1280,103),fill=NAVY)
    txt(draw,38,20,'AUTONOMOUS TUG-ASSISTED TANKER BERTHING',25,WHITE,True)
    txt(draw,40,61,'Decentralized predictive teacher vs. imitation-learning controller',16,MUTED)


def panel(draw,box,title,colour,df,t,geom='ship'):
    x0,y0,x1,y1=box
    rounded(draw,(x0,y0,x1,y1),'#ffffff','#d0dce6',r=11,width=2)
    draw.rectangle((x0+2,y0+2,x1-2,y0+38),fill=colour)
    txt(draw,x0+15,y0+8,title,18,WHITE,True)
    # fixed earth-frame footprint, same for both methods. Equal scales so
    # apparent size and clearance comparisons are physically interpretable.
    r=sample(df,t)
    px0,px1=x0+34,x1-20;py0,py1=y0+56,y1-21
    # Map earth x[-240,230], y[-170,155] to subpanel pixels.
    scale=min((px1-px0)/470.,(py1-py0)/325.)
    origin_x=(px0+px1)/2-(-5)*scale
    origin_y=(py0+py1)/2+(-7.5)*scale
    def proj(X,Y):return (origin_x+X*scale,origin_y-Y*scale)
    # calm chart background + coordinate grid
    for x in range(-200,201,100):
        X=proj(x,0)[0]; line(draw,[(X,py0),(X,py1)],'#e0e8ee',1)
    for y in range(-150,151,75):
        Y=proj(0,y)[1]; line(draw,[(px0,Y),(px1,Y)],'#e0e8ee',1)
    Yq=proj(0,-134)[1]
    draw.rectangle((px0,Yq,px1,py1),fill='#e1e8ed')
    line(draw,[(px0,Yq),(px1,Yq)],'#a07343',3)
    # reconstructed target silhouette, dot and dashed contour
    target=rot(SHIP,50,-100,0)
    targetxy=[proj(xx,yy) for xx,yy in target]
    line(draw,targetxy+[targetxy[0]],'#82aaa9',2)
    Xtar,Ytar=proj(50,-100)
    draw.ellipse((Xtar-4,Ytar-4,Xtar+4,Ytar+4),fill=TEAL)
    # entire logged ship trajectory to current replay time
    use=df[df['time_s']<=t]
    pathxy=[proj(xx,yy) for xx,yy in zip(use['x_m'],use['y_m'])]
    if pathxy:line(draw,pathxy,colour,2)
    # ship hull based on logged state + actual hull geometry
    shp=rot(SHIP,r['x_m'],r['y_m'],r['psi_deg'])
    draw.polygon([proj(xx,yy) for xx,yy in shp],fill='#bbccd8')
    line(draw,[proj(xx,yy) for xx,yy in shp]+[proj(*shp[0])],NAVY,2)
    for i in range(1,5):
        tug=rot(TUG,r[f'tug{i}_x_m'],r[f'tug{i}_y_m'],r[f'tug{i}_psi_deg'])
        if i>=3:
            c=(r[f'tug{i}_contact_x_m'],r[f'tug{i}_contact_y_m'])
            a=(r[f'tug{i}_rope_anchor_x_m'],r[f'tug{i}_rope_anchor_y_m'])
            line(draw,[proj(*c),proj(*a)],COLS[i-1],2)
        draw.polygon([proj(xx,yy) for xx,yy in tug],fill=COLS[i-1])
        line(draw,[proj(xx,yy) for xx,yy in tug]+[proj(*tug[0])],NAVY,1)
    # annotations anchored outside physical plot
    txt(draw,x0+20,y1-25,'Ship and four tugboats • reconstructed logged poses',11,'#5c7387')
    return r


def errorplot(draw,box,teacher,student,t):
    x0,y0,x1,y1=box
    rounded(draw,box,'#ffffff','#d0dce6',10,2)
    txt(draw,x0+14,y0+9,'POSITION ERROR  |  dashed threshold = 2.5 m',14,NAVY,True)
    xmin,xmax=x0+50,x1-25;ymin,ymax=y0+45,y1-20
    def xy(tt,ee):return (xmin+(tt/6500)*(xmax-xmin),ymax-(ee/120.)*(ymax-ymin))
    for e in [0,30,60,90,120]:
        Y=xy(0,e)[1]
        line(draw,[(xmin,Y),(xmax,Y)],'#e8edf2',1)
    a=teacher.loc[teacher['time_s']<=t,['time_s','position_error_m']].to_numpy()
    b=student.loc[student['time_s']<=t,['time_s','position_error_m']].to_numpy()
    if len(a)>1:line(draw,[xy(*p) for p in a],BLUE,3)
    if len(b)>1:line(draw,[xy(*p) for p in b],TEAL,3)
    yy=xy(0,2.5)[1]
    for X in range(xmin,xmax,12):line(draw,[(X,yy),(min(X+6,xmax),yy)],'#c07860',2)
    txt(draw,xmin,ymax+5,'0 min',11,'#5a7084')
    txt(draw,xmax-85,ymax+5,'108 min',11,'#5a7084')
    txt(draw,xmax-375,y0+9,'Dec-NMPC',12,BLUE,True)
    txt(draw,xmax-225,y0+9,'Proposed IL',12,TEAL,True)


def intro(draw,p):
    back(draw)
    txt(draw,40,131,'Method summary',25,NAVY,True)
    cards=[('1','Shared measured ship state\n+ own tug measurements','#e0ebf3'),
           ('2','Dec-NMPC demonstrations\n+ dataset aggregation','#dfedeb'),
           ('3','Independent neural specialists\n+ environment routing','#fbf0dd'),
           ('4','Local progress/approach filter\n(no NMPC at deployment)','#e8e1f0')]
    for i,(n,lab,col) in enumerate(cards):
        x=39+i*309
        rounded(draw,(x,230,x+285,400),col,'#b7cad7',13,2)
        txt(draw,x+20,248,n,35,NAVY,True)
        for j,ll in enumerate(lab.split('\n')):txt(draw,x+17,325+j*24,ll,14,NAVY)
        if i<3:txt(draw,x+288,297,'›',34,NAVY,True)
    txt(draw,40,464,'Physical plant: KVLCC2 tanker  +  2 pushing tugs  +  2 towing tugs',21,NAVY,True)
    txt(draw,40,504,'Each local actor accesses the common measured ship state but no peer-tug commands.',18,'#526c82')
    txt(draw,40,622,'This is a compressed numerical simulation replay, not a sea-trial recording.',16,'#9d5e49')


def outro(draw,p):
    back(draw)
    txt(draw,40,130,'Quantitative results from recorded simulations',25,NAVY,True)
    rows=[('90° / 90°  | first acceptance','37.63 min','37.70 min'),
          ('270° / 270°  | first acceptance','96.07 min','57.22 min'),
          ('Four-tug CPU decision median','110.08 ms','2.27 ms')]
    txt(draw,50,208,'Metric',16,NAVY,True)
    txt(draw,730,208,'Dec-NMPC',17,BLUE,True)
    txt(draw,995,208,'Proposed IL',17,TEAL,True)
    for i,(name,tec,stu) in enumerate(rows):
        y=254+i*77
        rounded(draw,(40,y-6,1239,y+57),'#ffffff','#cfdae4',8,1)
        txt(draw,54,y+12,name,17,NAVY,True)
        txt(draw,746,y+12,tec,17,BLUE,True)
        txt(draw,1015,y+12,stu,17,TEAL,True)
    txt(draw,45,539,'Teacher-command imitation: 0.0367 thrust-fraction MAE on 1,056 archived test decisions',17,NAVY,True)
    txt(draw,45,588,'Only two matched development cases; supplementary IL tests: 4/6 accepted.',17,'#536a80')
    txt(draw,45,624,'No field validation, universal robustness claim, or formal safety guarantee.',15,'#9d5e49')


def frame(t,dfs):
    image=Image.new('RGB',(1280,720),'#edf4f8');draw=ImageDraw.Draw(image)
    # segments: intro 6s, 90-degree 20s, 270-degree 26s, outro 6s
    if t<6:
        intro(draw,t/6);return image
    if t>=52:
        outro(draw,(t-52)/6);return image
    case=90 if t<26 else 270
    t0=6 if case==90 else 26
    seglen=20 if case==90 else 26
    dt=(t-t0)/seglen*6500
    tc,il=dfs[case]
    back(draw)
    txt(draw,39,114,f'Case: wind {case}° toward  /  current {case}° toward',20,NAVY,True)
    txt(draw,810,115,f'Simulated time: {dt/60:.1f} min',17,'#586f85',True)
    tre=panel(draw,(28,151,628,486),'REFERENCE: Dec-NMPC',BLUE,tc,dt)
    ire=panel(draw,(652,151,1252,486),'LEARNED: Proposed IL',TEAL,il,dt)
    errorplot(draw,(30,502,1251,687),tc,il,dt)
    for x,r,df,cc in [(45,tre,tc,BLUE),(668,ire,il,TEAL)]:
        status='Accepted' if dt>=float(df['time_s'].iloc[-1]) else 'In progress'
        # keep label in header section outside plots
        txt(draw,x+390,157,status,13,'#ffffff',True)
    txt(draw,835,695,'Interpolation between logged states; time accelerated',10,'#5b7185')
    return image


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=ROOT/'supplementary/Video_S1_Berthing_Comparison.mp4')
    ap.add_argument('--fps',type=int,default=16)
    ap.add_argument('--seconds',type=int,default=58)
    args=ap.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    dfs={}
    for wd in [90,270]:
        name=f'combined_psi20_w8_wd{wd}_c0p25_cd{wd}'
        tc=pd.read_csv(ROOT/'expert_archive/environment_4x4_v31'/name/'trajectory.csv')
        il=pd.read_csv(ROOT/f'il_results/stage4/routed_eval/{wd}/trajectory.csv')
        dfs[wd]=(tc,il)
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(args.fps),'-i','pipe:0','-an','-c:v','libx264','-pix_fmt','yuv420p','-preset','veryfast','-crf','23','-movflags','+faststart',str(args.out)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for j in range(args.seconds*args.fps):
            tm=j/args.fps
            im=frame(tm,dfs)
            if j==round(39*args.fps):
                im.save(args.out.with_name('Video_S1_Thumbnail.png'))
            proc.stdin.write(im.tobytes())
        proc.stdin.close()
        rc=proc.wait()
        if rc:raise RuntimeError('ffmpeg returned '+str(rc))
    except BaseException:
        proc.kill();raise
    print('Saved',args.out)

if __name__=='__main__':main()
