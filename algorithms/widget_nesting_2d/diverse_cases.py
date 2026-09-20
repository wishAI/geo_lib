"""Mixed source outlines and deterministic, structurally varied stress geometry."""
import copy
import json
import math
import random
from pathlib import Path

from shapely.affinity import scale, translate
from shapely.geometry import Point, Polygon, MultiPoint, box
from shapely.ops import unary_union


def _spec(polygon):
    if polygon.geom_type != 'Polygon' or not polygon.is_valid or polygon.area <= 0:
        raise ValueError('Diversity generator produced invalid or disconnected geometry')
    x, y, _, _ = polygon.bounds
    polygon = translate(polygon, xoff=-x, yoff=-y)
    return {'shell': list(map(list, polygon.exterior.coords[:-1])),
            'holes': [list(map(list, r.coords[:-1])) for r in polygon.interiors]}


def _procedural(index, rng, *, hole_free=False):
    w, h = rng.uniform(9,28), rng.uniform(8,25)
    a, b = rng.uniform(.15,.4), rng.uniform(.2,.45)
    family = index % 12
    if family == 0:
        p = MultiPoint([(w*rng.random(),h*rng.random()) for _ in range(9+(index//12)%6)]).convex_hull
        name = 'convex'
    elif family == 1:
        p = Polygon([(0,h*b),(w*.6,h*b),(w*.6,0),(w,h*.5),(w*.6,h),(w*.6,h*(1-b)),(0,h*(1-b))]); name='arrow'
    elif family == 2:
        p = box(0,0,w,h).difference(box(w*a,h*b,w+1,h+1)); name='L_bracket'
    elif family == 3:
        p = unary_union([box(0,h*(1-b),w,h),box(w*a,0,w*(1-a),h)]); name='T_tab'
    elif family == 4:
        p = box(0,0,w,h).difference(box(w*a,h*b,w*(1-a),h+1)); name='U_channel'
    elif family == 5:
        p = Polygon([(0,0),(w,0),(w,h*b),(w*a,h*(1-b)),(w,h*(1-b)),(w,h),(0,h),(0,h*(1-b)),(w*(1-a),h*b),(0,h*b)]); name='zigzag'
    elif family == 6:
        teeth=3+index%5
        p=unary_union([box(0,0,w,h*b)]+[box(j*w/teeth,0,(j+.5+a)*w/teeth,h) for j in range(teeth)]); name='comb'
    elif family == 7:
        n=5+(index//12)%6
        p=Polygon([(w*.5+w*.5*(1 if j%2==0 else a+.25)*math.cos(math.pi*j/n),
                    h*.5+h*.5*(1 if j%2==0 else a+.25)*math.sin(math.pi*j/n)) for j in range(2*n)]); name='star'
    elif family == 8:
        p=Point(0,0).buffer(1,quad_segs=12).difference(Point(a+.25,0).buffer(.9,quad_segs=12))
        p=scale(p,xfact=w/2,yfact=h/2,origin=(0,0)); name='crescent'
    elif family == 9:
        radius=min(w/3,h/2)
        p=unary_union([Point(radius,radius).buffer(radius,quad_segs=8),
                       Point(w-radius,radius).buffer(radius*(.75+a/2),quad_segs=8),
                       box(radius,radius*(1-b),w-radius,radius*(1+b))]); name='dogbone'
    elif family == 10:
        p=box(0,0,w,h).difference(unary_union([box(w*a,-1,w*(a+b),h*b),
                                             box(w*(1-b),h*(1-a),w+1,h+1)])); name='notched_plate'
    elif hole_free:
        p=Polygon([(0,0),(w,0),(w,h*a),(w*(1-a),h*a),(w*(1-a),h*.65),
                   (w*b,h*.65),(w*b,h),(0,h)]); name='stairs'
    else:
        n=1+(index//12)%3
        p=box(0,0,w,h).difference(unary_union([box(w*(j+.2)/n,h*b,w*(j+.8)/n,h*(1-a)) for j in range(n)])); name='frame'
    return name, _spec(p)


def build_diverse_cases(base_cases, CaseDefinition):
    root=Path(__file__).resolve().parent/'inputs'
    catalog=[]
    def add_family(family, polygons):
        factor=24/max(max(p.bounds[2]-p.bounds[0],p.bounds[3]-p.bounds[1]) for _,p in polygons)
        for key,p in polygons:
            catalog.append({'id':f'{family}_{key}','quantity':1,'allowed_angles_degrees':[0,90,180,270],
                            'polygon':_spec(scale(p,xfact=factor,yfact=factor,origin=(0,0))), 'family':family})
    for family in ('shirts','trousers','swim'):
        data=json.loads((root/'public_benchmarks'/f'{family}.json').read_text())
        add_family(family,[(str(w['id']),Polygon(w['shape']['data'])) for w in data['items']])
    alphabet=next(c.problem for c in base_cases if c.case_id=='alphabet_abcdefghijklmnopqrstuvwxyz')
    add_family('letter',[(w['id'],Polygon(w['polygon']['shell'],w['polygon'].get('holes',[]))) for w in alphabet['widgets']])
    existing=json.loads((root/'complex_dual_board.json').read_text())
    add_family('existing',[(w['id'],Polygon(w['polygon']['shell'],w['polygon'].get('holes',[]))) for w in existing['widgets']])
    assert len(catalog)==66
    def extend(count, hole_free):
        widgets=copy.deepcopy([w for w in catalog if not hole_free or not w['polygon']['holes']])
        rng=random.Random(90421)
        for index in range(count-len(widgets)):
            family,polygon=_procedural(index,rng,hole_free=hole_free)
            widgets.append({'id':f'generated_{family}_{index}','quantity':1,'allowed_angles_degrees':[0,90,180,270],
                            'polygon':polygon,'family':family})
        return widgets
    recipes=[('source_mix_66',66,66,False,None),('source_mix_66_dense',66,66,False,.60),
             ('diverse_128',128,128,False,None),('diverse_256',256,256,False,None),
             ('diverse_512',256,512,False,None),('diverse_1000',256,1000,False,None),
             ('simple_mix_128_dense',128,128,True,.55),('diverse_256_dense',256,256,False,.60),
             ('simple_mix_256_dense',256,256,True,.60),('simple_mix_1000',256,1000,True,None),
             ('unique_1000',1000,1000,False,None)]
    cases=[]
    for name,types,count,hole_free,load in recipes:
        widgets=extend(types,hole_free)
        for i,w in enumerate(widgets):
            w['quantity']=count//types+int(i<count%types)
        rng=random.Random(163);rng.shuffle(widgets)
        polygons=[Polygon(w['polygon']['shell'],w['polygon']['holes']) for w in widgets]
        cw=max(p.bounds[2] for p in polygons)+.5
        ch=max(p.bounds[3] for p in polygons)+.5
        columns=math.ceil(math.sqrt(count*ch/cw*1.5));rows=math.ceil(count/columns)
        width,height=columns*cw,rows*ch
        witness=None
        if load:
            area=sum(p.area*w['quantity'] for p,w in zip(polygons,widgets))
            height=math.sqrt(area/load/1.5);width=height*1.5
        else:
            witness={'method':'independent nonoverlapping grid cells','placements':[]}
            index=0
            for w in widgets:
                for j in range(w['quantity']):
                    witness['placements'].append({'item':f"{w['id']}#{j+1}", 'x':index%columns*cw,
                                                   'y':index//columns*ch,'rotation':0})
                    index+=1
        families={family:sum(w['family']==family for w in widgets) for family in {w['family'] for w in widgets}}
        source={'kind':'diverse_mixed_outlines','distinct_types':types,'instances':count,'families':families,
                'known_feasible':witness is not None,'nominal_area_load':load,
                'sources':['jagua-rs shirts/trousers/swim','DejaVu Sans','complex_dual_board.json','12 procedural families'],
                'normalization':'one uniform scale per source family; aspect ratios and holes preserved'}
        for w in widgets:
            del w['family']
        problem={'units':'test_unit','source':source,
                 'boards':[{'id':'board','polygon':{'shell':[[0,0],[width,0],[width,height],[0,height]]}}],
                 'widgets':widgets,'config':{'rotation_step_degrees':90,'beam_width':2,'population_size':12,
                                            'generations':100000,'max_candidates_per_item':12}}
        cases.append(CaseDefinition(name,f'{count} instances / {types} distinct mixed outlines; '+
                                    ('certified capacity case' if witness else 'dense quality challenge'),problem,witness))
    return cases
