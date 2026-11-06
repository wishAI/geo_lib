"""Ordered behavioral and speed gates, preserving historical milestone IDs."""


def ordered_milestones(ledger):
    records=[*ledger.get('milestones',[]),*ledger.get('speedMilestones',[])]
    by_id={item['id']:item for item in records}
    order=ledger.get('executionOrder',[item['id'] for item in records])
    if len(by_id)!=len(records) or len(order)!=len(records) or set(order)!=set(by_id):
        raise ValueError('Milestone execution order must include every gate exactly once')
    return [by_id[key] for key in order]


def active_milestone(ledger):
    return next((item for item in ordered_milestones(ledger) if item.get('status')!='passed'),None)


def advance_schedule(ledger):
    records=ordered_milestones(ledger);current=active_milestone(ledger)
    for item in records:
        if item.get('status')!='passed':item['status']='in_progress' if item is current else 'not_started'
    ledger['currentGateId']=current['id'] if current else None
    ledger['implementationStatus']='all_milestones_passed' if current is None else current['id']+'_in_progress'
    if 'sessionStatus' in ledger:
        session=ledger['sessionStatus'];session['certifiedMilestones']=sum(item.get('status')=='passed' for item in records)
        session['nextMilestone']=ledger['currentGateId']
        if 'currentExperiment' in session:session['currentExperiment']['nextMilestone']=ledger['currentGateId']


def require_current_gate(ledger,gate_id):
    current=active_milestone(ledger)
    if current is None or current['id']!=gate_id:
        raise ValueError('Only the earliest unresolved scheduled gate can be certified')
