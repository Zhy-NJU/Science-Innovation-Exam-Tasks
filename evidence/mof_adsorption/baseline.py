#!/usr/bin/env python3
"""Pure-stdlib training-only median reference, no hidden-data access."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics

def rows(paths):
    for path in paths:
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:yield json.loads(line)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    groups=defaultdict(list)
    all_train=[]
    train_paths=sorted((a.input_dir/'train').glob('part-*.jsonl.gz'))
    assert train_paths, 'Missing training shards'
    for row in rows(train_paths):
        value=float(row['adsorption_energy_eV'])
        groups[(int(row['nco2']),int(row['nh2o']))].append(value)
        all_train.append(value)
    assert len(all_train)==162224
    medians={key:statistics.median(values) for key,values in groups.items()}
    fallback=statistics.median(all_train)
    predictions=[]
    fallback_count=0
    with gzip.open(a.input_dir/'test.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            assert 'adsorption_energy_eV' not in row
            key=(int(row['nco2']),int(row['nh2o']))
            fallback_count += int(key not in medians)
            predictions.append((row['system_id'],medians.get(key,fallback)))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,lineterminator='\n')
        w.writerow(['system_id','adsorption_energy_eV'])
        for sid,value in sorted(predictions):
            w.writerow([sid,repr(value)])
    print(json.dumps({'training_count':len(all_train),'test_predictions':len(predictions),'group_train_counts':{f'{c},{w}':len(values) for (c,w),values in sorted(groups.items())},'medians_eV':{f'{c},{w}':y for (c,w),y in sorted(medians.items())},'global_median_eV':fallback,'fallback_count':fallback_count},indent=2))

if __name__=='__main__':
    main()
