#!/usr/bin/env python3
"""Actual offline initial geometry -> CHGNet -> bounded cell/atom relaxation.

Reads only the task's clean input structures, elemental references and local
checkpoint. Produces one demonstration formation-energy value, not a complete
215488-row benchmark submission and not an accuracy or convergence claim.
"""
import argparse,collections,csv,gzip,hashlib,importlib.metadata,json,math,os,platform,socket,time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
NETWORK_ATTEMPTS=[]
def no_network(*args,**kwargs):
    NETWORK_ATTEMPTS.append('blocked_socket_connection')
    raise RuntimeError('Socket network connections are forbidden for this offline smoke')
socket.create_connection=no_network
socket.socket.connect=no_network
socket.socket.connect_ex=no_network

import numpy as np
import torch
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from pymatgen.core import Structure


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def serialize(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)

def predict(model,numbers,cell,positions):
    s=Structure(cell,numbers,positions,coords_are_cartesian=True)
    result={k:np.asarray(v,dtype=np.float64) for k,v in model.predict_structure(s,task='efs').items()}
    assert all(np.isfinite(v).all() for v in result.values())
    assert result['f'].shape==(len(numbers),3) and result['s'].shape==(3,3)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--steps',type=int,default=10);ap.add_argument('--fmax',type=float,default=0.01);a=ap.parse_args()
    if not 1<=a.steps<=100:raise ValueError('Use 1..100 steps for this bounded smoke')
    if not math.isfinite(a.fmax) or a.fmax<=0:raise ValueError('fmax must be finite and positive')
    a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    references_path=a.input/'elemental_reference_energies.json';structure_path=a.input/'test_structures.jsonl.gz';checkpoint=a.input/'model_weights/chgnet_0.3.0_e29f68s314m37.pth.tar'
    references=json.loads(references_path.read_text());assert len(references)==89
    ref={int(r['atomic_number']):float(r['reference_energy_eV_per_atom']) for r in references}
    assert all(math.isfinite(v) for v in ref.values())
    selected=None;skipped=[]
    with gzip.open(structure_path,'rt') as f:
        for index,line in enumerate(f,1):
            row=json.loads(line);numbers=row['atomic_numbers'];cell=np.asarray(row['cell_angstrom'],float);positions=np.asarray(row['positions_angstrom'],float)
            valid=(len(numbers)>0 and positions.shape==(len(numbers),3) and cell.shape==(3,3) and np.isfinite(positions).all() and np.isfinite(cell).all() and abs(float(np.linalg.det(cell)))>1e-6 and all(1<=int(z)<=94 and z in ref for z in numbers) and row['pbc']==[True,True,True])
            if valid:selected=row;selected_row=index;break
            skipped.append({'material_id':row['material_id'],'reason':'geometry, supported atomic number, periodicity or elemental reference precheck failed'})
    assert selected is not None
    checkpoint_sha=sha(checkpoint);assert checkpoint_sha=='d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1'
    model=CHGNet.from_file(str(checkpoint),mlp_out_bias=False,version='0.3.0').to('cpu');model.eval()
    numbers=selected['atomic_numbers'];cell=np.asarray(selected['cell_angstrom'],float);positions=np.asarray(selected['positions_angstrom'],float)
    initial=predict(model,numbers,cell,positions)
    atoms=Atoms(numbers=numbers,positions=positions,cell=cell,pbc=selected['pbc'])
    atoms.calc=CHGNetCalculator(model=model,use_device='cpu',stress_weight=1/160.21766208,on_isolated_atoms='error')
    filtered=FrechetCellFilter(atoms)
    history=[]
    def record():
        f=atoms.get_forces();ff=filtered.get_forces();stress=atoms.get_stress(voigt=False);energy=float(atoms.get_potential_energy());volume=float(atoms.get_volume())
        assert all(np.isfinite(x).all() for x in [f,ff,stress,atoms.positions,atoms.cell.array])
        assert math.isfinite(energy) and math.isfinite(volume) and volume>0
        history.append({'optimizer_step':len(history),'energy_eV_total':energy,'energy_eV_per_atom':energy/len(atoms),'volume_A3':volume,'max_atomic_force_eV_per_A':float(np.linalg.norm(f,axis=1).max()),'max_filter_force':float(np.linalg.norm(ff,axis=1).max()),'stress_eV_per_A3':stress.tolist(),'positions_angstrom':atoms.positions.tolist(),'cell_angstrom':atoms.cell.array.tolist()})
    optimizer=FIRE(filtered,dt=0.05,maxstep=0.05,logfile=str(a.output/'relaxation.log'))
    optimizer.attach(record,interval=1)
    converged=bool(optimizer.run(fmax=a.fmax,steps=a.steps))
    final=predict(model,numbers,atoms.cell.array,atoms.positions)
    assert optimizer.nsteps>0,'Smoke requires at least one real geometry update'
    assert len(history)==optimizer.nsteps+1
    final_energy=float(atoms.get_potential_energy())/len(atoms)
    assert math.isclose(final_energy,float(final['e']),rel_tol=0,abs_tol=1e-6)
    reference_average=sum(ref[z] for z in numbers)/len(numbers)
    formation_energy=final_energy-reference_average
    assert math.isfinite(formation_energy)
    cell_change=float(np.linalg.norm(atoms.cell.array-cell));position_change=float(np.linalg.norm(atoms.positions-positions))
    assert cell_change>0 and position_change>0
    assert converged==(history[-1]['max_filter_force']<a.fmax)
    assert not NETWORK_ATTEMPTS
    with (a.output/'one_row_formation_energy_example.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['material_id','e_form_eV_per_atom']);writer.writerow([selected['material_id'],format(formation_energy,'.16g')])
    result={'status':'passed_actual_initial_structure_efs_cell_atom_relaxation_and_formation_energy','scope':'One legal initial test structure; genuine bounded relaxation and elemental-reference subtraction. No withheld labels read. No accuracy score, full benchmark prediction, or full-convergence claim.','material_id':selected['material_id'],'selected_input_row_1_based':selected_row,'selection_policy':'First row with finite positive-volume periodic geometry, CHGNet-supported atomic numbers 1..94 and available elemental references; no target-label-based selection.','skipped':skipped,'input_files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in [structure_path,references_path,checkpoint]},'input_structure':selected,'element_reference_count':len(references),'composition_counts':dict(collections.Counter(numbers)),'elemental_references_used_eV_per_atom':{str(z):ref[z] for z in sorted(set(numbers))},'initial_prediction':initial,'final_prediction':final,'relaxation':{'optimizer':'ASE FIRE','cell_filter':'FrechetCellFilter','relax_atoms':True,'relax_cell':True,'max_steps':a.steps,'performed_steps':optimizer.nsteps,'fmax':a.fmax,'fmax_scope':'Maximum norm of all FrechetCellFilter generalized forces, including cell components, as used by the ASE optimizer.','dt':0.05,'maxstep':0.05,'converged':converged,'cell_change_frobenius_A':cell_change,'cartesian_position_change_frobenius_A':position_change,'trajectory':history},'formation_energy':{'final_total_model_energy_eV_per_atom':final_energy,'stoichiometric_mean_elemental_reference_eV_per_atom':reference_average,'predicted_formation_energy_eV_per_atom':formation_energy,'formula':'E_model_total / N_atoms - sum(n_Z * E_reference_Z) / N_atoms','additional_MP2020_compatibility_corrections_applied':False,'submission_scope':'Only one demonstration row, not a valid complete benchmark submission.'},'host':{'platform':platform.platform(),'machine':platform.machine(),'python':platform.python_version()},'versions':{n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','chgnet','pymatgen','ase']},'device':'cpu','torch_threads':2,'network_block':'Python socket.connect, connect_ex and create_connection disabled before scientific imports; no OS network namespace claim.','blocked_network_attempt_count':len(NETWORK_ATTEMPTS),'timing_scope':'Elapsed calculation/setup time inside main(); interpreter and import startup excluded.','wall_seconds':time.monotonic()-start}
    (a.output/'smoke_result.json').write_text(json.dumps(result,indent=2,default=serialize,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'material_id':selected['material_id'],'steps':optimizer.nsteps,'converged':converged,'formation_energy_eV_per_atom':formation_energy,'wall_seconds':result['wall_seconds']}))
if __name__=='__main__':main()
