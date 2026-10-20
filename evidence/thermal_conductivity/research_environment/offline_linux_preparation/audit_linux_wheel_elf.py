import argparse,json,re,struct,zipfile
from pathlib import Path

def read_at(z,name,offset,size):
    with z.open(name) as f:
        while offset:
            b=f.read(min(offset,1024*1024))
            if not b:raise ValueError('short ELF stream')
            offset-=len(b)
        b=f.read(size)
        if len(b)!=size:raise ValueError('short ELF region')
        return b

def inspect(z,name):
    header=read_at(z,name,0,64)
    if header[:4]!=b'\x7fELF':return None
    assert header[4:6]==b'\x02\x01',(name,'not ELF64 little endian')
    h=struct.unpack('<16sHHIQQQIHHHHHH',header);assert h[2]==62,(name,'not x86_64')
    raw=read_at(z,name,h[5],h[9]*h[10]);ph=[struct.unpack('<IIQQQQQQ',raw[i*h[9]:i*h[9]+56]) for i in range(h[10])]
    dyn=next((x for x in ph if x[0]==2),None)
    if dyn is None:return {'path':name,'architecture':'x86_64','needed':[]}
    raw=read_at(z,name,dyn[2],dyn[5]);dt=[]
    for i in range(0,len(raw),16):
        k,v=struct.unpack('<qQ',raw[i:i+16])
        if k==0:break
        dt.append((k,v))
    st=next(v for k,v in dt if k==5);sz=next(v for k,v in dt if k==10)
    load=next(x for x in ph if x[0]==1 and x[3]<=st<x[3]+x[5]);strings=read_at(z,name,load[2]+st-load[3],sz)
    get=lambda i:strings[i:strings.index(b'\0',i)].decode('utf-8')
    versions=re.findall(rb'GLIBC_(\d+\.\d+(?:\.\d+)?)',strings)
    return {'path':name,'architecture':'x86_64','needed':[get(v) for k,v in dt if k==1],'soname':next((get(v) for k,v in dt if k==14),None),'rpath':next((get(v) for k,v in dt if k==15),None),'runpath':next((get(v) for k,v in dt if k==29),None),'glibc_version_tags':sorted(set(v.decode() for v in versions),key=lambda s:tuple(map(int,s.split('.'))))}

ap=argparse.ArgumentParser();ap.add_argument('--wheels',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();records=[];provided=set();needed=set()
for p in sorted(a.wheels.glob('*.whl')):
    rows=[]
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            if not ('.so' in Path(n).name or '/bin/' in n) or n.endswith('/'):continue
            if z.getinfo(n).file_size<64:continue
            row=inspect(z,n)
            if row:
                rows.append(row);provided.add(Path(n).name)
                if row.get('soname'):provided.add(row['soname'])
                needed.update(row['needed'])
    if rows:records.append({'wheel':p.name,'elf_files':rows})
external=sorted(needed-provided)
expected={'libc.so.6','libdl.so.2','libgcc_s.so.1','libm.so.6','libpthread.so.0','librt.so.1','libstdc++.so.6','libz.so.1','ld-linux-x86-64.so.2','libutil.so.1'}
unknown=set(external)-expected;assert not unknown,unknown
versions=[v for w in records for r in w['elf_files'] for v in r.get('glibc_version_tags',[])];highest=max(versions,key=lambda s:tuple(map(int,s.split('.'))))
assert tuple(map(int,highest.split('.')))<=(2,28),highest
result={'scope':'Static ELF DT_NEEDED audit of all shared objects and wheel bin executables; no Linux native code was executed on the macOS build host.','elf_files':sum(len(w['elf_files']) for w in records),'native_wheels':len(records),'all_elf_architectures':'x86_64','highest_required_glibc_symbol_version':highest,'external_system_libraries':external,'base_system_assumption':'Operator CPython 3.11 base is Debian Bookworm (CPU) or its documented Linux GPU base with build-essential. glibc, libgcc, libstdc++ and libz are required from that base. Build-time native-import checks will fail if any is missing.','unexpected_external_libraries':sorted(unknown),'wheels':records}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='wheels'}))
