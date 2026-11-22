#!/usr/bin/env python3
"""Paired-replica OPC cooling-path apparent Cv and thermal expansivity.
Uses existing density manifest and frame_geometry.csv; MDOUT map is required for Cv.
Physical rate mapping: 4=>4.44,11=>11.1,22=>22.2,44=>44.4,111=>111 K/ns.
"""
import argparse,csv,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t as td
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.manifest import load_manifest, result_directory, trajectory_list
from common.mdout import nstep_span_matches, parse_dataframe
from common.rates import as_physical_rate, color_for
from common.trajectory import frame_spacing_ps, netcdf_time_ps, ntwx_from_frame_ps, resolve_n_water

REQ=('kind','replica','rate_code','temperature_start_K','temperature_end_K','trajectory','outdir','label')
def stats(x):
    a=np.asarray(x,dtype=float);a=a[np.isfinite(a)];n=len(a)
    if not n:return (np.nan,np.nan,np.nan,0)
    mu=float(np.mean(a));sd=float(np.std(a,ddof=1)) if n>1 else np.nan
    hw=float(td.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else np.nan
    return mu,sd,hw,n
def derivative(g,field,k,nwaters):
    g=g.sort_values('temperature_K');T=g.temperature_K.to_numpy(float);Y=g[field].to_numpy(float)
    valid=np.isfinite(T)&np.isfinite(Y);T=T[valid];Y=Y[valid]
    if len(T)<k:return []
    if len(np.unique(T))!=len(T):raise ValueError('Duplicate temperature points for one replica/rate')
    rows=[];h=k//2
    for i,temp in enumerate(T):
        start=max(0,min(i-h,len(T)-k));tt=T[start:start+k];yy=Y[start:start+k]
        slope=np.polyfit(tt,yy,1)[0]
        # Configurational heat capacity from EPtot, in kcal mol^-1 K^-1 per water.
        # Add 3R for the total heat capacity of rigid water.
        result = (
            slope / nwaters
            if field == 'EPtot'
            else (-slope / Y[i] if field == 'rho_core_g_cm3' else slope / Y[i])
        )
        rows.append({'temperature_K':temp,'value':result,'window_Tmin_K':tt.min(),'window_Tmax_K':tt.max()})
    return rows
def plot(df,field,ylabel,outfile,zero=False):
    fig,ax=plt.subplots(figsize=(8.6,5.3))
    for rate,g in df.groupby('rate_K_ns'):
        g=g.sort_values('temperature_K');color=color_for(rate)
        ax.errorbar(g.temperature_K,g['mean'],yerr=g['sd'],label=f'{rate:g} K/ns',color=color,lw=1.7,marker='o',ms=3.6,capsize=2)
    if zero:ax.axhline(0,color='gray',ls=':',lw=1)
    ax.set(xlabel='Nominal cooling-window midpoint (K)',ylabel=ylabel)
    ax.invert_xaxis();ax.grid(alpha=.17)
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,title='Cooling rate')
    fig.tight_layout();fig.savefig(outfile.with_suffix('.png'),dpi=300,bbox_inches='tight');fig.savefig(outfile.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--density-manifest',required=True,type=Path)
    ap.add_argument('--energy-map',type=Path,help='TSV: trajectory,mdout. Required to compute Cv; omit for alpha-only')
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--topology',type=Path,default=None,help='Required with --energy-map. N is read from this prmtop.')
    ap.add_argument('--nwaters',type=int,default=None,help='Optional cross-check against --topology.')
    ap.add_argument('--window',type=int,default=3,help='Main odd derivative width. Window 5 is always written as the sensitivity check.')
    ap.add_argument('--max-temp-mismatch-K',type=float,default=3.,help='Absolute tolerance between observed mean temperature and nominal chunk midpoint')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.window<3 or a.window%2!=1:ap.error('--window must be odd and >=3')
    if a.energy_map:
        a.nwaters = resolve_n_water(a.topology, a.nwaters)
    manifest=load_manifest(a.density_manifest)
    repo=Path(__file__).resolve().parents[3]
    for row in manifest:
        row['outdir']=str(result_directory(repo, 'profiles', row))
    if not set(REQ).issubset(manifest[0]):raise SystemExit('Invalid density manifest headers')
    mapping={}
    if a.energy_map:
        with a.energy_map.open(newline='') as f:emap=list(csv.DictReader(f,delimiter='\t'))
        if not emap or not {'trajectory','mdout'}.issubset(emap[0]):raise SystemExit('Energy map must be TSV with trajectory and mdout headers')
        for x in emap:
            tr=str(Path(x['trajectory']).resolve());md=Path(x['mdout']).expanduser()
            if tr in mapping:raise ValueError(f'Duplicate trajectory mapping: {tr}')
            mapping[tr]=md
    rows=[];qc=[]
    for r in manifest:
        if r['kind']!='ramp':continue
        try:
            T=(float(r['temperature_start_K'])+float(r['temperature_end_K']))/2
            rate=float(r['rate_K_per_ns']) if str(r.get('rate_K_per_ns','')).strip() else as_physical_rate(int(r['rate_code']))
        except (ValueError,KeyError):continue
        paths=trajectory_list(r)
        if not paths:continue
        tr=str(paths[0].resolve());path=Path(r['outdir'])/'frame_geometry.csv'
        if not path.is_file():qc.append({'trajectory':tr,'status':'MISSING_DENSITY'});continue
        d=pd.read_csv(path)
        needed={'rho_core_g_cm3','h_gds_A','area_A2'}
        if not needed.issubset(d):qc.append({'trajectory':tr,'status':'MISSING_GEOMETRY_COLUMNS'});continue
        rho=float(pd.to_numeric(d.rho_core_g_cm3,errors='coerce').mean())
        h=float(pd.to_numeric(d.h_gds_A,errors='coerce').mean())
        area=float(pd.to_numeric(d.area_A2,errors='coerce').mean())
        if not(np.isfinite(rho) and rho>0 and np.isfinite(h) and h>0):qc.append({'trajectory':tr,'status':'INVALID_DENSITY'});continue
        rec=dict(replica=r['replica'],rate_K_ns=rate,temperature_K=T,trajectory=tr,rho_core_g_cm3=rho,h_gds_A=h,area_A2=area,Etot=np.nan,EPtot=np.nan,n_energy_records=0,temp_mdout_K=np.nan)
        status='DENSITY_ONLY'
        if mapping:
            md=mapping.get(tr)
            if md is None or not md.is_file():status='MISSING_MDOUT_MAPPING_OR_FILE'
            else:
                try:
                    e=parse_dataframe(md)
                    if 'temp_K' not in e or not e.temp_K.notna().any():raise ValueError('Missing instantaneous temperatures')
                    meanT=float(e.temp_K.mean());rec['temp_mdout_K']=meanT
                    if abs(meanT-T)>a.max_temp_mismatch_K:raise ValueError(f'Temperature mismatch {meanT:.1f} versus {T:.1f} K')
                    times=netcdf_time_ps(tr)
                    ntwx=ntwx_from_frame_ps(frame_spacing_ps(times))
                    if not nstep_span_matches(e.nstep.min(), e.nstep.max(), len(times), ntwx):
                        raise ValueError('mdout nstep range does not match the trajectory frame count times ntwx')
                    for ename in ('Etot','EPtot'):rec[ename]=float(e[ename].mean())
                    rec['n_energy_records']=len(e);status='OK_ENERGY'
                except (ValueError,KeyError) as exc:status=f'ENERGY_REJECTED: {exc}'
        qc.append({'trajectory':tr,'mdout':str(mapping.get(tr,'')),'status':status,'nominal_T_K':T,'observed_T_K':rec['temp_mdout_K'],'n_energy_records':rec['n_energy_records']})
        rows.append(rec)
    pd.DataFrame(qc).to_csv(a.out/'input_qc.csv',index=False)
    if not rows:raise SystemExit('No valid density rows. See input_qc.csv')
    chunks=pd.DataFrame(rows);chunks.to_csv(a.out/'thermo_chunk_means.csv',index=False)
    # Reject duplicate chunks for same nominal T to avoid silent mismatching; replicate chunks must be aggregated deliberately.
    dupe=chunks.duplicated(['replica','rate_K_ns','temperature_K'],keep=False)
    if dupe.any():
        chunks.loc[dupe].to_csv(a.out/'duplicate_temperature_rows.csv',index=False)
        raise SystemExit('Duplicate temperatures within replica/rate: see duplicate_temperature_rows.csv')
    def collect(window):
        """Local slopes for one odd temperature window.

        Parameters
        ----------
        window : int
            Number of nominal temperatures in the straight-line fit.

        Returns
        -------
        list of dict
            One derivative row per replica, rate, temperature, and metric.
        """
        derived=[]
        for (rep,rate),g in chunks.groupby(['replica','rate_K_ns']):
            for field,outmetric in [('rho_core_g_cm3','alpha_core_app_1K'),('h_gds_A','alpha_thickness_app_1K'),('EPtot','Cv_app_kcal_mol_K')]:
                gg=g.dropna(subset=[field])
                for r in derivative(gg,field,window,a.nwaters):
                    derived.append(dict(replica=rep,rate_K_ns=rate,metric=outmetric,derivative_window=window,**r))
        return derived
    derived=collect(a.window)
    if not derived:raise SystemExit('No derivatives; check temperature coverage')
    der=pd.DataFrame(derived);der.to_csv(a.out/'thermo_derivatives_by_replica.csv',index=False)
    if a.window!=5:
        pd.DataFrame(collect(5)).to_csv(a.out/'thermo_derivatives_window5_by_replica.csv',index=False)
    summary=[]
    for (rate,temp,metric),g in der.groupby(['rate_K_ns','temperature_K','metric']):
        mu,sd,hw,n=stats(g.value)
        summary.append(dict(rate_K_ns=rate,temperature_K=temp,metric=metric,mean=mu,sd=sd,ci95_halfwidth=hw,n_rep=n))
    summary=pd.DataFrame(summary);summary.to_csv(a.out/'thermo_response_replica_summary.csv',index=False)
    for metric,lab,stem in [('alpha_core_app_1K',r'Apparent $\alpha_{V,core}$ (K$^{-1}$)','alpha_core_vs_T'),('alpha_thickness_app_1K',r'Apparent $\alpha_{h}$ (K$^{-1}$)','alpha_thickness_vs_T'),('Cv_app_kcal_mol_K',r'Configurational $C_{V,\mathrm{app}}$ (kcal mol$^{-1}$ K$^{-1}$)','Cv_apparent_vs_T')]:
        sel=summary[summary.metric==metric]
        if not sel.empty:plot(sel,metric,lab,a.out/stem,zero=(metric!='Cv_app_kcal_mol_K'))
    # Cross-check alpha_core and thickness estimates at fixed A
    wide=der.pivot_table(index=['replica','rate_K_ns','temperature_K'],columns='metric',values='value',aggfunc='first').reset_index()
    if {'alpha_core_app_1K','alpha_thickness_app_1K'}.issubset(wide):
        wide['alpha_disagreement_1K']=wide['alpha_core_app_1K']-wide['alpha_thickness_app_1K']
    wide.to_csv(a.out/'alpha_consistency_by_replica.csv',index=False)
    print('Completed:',a.out,'; density chunks:',len(chunks),'energy matched:',sum(pd.DataFrame(qc).status=='OK_ENERGY'))
    print(f'Main derivative window is {a.window} points. Window 5 is thermo_derivatives_window5_by_replica.csv.')
    print('C_V,app is dEPtot/dT per water. Add 3R for the total heat capacity of rigid water.')
    if a.energy_map and not any(pd.DataFrame(qc).status=='OK_ENERGY'):print('WARNING: No energy matches: only alpha figures were created. See input_qc.csv',file=sys.stderr)
if __name__=='__main__':main()
