#!/usr/bin/env python3
"""Regional relaxation, hydrogen-bond retention, and translation-rotation decoupling.

Resolved times are plotted. Censored times are drawn as bounds and are not
extrapolated. Paired differences use the same replica and temperature.
The bulk is a solid line and the interface is dashed. The temperature axis
runs from hot to cold. Missing optional inputs are reported and not invented.

Example::

    python plot_all.py --corrected-dir corrected/ --hb-dir hbonds/ --out figures --rate 4.44
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import REGION_STYLE
from common.contrast import bulk_minus_awi
import numpy as np
import pandas as pd
from scipy.stats import t as tdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

REGIONS = REGION_STYLE
METRICS = {'Fs': (r'$F_s^{\parallel}$', 'o', '#1f77b4'), 'C1': (r'$C_1$', 's', '#d95f02'), 'C2': (r'$C_2$', '^', '#2ca02c')}
K = ['replica', 'rate', 'temperature_K', 'region']


def design():
    """Matplotlib defaults for the regional-dynamics figures.

    Returns
    -------
    result
        Matplotlib defaults for the regional-dynamics figures.
    """
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,
        'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.9,
        'lines.linewidth':1.9,'savefig.dpi':300,'pdf.fonttype':42,'ps.fonttype':42})


def figure():
    """A new figure and axes at the shared size.

    Returns
    -------
    result
        A new figure and axes at the shared size.
    """
    return plt.subplots(figsize=(8.1,4.9))


def finish(fig,ax,out,temperature_axis=False,logx=False,logy=False,ylim=None):
    """Style, save, and close one figure. Returns the PNG and PDF paths.

    Parameters
    ----------
    fig : 
        Matplotlib figure.
    ax : 
        Matplotlib axes.
    out : 
        Output path or directory.
    temperature_axis : 
        Invert the temperature axis so cooling runs left to right.
    logx : 
        Use a logarithmic horizontal axis.
    logy : 
        Use a logarithmic vertical axis.
    ylim : 
        Vertical axis limits.

    Returns
    -------
    result
        Style, save, and close one figure. Returns the PNG and PDF paths.
    """
    if temperature_axis: ax.invert_xaxis()
    if logx:ax.set_xscale('log')
    if logy:ax.set_yscale('log')
    if ylim is not None:ax.set_ylim(*ylim)
    ax.grid(alpha=.15,which='major')
    if logx or logy:ax.grid(alpha=.065,which='minor')
    ax.tick_params(direction='out',length=3)
    handles,labels=ax.get_legend_handles_labels()
    if handles:ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,title='Series')
    fig.tight_layout()
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out.with_suffix('.png'),dpi=300,bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)
    return [str(out.with_suffix('.png')),str(out.with_suffix('.pdf'))]


def mean_sd(vals):
    """Mean, sample standard deviation, and count of the finite values.

    Parameters
    ----------
    vals : 
        Numeric observations.

    Returns
    -------
    result
        Mean, sample standard deviation, and count of the finite values.
    """
    x=np.asarray(vals,float);x=x[np.isfinite(x)]
    if not len(x):return np.nan,np.nan,0
    return float(x.mean()),float(x.std(ddof=1)) if len(x)>1 else np.nan,len(x)


def clean_rate(df, rate):
    """Rows for one physical cooling rate.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with a rate column. A stored directory code is converted.
    rate : float
        Physical rate in K/ns.

    Returns
    -------
    pandas.DataFrame
        Matching rows.
    """
    from common.rates import select_rate
    return select_rate(df, rate)


def temperature_colors(temps):
    """A color for each temperature on a cool-warm scale.

    Parameters
    ----------
    temps : 
        Temperatures in kelvin.

    Returns
    -------
    result
        A color for each temperature on a cool-warm scale.
    """
    vals=sorted(set(float(t) for t in temps))
    low=min(vals);high=max(vals)
    normal=Normalize(vmin=low-1e-6,vmax=high+1e-6)
    palette=plt.get_cmap('coolwarm')
    return {t:palette(normal(t)) for t in vals}


def available_csv(path,columns):
    """The CSV if it exists and contains the requested columns, otherwise None.

    Parameters
    ----------
    path : 
        Input file or directory.
    columns : 
        Column names.

    Returns
    -------
    result
        The CSV if it exists and contains the requested columns, otherwise None.
    """
    if path is None or not Path(path).is_file():return None
    df=pd.read_csv(path)
    absent=set(columns)-set(df.columns)
    if absent:raise ValueError(f'{path}: missing columns {sorted(absent)}')
    return df


def plot_relaxation_curves(rep,out,rate,temps,band):
    """Regional correlation functions at the requested temperatures.

    Parameters
    ----------
    rep : 
        Per-replica table.
    out : 
        Output path or directory.
    rate : 
        Cooling-rate code or rate in K/ns, as stored in the table.
    temps : 
        Temperatures in kelvin.
    band : 
        Half-width of the shaded band in multiples of the replica standard deviation.

    Returns
    -------
    result
        Regional correlation functions at the requested temperatures.
    """
    d=clean_rate(rep,rate);d=d[d.region.isin(REGIONS)]
    d=d[d.temperature_K.isin(temps)] if temps else d
    if d.empty:return []
    color_by_temp=temperature_colors(d.temperature_K.unique())
    results=[]
    for metric,(display,_,_) in METRICS.items():
        if metric not in d:continue
        fig,ax=figure();num=0
        for T in sorted(d.temperature_K.unique(),reverse=True):
            for region,(label,ls) in REGIONS.items():
                group=d[(d.temperature_K==T)&(d.region==region)]
                if group.empty:continue
                # The independent sample unit is a replica, not a lag point or water molecule.
                s=group.groupby('lag_ps')[metric].agg(['mean','std','count']).reset_index().sort_values('lag_ps')
                s=s[(s.lag_ps>0)&np.isfinite(s['mean'])]
                if s.empty:continue
                x=s.lag_ps.to_numpy(float);y=s['mean'].to_numpy(float);sd=s['std'].to_numpy(float)
                ax.plot(x,y,ls,color=color_by_temp[float(T)],label=f'{T:g} K — {label}',lw=1.85)
                if band>0 and np.isfinite(sd).any():
                    lo=np.clip(y-band*np.nan_to_num(sd),-1,1);hi=np.clip(y+band*np.nan_to_num(sd),-1,1)
                    ax.fill_between(x,lo,hi,where=np.isfinite(sd),color=color_by_temp[float(T)],alpha=.075,lw=0)
                num+=1
        if not num:plt.close(fig);continue
        ax.axhline(np.exp(-1),color='.4',ls=':',lw=1)
        ax.set(xlabel='Lag time (ps)',ylabel=display)
        results+=finish(fig,ax,out/f'V4_{metric}_regional_correlation',logx=True)
    return results


def plot_tau_and_bounds(tau,summ,out,rate,band):
    """Resolved relaxation times and the censoring bounds.

    Parameters
    ----------
    tau : 
        Relaxation time.
    summ : 
        Temperature-level summary table.
    out : 
        Output path or directory.
    rate : 
        Cooling-rate code or rate in K/ns, as stored in the table.
    band : 
        Half-width of the shaded band in multiples of the replica standard deviation.

    Returns
    -------
    result
        Resolved relaxation times and the censoring bounds.
    """
    raw=clean_rate(tau,rate); raw=raw[raw.region.isin(REGIONS)]
    summ=clean_rate(summ,rate);summ=summ[summ.region.isin(REGIONS)]
    results=[]
    for metric,(title,mark,color) in METRICS.items():
        d=raw[raw.metric==metric];s=summ[summ.metric==metric]
        if d.empty:continue
        fig,ax=figure(); nseries=0
        for reg,(label,ls) in REGIONS.items():
            z=s[(s.region==reg)&(s.n_resolved>=2)&(s.n_resolved==s.n_replicas)].sort_values('temperature_K')
            z=z[np.isfinite(z.log10_tau_mean)&np.isfinite(z.log10_ci_low)&np.isfinite(z.log10_ci_high)]
            if not z.empty:
                x=z.temperature_K.to_numpy(float);y=10**z.log10_tau_mean.to_numpy(float)
                lo=10**z.log10_ci_low.to_numpy(float);hi=10**z.log10_ci_high.to_numpy(float)
                ax.plot(x,y,ls,color=color,marker='o' if reg=='interior' else 's',ms=4.5,label=f'{label}: resolved')
                ax.fill_between(x,lo,hi,color=color,alpha=.11,lw=0)
                nseries+=1
            for status,sym,factor,lab in [('slow_censored','^',1.28,'lower bound'),('fast_censored','v',.78,'upper bound')]:
                v=d[(d.region==reg)&(d.status==status)&np.isfinite(d.bound_ps)]
                if v.empty:continue
                ax.scatter(v.temperature_K,v.bound_ps,marker=sym,s=42,facecolors='white',edgecolors=color,linewidth=1,alpha=.9,zorder=3,
                           label=f'{label}: {lab}')
                for rr in v.itertuples():
                    yy=float(rr.bound_ps)
                    if yy<=0:continue
                    ax.annotate('',xy=(rr.temperature_K, yy*factor),xytext=(rr.temperature_K,yy),
                        arrowprops={'arrowstyle':'->','color':color,'lw':.75},annotation_clip=False)
                nseries+=1
        if not nseries:plt.close(fig);continue
        ax.set(xlabel='Cooling-window midpoint temperature (K)',ylabel=f'{title} relaxation time or bound (ps)')
        results+=finish(fig,ax,out/f'V4_tau_{metric}_censoring',temperature_axis=True,logy=True)
    return results


def paired_contrast(pairs, out, rate):
    """Plot D_TR,1 and D_TR,2 using matched, resolved Fs/C1/C2 tau records.

    Input decoupling_paired_resolved.csv stores *times* (ps) in Fs, C1, C2.
    Compute per replica before any averaging, and use only same-replica
    AWI/interior pairs for spatial differences. No censor extrapolation.

    Parameters
    ----------
    pairs : 
        Replica-matched relaxation times.
    out : 
        Output path or directory.
    rate : 
        Cooling-rate code or rate in K/ns, as stored in the table.

    Returns
    -------
    result
        Plot D_TR,1 and D_TR,2 using matched, resolved Fs/C1/C2 tau records.
    """
    x = clean_rate(pairs, rate)
    x = x[x.region.isin(REGIONS)].copy()
    if x.empty:
        return []
    keys = ['replica', 'rate', 'temperature_K', 'region']
    if x.duplicated(keys).any():
        raise ValueError('Duplicate paired decoupling records')
    needed = {'Fs', 'C1', 'C2'}
    missing = needed - set(x.columns)
    if missing:
        raise ValueError('To plot DTR1 and DTR2, the resolved decoupling CSV must '
                         f'contain Fs/C1/C2 (missing {sorted(missing)})')
    for metric in ('Fs', 'C1', 'C2'):
        x[metric] = pd.to_numeric(x[metric], errors='coerce')
    valid = np.isfinite(x[['Fs', 'C1', 'C2']]).all(axis=1) & (x[['Fs', 'C1', 'C2']] > 0).all(axis=1)
    if (~valid).any():
        print(f'WARNING: dropping {(~valid).sum()} rows with nonfinite/nonpositive '
              'relaxation times; no extrapolation performed', file=sys.stderr)
    x = x.loc[valid].copy()
    if x.empty:
        return []
    x['D_TR1_log10'] = np.log10(x['C1'] / x['Fs'])
    x['D_TR2_log10'] = np.log10(x['C2'] / x['Fs'])
    if 'D_TR_log10' in x.columns:
        old = pd.to_numeric(x['D_TR_log10'], errors='coerce')
        has_old = np.isfinite(old)
        if has_old.any() and not np.allclose(
            old[has_old], x.loc[has_old, 'D_TR1_log10'], rtol=1e-7, atol=1e-8
        ):
            raise ValueError('Existing D_TR_log10 disagrees with log10(C1/Fs); '
                             'check source definitions before plotting.')
    out.mkdir(parents=True, exist_ok=True)
    x.to_csv(out / 'V4_DTR1_DTR2_resolved_replica_values.csv', index=False)
    outputs = []
    for order, column in ((1, 'D_TR1_log10'), (2, 'D_TR2_log10')):
        prefix = f'V4_DTR{order}'
        # Keep original V4 names for DTR1, to avoid breaking existing references.
        legacy = 'V4_DTR' if order == 1 else prefix
        stats = []
        for (T, reg), v in x.groupby(['temperature_K', 'region']):
            mu, sd, n = mean_sd(v[column])
            stats.append({'temperature_K': T, 'region': reg,
                          'mean': mu, 'sd': sd, 'n': n})
        rs = pd.DataFrame(stats).sort_values(['temperature_K', 'region'])
        rs.to_csv(out / f'{prefix}_region_stats.csv', index=False)
        if order == 1:
            rs.to_csv(out / 'V4_DTR_region_stats.csv', index=False)
        fig, ax = figure()
        for reg, (lab, ls) in REGIONS.items():
            z = rs[rs.region == reg].sort_values('temperature_K')
            if z.empty:
                continue
            xx = z.temperature_K.to_numpy(float)
            yy = z['mean'].to_numpy(float)
            sd = z['sd'].to_numpy(float)
            ax.plot(xx, yy, ls, color='black',
                    marker='o' if reg == 'interior' else 's',
                    ms=4.5, label=lab)
            where = np.isfinite(sd)
            if where.any():
                ax.fill_between(xx, yy - np.nan_to_num(sd), yy + np.nan_to_num(sd),
                                color='black', alpha=.11 if reg == 'interior' else .055,
                                lw=0, where=where)
        ax.axhline(0, color='.4', ls=':', lw=1)
        ax.set(xlabel='Cooling-window midpoint temperature (K)',
               ylabel=rf'$D_{{\mathrm{{TR}},{order}}}=\log_{{10}}(\tau_{{\mathrm{{rot}},{order}}}/\tau_{{\alpha,\parallel}})$')
        outputs += finish(fig, ax, out / f'{legacy}_core_vs_AWI', temperature_axis=True)
        if order == 1:
            # Also emit explicitly numbered filename for symmetry.
            # It is the same plotting data as the original V4_DTR_core_vs_AWI.
            pass

        wide = x.pivot(index=['replica', 'rate', 'temperature_K'],
                       columns='region', values=column).dropna(subset=['interior', 'awi'])
        if wide.empty:
            continue
        wide = wide.reset_index()
        wide['delta_bulk_minus_awi'] = bulk_minus_awi(wide.interior, wide.awi)
        wide.to_csv(out / f'{prefix}_paired_bulk_minus_AWI_replica_delta.csv', index=False)
        if order == 1:
            wide.to_csv(out / 'V4_DTR_paired_bulk_minus_AWI_replica_delta.csv', index=False)
        summary = []
        for T, g in wide.groupby('temperature_K'):
            mu, sd, n = mean_sd(g.delta_bulk_minus_awi)
            hw = tdist.ppf(.975, n - 1) * sd / np.sqrt(n) if n >= 2 else np.nan
            summary.append({'temperature_K': T, 'mean': mu, 'sd': sd,
                            'n_paired': n, 'ci95_low': mu - hw,
                            'ci95_high': mu + hw})
        ss = pd.DataFrame(summary).sort_values('temperature_K')
        ss.to_csv(out / f'{prefix}_paired_delta_summary.csv', index=False)
        if order == 1:
            ss.to_csv(out / 'V4_DTR_paired_delta_summary.csv', index=False)
        good = ss[ss.n_paired >= 2]
        if not good.empty:
            fig, ax = figure()
            xx = good.temperature_K.to_numpy(float)
            yy = good['mean'].to_numpy(float)
            lo = good.ci95_low.to_numpy(float)
            hi = good.ci95_high.to_numpy(float)
            ax.plot(xx, yy, 'o-', color='black', label='Bulk − AWI (matched replicas)')
            ax.fill_between(xx, lo, hi, color='black', alpha=.12, lw=0)
            ax.axhline(0, color='.4', ls=':', lw=1)
            ax.set(xlabel='Cooling-window midpoint temperature (K)',
                   ylabel=rf'$D_{{\mathrm{{TR}},{order}}}^{{\mathrm{{bulk}}}}-D_{{\mathrm{{TR}},{order}}}^{{\mathrm{{AWI}}}}$')
            outputs += finish(fig, ax, out / f'{legacy}_paired_bulk_minus_AWI',
                              temperature_axis=True)
    return outputs


def region_tau_ratio(tau,out,rate):
    """Logarithm of the bulk-to-interface relaxation-time ratio.

    Parameters
    ----------
    tau : 
        Relaxation time.
    out : 
        Output path or directory.
    rate : 
        Cooling-rate code or rate in K/ns, as stored in the table.

    Returns
    -------
    result
        Logarithm of the bulk-to-interface relaxation-time ratio.
    """
    x=clean_rate(tau,rate);x=x[x.region.isin(REGIONS)]
    x=x[x.status=='resolved'].copy()
    if x.empty:return []
    if x.duplicated(['replica','rate','temperature_K','region','metric']).any():
        raise ValueError('Nonunique tau records: refuse to form apparent pairs')
    wide=x.pivot(index=['replica','rate','temperature_K','metric'],columns='region',values='tau_ps').dropna(subset=['interior','awi'])
    if wide.empty:return []
    wide=wide.reset_index();wide['log10_bulk_over_awi']=np.log10(wide.interior/wide.awi)
    wide.to_csv(out/'V4_relaxation_region_paired.csv',index=False)
    fig,ax=figure();summ=[]
    for met,(_,mark,col) in METRICS.items():
        ss=[]
        for T,g in wide[wide.metric==met].groupby('temperature_K'):
            mu,sd,n=mean_sd(g.log10_bulk_over_awi)
            hw=tdist.ppf(.975,n-1)*sd/np.sqrt(n) if n>=2 else np.nan
            summ.append({'metric':met,'temperature_K':T,'mean':mu,'sd':sd,'n_paired':n,'ci95_halfwidth':hw})
            if n>=2:ss.append((T,mu,hw))
        if ss:
            ss=np.array(sorted(ss));x=ss[:,0];y=ss[:,1];h=ss[:,2]
            ax.plot(x,y,color=col,marker=mark,ls='-',label=met+' bulk/AWI')
            ax.fill_between(x,y-h,y+h,color=col,alpha=.11,lw=0)
    if not summ:plt.close(fig);return []
    pd.DataFrame(summ).to_csv(out/'V4_relaxation_region_contrast_stats.csv',index=False)
    ax.axhline(0,color='.4',ls=':',lw=1)
    ax.set(xlabel='Cooling-window midpoint temperature (K)',ylabel=r'$\log_{10}(\tau_{\mathrm{bulk}}/\tau_{\mathrm{AWI}})$')
    return finish(fig,ax,out/'V4_relaxation_bulk_over_AWI',temperature_axis=True)


def hb_coord_figs(coord, out, rate, temperatures):
    """Hydrogen-bond coordination histogram of each region.

    Parameters
    ----------
    coord :
        Hydrogen-bond coordination table.
    out :
        Output path or directory.
    rate :
        Physical cooling rate in K/ns.
    temperatures : sequence of float
        Temperatures to draw, in kelvin. Each is matched to the nearest
        temperature present in the table.

    Returns
    -------
    list
        Written figure paths.
    """
    x=clean_rate(coord,rate);x=x[x.region.isin(REGIONS)]
    if x.empty:return []
    # Average within replica BEFORE between-replica statistics.
    if {'n_hb','prob','temp_mid_K'}.issubset(x):
        d=x.copy()
    elif {'n_hb','temp_mid_K'}.issubset(x):
        # Raw samples, construct replica-specific normalized histograms.
        d=x.groupby(['replica','temp_mid_K','region','n_hb']).size().rename('count').reset_index()
        d['prob']=d['count']/d.groupby(['replica','temp_mid_K','region'])['count'].transform('sum')
    else:return []
    d=d.groupby(['replica','temp_mid_K','region','n_hb'],as_index=False).prob.mean()
    summ=d.groupby(['temp_mid_K','region','n_hb']).prob.agg(['mean','std','count']).reset_index()
    summ.to_csv(out/'V4_HB_coordination_replica_summary.csv',index=False)
    available=[float(t) for t in sorted(summ.temp_mid_K.unique())]
    if not available:return []
    outputs=[]
    for requested in temperatures:
        temp=min(available, key=lambda t: abs(t-requested))
        fig,ax=figure()
        for reg,(lab,ls) in REGIONS.items():
            z=summ[(np.isclose(summ.temp_mid_K, temp))&(summ.region==reg)].sort_values('n_hb')
            if z.empty:continue
            xx=z.n_hb.to_numpy(float);y=z['mean'].to_numpy(float);sd=np.nan_to_num(z['std'].to_numpy(float))
            ax.plot(xx,y,ls,color='black',marker='o' if reg=='interior' else 's',label=lab)
            ax.fill_between(xx,np.clip(y-sd,0,1),np.clip(y+sd,0,1),color='black',alpha=.10 if reg=='interior' else .055)
        title=f'{temp:g} K' if abs(temp-requested)<0.5 else f'{temp:g} K, nearest available to {requested:g} K'
        ax.set(xlabel='Hydrogen bonds per water',ylabel=r'$P(n_{\mathrm{HB}})$', title=f'{title}, AWI 3–12 Å')
        outputs+=finish(fig,ax,out/f'V4_HB_coordination_T{temp:g}K')
    return outputs


def hb_retention_figs(hb,out,rate,temps,band,selected_lag):
    """Intermittent hydrogen-bond retention and the regional difference.

    Parameters
    ----------
    hb : 
        Hydrogen-bond retention table.
    out : 
        Output path or directory.
    rate : 
        Cooling-rate code or rate in K/ns, as stored in the table.
    temps : 
        Temperatures in kelvin.
    band : 
        Half-width of the shaded band in multiples of the replica standard deviation.
    selected_lag : 
        Requested lag in picoseconds.

    Returns
    -------
    result
        Intermittent hydrogen-bond retention and the regional difference.
    """
    d=clean_rate(hb,rate);d=d[d.region.isin(REGIONS)].copy()
    if d.empty:return []
    # V3 aggregation yields one unique replica/T/lag/region observation.
    keys=['replica','temp_mid_K','region','lag_ps']
    if d.duplicated(keys).any():raise ValueError('HB intermittent file has duplicate replica/T/region/lag entries')
    summ=d.groupby(['temp_mid_K','region','lag_ps']).C_HB_intermittent.agg(['mean','std','count']).reset_index()
    summ.to_csv(out/'V4_HB_retention_replica_summary.csv',index=False)
    wanted=[t for t in temps if np.any(np.isclose(d.temp_mid_K,t))] if temps else sorted(d.temp_mid_K.unique(),reverse=True)
    colors=temperature_colors(wanted);results=[]
    if wanted:
        fig,ax=figure()
        for T in sorted(wanted,reverse=True):
            for reg,(label,ls) in REGIONS.items():
                s=summ[(summ.temp_mid_K==T)&(summ.region==reg)].sort_values('lag_ps')
                if s.empty:continue
                x=s.lag_ps.to_numpy(float);y=s['mean'].to_numpy(float);sd=np.nan_to_num(s['std'].to_numpy(float))
                ax.plot(x,y,ls,color=colors[float(T)],label=f'{T:g} K — {label}')
                if band>0:ax.fill_between(x,np.clip(y-band*sd,0,1),np.clip(y+band*sd,0,1),color=colors[float(T)],alpha=.065,lw=0)
        ax.set(xlabel='Lag time (ps)',ylabel='Intermittent original-partner H-bond retention')
        results+=finish(fig,ax,out/'V4_HB_retention_by_temperature',logx=True,ylim=(0,1.05))
    # For temperature curves select COMMON actual lag for every included replica and both regions.
    requested=float(selected_lag);rows=[]
    for (rep,T),g in d.groupby(['replica','temp_mid_K']):
        piv=g.pivot(index='lag_ps',columns='region',values='C_HB_intermittent').dropna(subset=['interior','awi'])
        if piv.empty:continue
        actual=float(piv.index[np.argmin(np.abs(piv.index.to_numpy(float)-requested))])
        if abs(actual-requested)>max(1e-8,.15*requested):continue
        rr=piv.loc[actual]
        rows.append({'replica':rep,'temperature_K':T,'lag_ps':actual,'interior':rr['interior'],'awi':rr['awi'],'delta':bulk_minus_awi(rr['interior'], rr['awi'])})
    if rows:
        p=pd.DataFrame(rows);p.to_csv(out/'V4_HB_matched_lag_replica.csv',index=False)
        fig,ax=figure()
        for reg,(label,ls) in REGIONS.items():
            z=p.groupby('temperature_K')[reg].agg(['mean','std','count']).reset_index().sort_values('temperature_K')
            x=z.temperature_K.to_numpy(float);y=z['mean'].to_numpy(float);sd=np.nan_to_num(z['std'].to_numpy(float))
            ax.plot(x,y,ls,color='black',marker='o' if reg=='interior' else 's',label=label)
            ax.fill_between(x,np.clip(y-sd,0,1),np.clip(y+sd,0,1),alpha=.11 if reg=='interior' else .055,color='black',lw=0)
        ax.set(xlabel='Cooling-window midpoint temperature (K)',ylabel=f'H-bond retention near {requested:g} ps')
        results+=finish(fig,ax,out/'V4_HB_retention_core_vs_AWI',temperature_axis=True,ylim=(0,1.05))
        dsum=p.groupby('temperature_K').delta.agg(['mean','std','count']).reset_index().sort_values('temperature_K')
        fig,ax=figure()
        ax.plot(dsum.temperature_K,dsum['mean'],'o-',color='black',label='Bulk − AWI (matched replicas)')
        sd=np.nan_to_num(dsum['std'].to_numpy(float));y=dsum['mean'].to_numpy(float)
        ax.fill_between(dsum.temperature_K,y-sd,y+sd,color='black',alpha=.11,lw=0)
        ax.axhline(0,color='.4',ls=':',lw=1)
        ax.set(xlabel='Cooling-window midpoint temperature (K)',ylabel='Bulk − AWI H-bond retention')
        results+=finish(fig,ax,out/'V4_HB_paired_retention_difference',temperature_axis=True)
    return results


def onset_figs(onset,out,rate):
    """Temperature onset of the regional retention difference at each lag.

    Parameters
    ----------
    onset : 
        Table of the temperature at which a lag first separates the regions.
    out : 
        Output path or directory.
    rate : 
        Cooling-rate code or rate in K/ns, as stored in the table.

    Returns
    -------
    result
        Temperature onset of the regional retention difference at each lag.
    """
    d=clean_rate(onset,rate)
    need={'temperature_K','lag_requested_ps','mean','sd'}
    if need-set(d):return []
    fig,ax=figure();temps=sorted(d.lag_requested_ps.unique())
    pal=plt.get_cmap('viridis')(np.linspace(.12,.85,len(temps)))
    for i,lag in enumerate(temps):
        x=d[d.lag_requested_ps==lag].sort_values('temperature_K')
        ax.plot(x.temperature_K,x['mean'],marker='o',color=pal[i],label=f'{lag:g} ps')
        sd=np.nan_to_num(x['sd'].to_numpy(float));yy=x['mean'].to_numpy(float)
        ax.fill_between(x.temperature_K,yy-sd,yy+sd,color=pal[i],alpha=.085,lw=0)
    ax.axhline(0,color='.4',ls=':',lw=1)
    ax.set(xlabel='Cooling-window midpoint temperature (K)',ylabel='Bulk − AWI H-bond retention')
    return finish(fig,ax,out/'V4_HB_onset_by_lag',temperature_axis=True)


def matched_figs(matched,out,rate):
    """Association between hydrogen-bond loss and reorientation.

    Parameters
    ----------
    matched : 
        Table of matched hydrogen-bond and reorientation events.
    out : 
        Output path or directory.
    rate : 
        Cooling-rate code or rate in K/ns, as stored in the table.

    Returns
    -------
    result
        Association between hydrogen-bond loss and reorientation.
    """
    d=clean_rate(matched,rate);d=d[d.region.isin(REGIONS)]
    if d.empty:return []
    key={'temp_mid_K','region','matched_risk_difference','replica'}
    if key-set(d):return []
    summ=d.groupby(['temp_mid_K','region']).matched_risk_difference.agg(['mean','std','count']).reset_index()
    summ.to_csv(out/'V4_matched_exchange_summary.csv',index=False)
    fig,ax=figure()
    for reg,(lab,ls) in REGIONS.items():
        z=summ[summ.region==reg].sort_values('temp_mid_K')
        if z.empty:continue
        x=z.temp_mid_K.to_numpy(float);y=z['mean'].to_numpy(float);sd=np.nan_to_num(z['std'].to_numpy(float))
        ax.plot(x,y,ls,color='black',marker='o' if reg=='interior' else 's',label=lab)
        ax.fill_between(x,y-sd,y+sd,color='black',alpha=.1 if reg=='interior' else .05,lw=0)
    ax.axhline(0,color='.4',ls=':',lw=1)
    ax.set(xlabel='Cooling-window midpoint temperature (K)',ylabel=r'$\Delta P(\mathrm{reorientation}\mid\mathrm{HB\ loss})$')
    return finish(fig,ax,out/'V4_HB_loss_reorientation_association',temperature_axis=True)


def predictive_figs(pred,out,rate):
    """Leave-one-replica log-loss improvement in each region.

    Parameters
    ----------
    pred : 
        Table of a predictive-model score.
    out : 
        Output path or directory.
    rate : 
        Cooling-rate code or rate in K/ns, as stored in the table.

    Returns
    -------
    result
        Leave-one-replica log-loss improvement in each region.
    """
    d=clean_rate(pred,rate);d=d[d.region.isin(REGIONS)]
    # Optionally visualizes original V2 leave-one-out gain or V3 nested contrasts.
    key='logloss_gain_both' if 'logloss_gain_both' in d else 'gain_logloss'
    if key not in d:return []
    n=d.groupby('region')[key].agg(['mean','std','count']).reset_index()
    n.to_csv(out/'V4_predictive_gain_summary.csv',index=False)
    fig,ax=figure()
    for reg,(lab,ls) in REGIONS.items():
        z=n[n.region==reg]
        if z.empty:continue
        x=0 if reg=='interior' else 1
        y=float(z['mean'].iloc[0]);sd=float(z['std'].iloc[0]) if len(z)>0 else np.nan
        ax.errorbar(x,y,yerr=sd if np.isfinite(sd) else None,marker='o' if reg=='interior' else 's',
                    color='black',ls=ls,ms=7,capsize=3,label=lab)
    ax.axhline(0,color='.4',ls=':',lw=1)
    ax.set(xticks=[0,1],xticklabels=['Bulk','AWI'],ylabel='Leave-one-replica-out log-loss improvement')
    return finish(fig,ax,out/'V4_predictive_logloss_gain')


def main(argv=None):
    """Write the regional dynamics figures from existing CSV tables.

    Parameters
    ----------
    argv : 
        Arguments excluding the program name. Defaults to sys.argv.

    Returns
    -------
    result
        Write the regional dynamics figures from existing CSV tables.
    """
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--corrected-dir',type=Path,help='Directory with V3 corrected tau/curves/decoupling CSVs')
    p.add_argument('--hb-dir',type=Path,help='Directory with V3 aggregated H-bond CSVs')
    p.add_argument('--onset-dir',type=Path,help='Directory containing HB_onset_temperature_lag_table.csv')
    p.add_argument('--matched-dir',type=Path,help='Directory containing matched_exchange_replica.csv')
    p.add_argument('--predictive-csv',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rate', default=4.44, type=float, help='Physical cooling rate in K/ns.')
    p.add_argument('--temps',default='235,215,195',help='Temperatures for multi-temperature lag curves')
    p.add_argument('--hb-temps',default='215,78',help='Coordination histograms, in kelvin. Each is matched to the nearest stored temperature.')
    p.add_argument('--lag-ps',type=float,default=320.)
    p.add_argument('--band-sd',type=float,default=1.,help='Shaded replica SD multiplier')
    p.add_argument('--strict',action=argparse.BooleanOptionalAction,default=True,help='Stop when a requested table is missing or has the wrong columns.')
    a=p.parse_args(argv);design();a.out.mkdir(parents=True,exist_ok=True)
    made=[];warnings=[]
    def optional(path,cols):
        """Load a CSV, or record that it is missing.

        Parameters
        ----------
        path : 
            Input file or directory.
        cols : 
            Required column names.

        Returns
        -------
        result
            Load a CSV, or record that it is missing.
        """
        if not path.is_file():warnings.append(f'Not found: {path}');return None
        return available_csv(path,cols)
    if a.corrected_dir:
        d=a.corrected_dir
        rc=optional(d/'replica_curves.csv',K+['lag_ps','Fs','C1','C2'])
        tau=optional(d/'tau_censoring_corrected.csv',K+['metric','status','tau_ps','bound_ps'])
        summ=optional(d/'tau_summary_corrected.csv',['rate','temperature_K','region','metric','n_resolved','n_replicas','log10_tau_mean','log10_ci_low','log10_ci_high'])
        pairs=optional(d/'decoupling_paired_resolved.csv',K+['Fs','C1','C2'])
        try:
            temps=[float(v) for v in a.temps.split(',') if v.strip()]
            if rc is not None:made+=plot_relaxation_curves(rc,a.out,a.rate,temps,a.band_sd)
            if tau is not None and summ is not None:
                made+=plot_tau_and_bounds(tau,summ,a.out,a.rate,a.band_sd)
                made+=region_tau_ratio(tau,a.out,a.rate)
            if pairs is not None:made+=paired_contrast(pairs,a.out,a.rate)
        except Exception as e:
            if a.strict:raise
            warnings.append(f'Corrected figures failed: {type(e).__name__}: {e}')
    if a.hb_dir:
        d=a.hb_dir
        try:
            coord=optional(d/'hb_coord_distributions.csv',['replica','rate','temp_mid_K','region','n_hb','prob'])
            hb=optional(d/'hb_intermittent_replica.csv',['replica','rate','temp_mid_K','region','lag_ps','C_HB_intermittent'])
            if coord is not None:made+=hb_coord_figs(coord,a.out,a.rate,[float(v) for v in a.hb_temps.split(',') if v.strip()])
            if hb is not None:made+=hb_retention_figs(hb,a.out,a.rate,temps if 'temps' in locals() else [235,215,195],a.band_sd,a.lag_ps)
        except Exception as e:
            if a.strict:raise
            warnings.append(f'HB figures failed: {type(e).__name__}: {e}')
    if a.onset_dir:
        pth=a.onset_dir/'HB_onset_temperature_lag_table.csv'
        d=optional(pth,['rate','temperature_K','lag_requested_ps','mean','sd'])
        if d is not None:made+=onset_figs(d,a.out,a.rate)
    if a.matched_dir:
        pth=a.matched_dir/'matched_exchange_replica.csv'
        d=optional(pth,['rate','replica','region','temp_mid_K','matched_risk_difference'])
        if d is not None:made+=matched_figs(d,a.out,a.rate)
    if a.predictive_csv:
        d=optional(a.predictive_csv,['rate','region'])
        if d is not None:made+=predictive_figs(d,a.out,a.rate)
    manifest={'rate_code':a.rate,'produced':made,'warnings':warnings,'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
      'scientific_note':'Conditional cooling-history estimates; no physical time validation or censorship correction beyond input tables performed by plotter.'}
    (a.out/'V4_figure_manifest.json').write_text(json.dumps(manifest,indent=2))
    for warning in warnings:print('WARNING:',warning,file=sys.stderr)
    print(f'Wrote {len(made)//2} figures (PNG + PDF) to {a.out}')
    if not made:raise SystemExit('No figures produced; check V4_figure_manifest.json and input directories')
    if a.strict and warnings:raise SystemExit('Strict mode: missing expected inputs')

if __name__=='__main__': main()
