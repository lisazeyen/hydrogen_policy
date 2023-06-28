#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:56:34 2023

@author: lisa
"""

import pypsa
import pandas as pd
import matplotlib.pyplot as plt

snakemake.config['tech_colors']["CI H2"] = snakemake.config['tech_colors']["offtake H"]

if 'snakemake' not in globals():
    import os
    os.chdir("/home/lisa/Documents/hourly_vs_annually/scripts")
    from _helpers import mock_snakemake
    snakemake = mock_snakemake('solve_base_network',
                            policy="ref", palette='p1', zone='DE', year='2025',
                            res_share="p0",
                            offtake_volume="3200")

def plot_balances(balances_df, policy="ref", store="", t="1H"):

    co2_carriers = ["co2", "co2 stored", "process emissions"]


    balances = {i.replace(" ","_"): [i] for i in balances_df.index.levels[0]}
    balances["energy"] = [i for i in balances_df.index.levels[0] if i not in co2_carriers]

    for k, v in balances.items():

        df = balances_df.loc[v]
        df = df.groupby(df.index.get_level_values(2)).sum()

        #convert MWh to TWh
        df = df / 1e6

        #remove trailing link ports
        df.index = [i[:-1] if ((i not in ["co2", "NH3", "H2", "CI H2"]) and (i[-1:] in ["0","1","2","3"])) else i for i in df.index]

        # df = df.groupby(df.index.map(rename_techs)).sum()
        df = df.groupby(df.index).sum()

        # df = df.droplevel([1,2], axis=1)


        to_drop = df.index[df.abs().max(axis=1) < 1] # snakemake.config['plotting']['energy_threshold']/10]


        df = df.drop(to_drop)

        # df = df.droplevel(0, axis=1)

        print(df.sum())

        if df.empty:
            continue


        fig, ax = plt.subplots(figsize=(12,8))

        df.T.plot(kind="bar",ax=ax,stacked=True, title=k,
                                                color=[snakemake.config['tech_colors'][i] for i in df.index],
                                                grid=True)

        handles,labels = ax.get_legend_handles_labels()

        handles.reverse()
        labels.reverse()

        if v[0] in co2_carriers:
            ax.set_ylabel("CO2 [MtCO2/a]")
        else:
            ax.set_ylabel("Energy [TWh/a]")

        ax.set_xlabel("")

        ax.grid(axis="x")

        ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)


        fig.savefig(f"/home/lisa/Documents/hourly_vs_annually/compare_1H_3H_withUC/balances_compare_{k}_{policy}_{store}_{t}.pdf", bbox_inches='tight')

        # generation diff ############
        generation = df[(df>0)]
        
        generation.dropna().plot(kind="bar")
        
        plt.ylabel("Energy [TWh/a]")
        
        plt.savefig(f"/home/lisa/Documents/hourly_vs_annually/compare_1H_3H_withUC/generation_compare_{k}_{policy}_{store}_{t}.pdf", bbox_inches='tight')

       
#%%

# import networks
n = pypsa.Network("/home/lisa/mnt/hourly_vs_annually/results/new_test_hourly_uc/networks/2025/DE/p1/ref_p0_3200volume_flexibledemand.nc")
m = pypsa.Network("/home/lisa/mnt/hourly_vs_annually/results/test_3hourly_uc/networks/2025/DE/p1/ref_p0_3200volume_flexibledemand.nc")
#%%
policy = "ref"
store="flexibledemand"
policy="res1p0"

csvs_n = pd.read_csv(f"/home/lisa/Documents/hourly_vs_annually/results/new_test_hourly_uc/csvs/2025/DE/p1/{policy}_p0_3200volume_{store}_supply_energy.csv",
                     index_col=[0,1,2], header=[0,1,2,3])

csvs_m =  pd.read_csv(f"/home/lisa/Documents/hourly_vs_annually/results/test_3hourly_uc/csvs/2025/DE/p1/{policy}_p0_3200volume_{store}_supply_energy.csv",
                      index_col=[0,1,2], header=[0,1,2,3])

supply_energy = pd.concat([csvs_n, csvs_m], axis=1)
supply_energy.columns = ["1H", "3H"]
plot_balances(supply_energy, policy=policy, store=store, t="")

#%%
pot_vres = n.generators_t.p_max_pu.mul(n.generators.p_nom_opt)
dispatch = n.generators_t.p
curtailed = (pot_vres - n.generators_t.p).sum().groupby(n.generators.carrier).sum()
pot_vres = m.generators_t.p_max_pu.mul(m.generators.p_nom_opt)
dispatch = m.generators_t.p
curtailed2 = (pot_vres - m.generators_t.p).sum().groupby(m.generators.carrier).sum() *3
c = pd.concat([curtailed, curtailed2], axis=1)
c.columns = ["1H", "3H"]
c = c[c.sum(axis=1)!=0]
(c/1e6).plot(kind="bar")
plt.ylabel("Curtailed energy \n [TWh]")
plt.savefig(f"/home/lisa/Documents/hourly_vs_annually/compare_1H_3H_withUC/curtailment_compare_{policy}_{store}.pdf", bbox_inches='tight')

#%%
c = pd.concat([n.generators_t.p.groupby(n.generators.carrier, axis=1).sum().sum(), m.generators_t.p.groupby(m.generators.carrier, axis=1).sum().sum()], axis=1)
c.columns = ["base", "prebase"]
(c/1e6).plot(kind="bar")
plt.ylabel("Energy \n [TWh]")
plt.savefig(f"/home/lisa/Documents/hourly_vs_annually/compare_1H_3H_withUC/generationRES_compare_ref_3H.pdf", bbox_inches='tight')

#%%
(a/1e6).plot(kind="bar", grid=True)
plt.ylabel("Energy \n [TWh]")
plt.savefig(f"/home/lisa/Documents/hourly_vs_annually/compare_1H_3H_withUC/compare_lhs_base_prebase_3H.pdf", bbox_inches='tight')
