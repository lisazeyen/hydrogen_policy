
import pypsa, pandas as pd
import numpy as np
from solve_network import (prepare_costs, palette, strip_network,
                           timescope, shutdown_lineexp, add_battery_constraints,
                           limit_resexp,set_co2_policy,
                           phase_outs, reduce_biomass_potential,
                           cost_parametrization, country_res_constraints,
                           average_every_nhours, add_unit_committment)
from resolve_network import (add_H2, add_dummies, res_constraints,
                             monthly_constraints, excess_constraints)


import logging
logger = logging.getLogger(__name__)

# Suppress logging of the slack bus choices
pypsa.pf.logger.setLevel(logging.WARNING)

from vresutils.benchmark import memory_logger
from _helpers import override_component_attrs


def solve_network(n, tech_palette):

    clean_techs, storage_techs, storage_chargers, storage_dischargers = palette(tech_palette)


    def extra_functionality(n, snapshots):

        add_battery_constraints(n)
        country_res_constraints(n, snakemake)

        if "res" in policy:
            logger.info("setting annual RES target")
            res_constraints(n, snakemake)
        if "monthly" in policy:
            logger.info("setting monthly RES target")
            monthly_constraints(n, snakemake)
        elif "exl" in policy:
            logger.info("setting excess limit on hourly matching")
            excess_constraints(n, snakemake)
            
        if snakemake.config["scenario"]["DE_target"] and "DE" in n.buses.country.unique():
            DE_targets_res(n, snakemake)


    if snakemake.config["global"]["must_run"]:
        coal_i = n.links[n.links.carrier.isin(["lignite","coal"])].index
        n.links.loc[coal_i, "p_min_pu"] = 0.9
    n.consistency_check()


    formulation = snakemake.config['solving']['options']['formulation']
    solver_options = snakemake.config['solving']['solver']
    solver_name = solver_options['name']
    solver_options["crossover"] = 0
    
    if snakemake.config["global"]["uc"]:
         add_unit_committment(n)
         
    linearized_uc = True if any(n.links.committable) else False

    
    # testing
    nhours = snakemake.config["scenario"]["temporal_resolution"]
    n = average_every_nhours(n, nhours)
        
    result, message = n.optimize(
           extra_functionality=extra_functionality,
           formulation=formulation,
           solver_name=solver_name,
           solver_options=solver_options,
           log_fn=snakemake.log.solver,
           linearized_unit_commitment=linearized_uc)
   
    if result != "ok" or message != "optimal":
        logger.info(f"solver ended with {result} and {message}, break")
        exit()
    
    return n

capacities_2025 = {
     "coal": 8e3,   # p.11 [1]
     "lignite": 14e3, # p.11 [1]
     "gas": 37e3,  # p.11 [1]
     "solar":  108e3,    # p.22 [1]
     "onwind": 77e3,  # p.22 [1]
     "offwind": 12e3,  # p.22 [1]
     }
 
# [2] https://www.bund-nrw.de/fileadmin/nrw/dokumente/braunkohle/221128_EBC_Aurora_Kohleausstiegspfad_und_Emissionen_as_sent.pdf
# [3] https://www.wirtschaft.nrw/system/files/media/document/file/eckpunktepapier-rwe-kohleausstieg_0.pdf
capacities_2030 = {
     "coal": 8e3,  # [2]
     "lignite": 6e3,  # [2, 3]
     "gas": 46e3,  # p.11 [1]
     "solar": 215e3,    # p.22 [1]
     "onwind": 115e3,  # p.22 [1]
     "offwind": 30e3 ,  # p.22 [1]
     }

def DE_targets(n, snakemake):
    
    """ Set capacities according to planned targets.
    from Agora report [1]  p.10
    [1] https://static.agora-energiewende.de/fileadmin/Projekte/2021/2021_11_DE_KNStrom2035/A-EW_264_KNStrom2035_WEB.pdf
    
    """  
    
    year = snakemake.wildcards.year    
 
    capacities = capacities_2025 if year=="2025" else capacities_2030  
    
    # conventional
    for carrier in ["lignite", "coal"]:
        links_i = n.links[(n.links.carrier==carrier)&(n.links.index.str[:2]=="DE")].index
        original_capacity = n.links.p_nom.mul(n.links.efficiency).loc[links_i].sum()
        scale_factor = capacities[carrier] / original_capacity
        scale_factor = 0 if np.isnan(scale_factor) else scale_factor
        n.links.loc[links_i, "p_nom"] *= scale_factor
        n.links.loc[links_i, "p_nom_extendable"] = False 
        logger.info(f"Scaling capacity of {carrier} by {scale_factor} from {original_capacity} to {capacities[carrier]}")
     
        
    return n


def DE_targets_res(n, snakemake, include_ci=True):
    """Add constraint for renewable capacities in Germany."""
    
    year = snakemake.wildcards.year   
    capacities = capacities_2025 if year=="2025" else capacities_2030  
    
    # renewable
    c = "Generator"
    for carrier in ["offwind", "solar", "onwind"]:
        if include_ci:
            res_bool = ((n.df(c).carrier.str.contains(carrier))
                        & ((n.df(c).index.str[:2]=="DE") | (n.df(c).index.str[:5]=="CI DE")))
        else:
            res_bool = ((n.df(c).carrier.str.contains(carrier))
                        &(n.df(c).index.str[:2]=="DE"))
        ext_i = (n.generators.p_nom_extendable & res_bool)
        fix_i = (~n.generators.p_nom_extendable & res_bool)
        gens_i = n.df(c)[ext_i].index
        planned_capacities = capacities[carrier] 
        existing = n.generators.loc[fix_i, "p_nom"].sum()
        rhs = planned_capacities - existing
        max_pot = n.generators.loc[gens_i, "p_nom_max"].sum()
        if rhs>max_pot:
            logger.info("Warning, technical potential of {carrier} is smaller than planned capacity")
        p_nom = n.model[f"{c}-p_nom"].loc[gens_i].sum("Generator-ext")
        logger.info(
        f"------------------------------------------"
        f"Add constraint for {carrier}: existing capacities {existing/1e3} GW, "
        f"planned capacities {planned_capacities/1e3} GW, remaining {rhs/1e3} GW "
        f"have to be fullfilled by {gens_i}"
        f"------------------------------------------"
        )

       
        n.model.add_constraints(
           p_nom >= rhs,
           name=f"GlobalConstraint-DE_target_{carrier}",
        )
        n.add(
           "GlobalConstraint",
           f"DE_target_{carrier}",
           constant=rhs,
           sense=">=",
           type="",
           )
    
def add_entsoe_demand(n):
    """
    Add demand forecast from ENTSO-E ERAA [1] 
    GB is not in the ENTSO-E ERAA dataset, demand time series of GB is scaled
    based on average increased electricity demand increase, this neglects
    higher demand in winter times in 2030 due to electrified heating.
    
    [1] https://www.entsoe.eu/outlooks/eraa/2022/eraa-downloads/
    """
    weather_year = 2013
    xls = pd.ExcelFile(snakemake.input.demand)
    sheet_names = xls.sheet_names
    countries = n.buses.country.unique()
    
    logger.info("Read in ENTSO-E ERAA demand data")
    data_dict = {sheet_name: pd.read_excel(xls, sheet_name=sheet_name, header=[10],
                                           index_col=[0,1])[weather_year] 
             for sheet_name in sheet_names if sheet_name[:2] in countries}
    
    # ENTSO-E ERAA demand
    demand = pd.concat(data_dict, axis=1)
    # sum demand by country   
    demand = demand.T.groupby(demand.columns.str[:2]).sum().T
    demand.index = n.snapshots
    
    # get average scaling for GB (this neglects higher demand in winter)
    load_grouped = n.loads_t.p_set.T.groupby(n.loads_t.p_set.columns.str[:2]).sum().T
    average_scale = (demand.sum()/load_grouped.sum()).mean()
    
    logger.info("overwrite PyPSA demand with ENTSO-E ERAA data")
    # create a new DataFrame with the same index as `demand` and columns as `n.loads_t.p_set`
    weighted_demand = pd.DataFrame(index=demand.index, columns=n.loads_t.p_set.columns)
    
    # calculate the weights for countries with multiple regions
    for col in n.loads_t.p_set.columns:
        country_code = col[:2]
        if country_code in demand.columns:
            # find the columns in `n.loads_t.p_set` that correspond to the same country
            related_cols = [c for c in n.loads_t.p_set.columns if c.startswith(country_code)]
            # calculate the total load for the country at each time point
            total_load = n.loads_t.p_set[related_cols].sum(axis=1)
            for related_col in related_cols:
                # assign weighted demand to the new DataFrame
                weighted_demand[related_col] = (n.loads_t.p_set[related_col] / total_load) * demand[country_code]
    # overwrite the `n.loads_t.p_set` values with the weighted demand
    n.loads_t.p_set.update(weighted_demand)
    
    # missing countries 
    missing = weighted_demand.columns[weighted_demand.isna().sum()!=0]
    if len(missing)!=0:
        logger.info(
        f"the data for the following nodes is missing: {missing},\n"
        f"scaling time-series with average scale factor of {average_scale}"
        )
        # scale missing countries with average scaling factor
        n.loads_t.p_set[missing] *= average_scale


#%%
if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('solve_network_together',
                                policy="res1p0", palette='p1',
                                zone='DE', year='2030',
                                res_share="p0",
                                offtake_volume="3200",
                                storage="flexibledemand")

    logging.basicConfig(filename=snakemake.log.python,
                    level=snakemake.config['logging_level'])

    #Wildcards & Settings ----------------------------------------------------
    policy = snakemake.wildcards.policy
    logger.info(f"solving network for policy: {policy}")

    tech_palette = snakemake.wildcards.palette
    logger.info(f"Technology palette: {tech_palette}")

    zone = snakemake.wildcards.zone
    logger.info(f"Bidding zone: {zone}")

    year = snakemake.wildcards.year
    logger.info(f"Year: {year}")

    area = snakemake.config['area']
    logger.info(f"Geoscope: {area}")

    res_share = float(snakemake.wildcards.res_share.replace("m","-").replace("p","."))
    if  snakemake.wildcards.res_share=="p0":
        res_share = timescope(zone, year, snakemake)["country_res_target"]
    logger.info(f"RES share: {res_share}")

    offtake_volume = float(snakemake.wildcards.offtake_volume)
    logger.info(f"H2 demand: {offtake_volume} MWh_h2/h")

    # import network -------------------------------------------------------
    n = pypsa.Network(timescope(zone, year, snakemake)['network_file'],
                      override_component_attrs=override_component_attrs())
    
    # adjust biomass CHP2 bus 2
    chp_i = n.links[n.links.carrier=="urban central solid biomass CHP"].index
    n.links.loc[chp_i, "bus2"] = ""
    remove_i = n.links[n.links.carrier.str.contains("biomass boiler")].index
    n.mremove("Link", remove_i)
    
    # adjust hydrogen storage in background system
    store_type = snakemake.config["scenario"]["h2_background"]["storage"]
    store_i = n.stores[n.stores.carrier=="H2 Store"].index
    store_cost = snakemake.config["global"]["H2_store_cost"][store_type][float(year)]
    logger.info(
    "Setting hydrogen storage costs in the background system"
    " to medium pressure steel tanks in all countries."
    )
    n.stores.loc[store_i, "capital_cost"] = store_cost 

    if not snakemake.config["scenario"]["h2_background"]["pipelines"]:
        remove_i = n.links[n.links.carrier=="H2 pipeline"].index
        n.mremove("Link", remove_i)

    Nyears = 1 # years in simulation
    costs = prepare_costs(timescope(zone, year, snakemake)['costs_projection'],
                          snakemake.config['costs']['USD2013_to_EUR2013'],
                          snakemake.config['costs']['discountrate'],
                          Nyears,
                          snakemake.config['costs']['lifetime'],
                          year,
                          snakemake)

    strip_network(n, zone, area, snakemake)
    add_entsoe_demand(n)
    shutdown_lineexp(n)
    limit_resexp(n,year, snakemake)
    phase_outs(n, snakemake)
    reduce_biomass_potential(n)
    cost_parametrization(n, snakemake)
    set_co2_policy(n, snakemake, costs)
    
    # add conventional power plant targets here and RES as constraint
    if snakemake.config["scenario"]["DE_target"] and "DE" in n.buses.country.unique():
        n = DE_targets(n, snakemake)

    add_H2(n, snakemake)
    add_dummies(n)
    
    with memory_logger(filename=getattr(snakemake.log, 'memory', None), interval=30.) as mem:

        n = solve_network(n, tech_palette)
        
        # save shadow prices from temporal matching
        for key in n.model.dual.keys():
            if "matching" in key:
                shadow_price = n.model.dual[key].data
                n.global_constraints.loc[key, "mu"] = shadow_price
                
        n.export_to_netcdf(snakemake.output.network)

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
