
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


def DE_targets(n, snakemake):
    
    """ Set capacities according to planned targets.
    from Agora report [1]  p.10
    [1] https://static.agora-energiewende.de/fileadmin/Projekte/2021/2021_11_DE_KNStrom2035/A-EW_264_KNStrom2035_WEB.pdf
    
    """  
    capacities_2025 = {
        "coal": 8e3,   # p.11 [1]
        "lignite": 14e3, # p.11 [1]
        "gas": 37e3,  # p.11 [1]
        "solar":  108e3,    # p.22 [1]
        "onwind": 77e3,  # p.22 [1]
        "offwind-dc": 12e3,  # p.22 [1]
        }
    
    capacities_2030 = {
        "coal": 0,   # p.11 [1]
        "lignite": 0, # p.11 [1]
        "gas": 46e3,  # p.11 [1]
        "solar":  215e3,    # p.22 [1]
        "onwind": 115e3,  # p.22 [1]
        "offwind-dc": 30e3 ,  # p.22 [1]
        }
    
    year = snakemake.wildcards.year    
    
    # scale electricity demand
    current_load = (n.loads_t.p_set
                    .mul(n.snapshot_weightings.generators, axis=0)["DE1 0"]
                    .sum())/1e6
    load_scale = {"2025":590/current_load,
                  "2030":(726-37)/current_load}
    logger.info("Increasing electricity demand by factor %.2f",
                round(load_scale[year], ndigits=2))
    n.loads_t.p_set["DE1 0"] *= load_scale[year]
    
    # scale wind
    existing_wind = n.generators[(n.generators.index.str[:2]=="DE")
                                 &(n.generators.carrier=="offwind")].p_nom.sum()
    capacities = capacities_2025 if year=="2025" else capacities_2030  
    capacities["offwind-dc"] -= existing_wind
    
    # conventional
    for carrier in ["lignite", "coal"]:
        links_i = n.links[(n.links.carrier==carrier)&(n.links.index.str[:2]=="DE")].index
        scale_factor = capacities[carrier] / n.links.p_nom.mul(n.links.efficiency).loc[links_i].sum()
        scale_factor = 0 if np.isnan(scale_factor) else scale_factor
        n.links.loc[links_i, "p_nom"] *= scale_factor
        n.links.loc[links_i, "p_nom_extendable"] = False 
        logger.info(f"Scaling capacity of {carrier} by {scale_factor}")
     
    # renewable
    c = "Generator"
    for carrier in ["offwind-dc", "solar", "onwind"]:
        gens_i = n.df(c)[(n.df(c).carrier==carrier)&(n.df(c).index.str[:2]=="DE")].index
        if carrier=="offwind-dc":
            n.df(c).loc[gens_i, "p_nom"] = capacities[carrier] 
        else:
            scale_factor = capacities[carrier] / n.df(c).p_nom.loc[gens_i].sum()
            n.df(c).loc[gens_i, "p_nom"] *= scale_factor
            print(f"Scaling capacity of {carrier} by {scale_factor}")
        
    return n

#%%
if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('solve_network_together',
                                policy="monthly", palette='p1',
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
    
    # adjust hydrogen storage in background system to medium pressure
    store_i = n.stores[n.stores.carrier=="H2 Store"].index
    store_cost = snakemake.config["global"]["H2_store_cost"]["mtank"][float(year)]
    logger.info(
    "Setting hydrogen storage costs in the background system"
    " to medium pressure steel tanks in all countries."
    )
    n.stores.loc[store_i, "capital_cost"] = store_cost 


    Nyears = 1 # years in simulation
    costs = prepare_costs(timescope(zone, year, snakemake)['costs_projection'],
                          snakemake.config['costs']['USD2013_to_EUR2013'],
                          snakemake.config['costs']['discountrate'],
                          Nyears,
                          snakemake.config['costs']['lifetime'],
                          year,
                          snakemake)

    strip_network(n, zone, area, snakemake)
    shutdown_lineexp(n)
    limit_resexp(n,year, snakemake)
    phase_outs(n, snakemake)
    reduce_biomass_potential(n)
    cost_parametrization(n, snakemake)
    set_co2_policy(n, snakemake, costs)
    
    if snakemake.config["scenario"]["DE_target"] and "DE" in n.buses.country.unique():
        n = DE_targets(n, snakemake)

    add_H2(n, snakemake)
    add_dummies(n)
    
    with memory_logger(filename=getattr(snakemake.log, 'memory', None), interval=30.) as mem:

        n = solve_network(n, tech_palette)
        
        n.export_to_netcdf(snakemake.output.network)

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
