from future_food.datablock_setup import datablock_setup
from future_food.pipeline_builder import pipeline_setup
from agrifoodpy.pipeline import Pipeline
import copy
import numpy as np
import sys

from agrifoodpy.food.food import FoodBalanceSheet
import matplotlib.pyplot as plt

from timer import Timer

params = {}

slider_values = {
    "ruminant" : 0,
    "pig_poultry" : 0,
    "fish_seafood" : 0,
    "dairy" : 0,
    "eggs" : 0,
    "fruit_veg" : 0,
    "pulses" : 0,
    "meat_alternatives" : 0,
    "dairy_alternatives" : 0,
    "waste" : 0,
    "foresting_pasture" : 13.17,
    "bdleaf_conif_ratio":75,
    "land_BECCS" : 0,
    "land_BECCS_pasture":0,
    "horticulture" : 0,
    "pulse_production" : 0,
    "lowland_peatland" : 0,
    "upland_peatland" : 0,
    "pasture_soil_carbon" : 0,
    "arable_soil_carbon" : 0,
    "mixed_farming" : 0,
    "silvopasture" : 0,
    "nitrogen" : 0,
    "methane_inhibitor" : 0,
    "stock_density" : 0,
    "manure_management" : 0,
    "livestock_yield":100,
    "animal_breeding" : 0,
    "fossil_livestock" : 0,
    "agroforestry" : 0,
    "vertical_farming" : 0,
    "fossil_arable" : 0,
    "waste_BECCS" : 0,
    "overseas_BECCS" : 0,
    "DACCS" : 0,
    "biochar":0
}

advanced_settings = {
    "pop_proj": "Medium",
    "yield_proj":0.0,
    "elasticity":0.5,
    "baseline_total_emissions":71,
    "baseline_agricultural_emissions":30,
    "ssr_metric":"g/cap/day",
    "baseline_beef_herd":5672659,
    "baseline_dairy_herd":3479950,
    "baseline_dairy_herd_breeding_aged_2_years_":1836442,
    "baseline_sheep_flock":31016701,
    "baseline_poultry_heads":178000000,
    "baseline_pig_heads":4715669,
    "baseline_potato_area":0.12,
    "baseline_oilseed_area":0.418,
    "baseline_horticulture_area":0.145,
    "baseline_cereal_area":3.1,
    "baseline_othercrops_area":0.75,
    "labmeat_co2e":2.2,
    "dairy_alternatives_co2e":0.31,
    "rda_kcal":2250,
    "n_scale":20,
    "bdleaf_seq_ha_yr":3.82,
    "conif_seq_ha_yr":7.63,
    "new_bdleaf_seq_ha_yr":2.1,
    "new_conif_seq_ha_yr":11.2,
    "peatland_seq_ha_yr":20,
    "managed_arable_seq_ha_yr":0.66,
    "managed_pasture_seq_ha_yr":0.66,
    "mixed_farming_seq_ha_yr":0.66,
    "beccs_crops_arable_seq_ha_yr":5.34,
    "beccs_crops_pasture_seq_ha_yr":11.82,
    "BECCS_arable_tco2_ha_yr":20.84,
    "BECCS_pasture_tco2_ha_yr":5.11,
    "dairy_herd_grazing":0.05,
    "dairy_herd_beef":0.52,
    "horticulture_land_ratio":0.086,
    "pulse_land_ratio":0.033,
    "mixed_farming_production_scale":0.93,
    "mixed_farming_secondary_production_scale":0.1,
    "agroecology_tree_coverage":0.1,
    "nitrogen_prod_factor":0,
    "nitrogen_ghg_factor":0.1,
    "manure_prod_factor":0,
    "manure_ghg_factor":0.07,
    "breeding_prod_factor":0,
    "breeding_ghg_factor":0.08,
    "methane_prod_factor":0,
    "methane_ghg_factor":0.13,
    "fossil_livestock_prod_factor":0,
    "fossil_livestock_ghg_factor":0.1,
    "fossil_arable_prod_factor":0,
    "fossil_arable_ghg_factor":0.1,
    "scaling_nutrient":"kCal/cap/day"
}

params.update(advanced_settings)
params.update(slider_values)

datablock = datablock_setup(
    AES_KEY="U19QNaXcSDjtC2h1SxfPsjCRR7bb06ufu2F571Y31so=",
    AES_IV="RTtcrRl2g/c4AQ9VxYTdeA==",
    advanced_settings=advanced_settings,
    years = [2020, 2030, 2040, 2050],
    # years = np.arange(2020, 2051),
)

times = []

if __name__ == "__main__":

    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    for i in range(runs):

        it = Timer()
        db_copy = copy.deepcopy(datablock)

        fs = Pipeline(datablock=db_copy)

        fs = pipeline_setup(
            fs,
            slider_values,
            advanced_settings,
            )

        # it.ping("Pipeline setup time: ")

        fs.run()
        it.ping(message=None)
        it.total("Total execution time: ")

        times.append(it.elapsed_time)

        print(fs.datablock["metrics"]["total_emissions"], end=" ")
        print(fs.datablock["metrics"]["SSR_metric_yr"].values)

print("Total runs: ", len(times))
print("Average time: ", sum(times)/len(times))
print("Min time: ", min(times))
print("Max time: ", max(times))
print("Standard deviation: ", np.std(times))

