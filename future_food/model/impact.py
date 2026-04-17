import numpy as np
import xarray as xr
from .food import get_items, logistic_food_supply

def compute_emissions(datablock):
    """
    Computes the emissions per capita per day and per year for each food item,
    using the per capita daily weights and PN18 emissions factors.
    """
    pop = datablock["population"]["population"]
    pop_world = pop.sel(Region = 826)

    # Compute emissions per capita per day
    co2e_cap_day_ag = datablock["food"]["g/cap/day"] * datablock["impact"]["gco2e/gfood"]
    co2e_cap_day_land = datablock["food"]["g/cap/day"] * datablock["impact"]["gco2e/gfood_land"]

    # Compute emissions per year
    datablock["food"]["g_co2e/cap/day"] = co2e_cap_day_ag
    datablock["food"]["g_co2e/cap/day_land"] = co2e_cap_day_land

    datablock["impact"]["g_co2e/year"] = co2e_cap_day_ag * pop_world * 365.25
    datablock["impact"]["g_co2e/year_land"] = co2e_cap_day_land * pop_world * 365.25

    return datablock

def ccs_model(
        datablock,
        waste_BECCS,
        overseas_BECCS,
        DACCS,
        biochar
        ):
    """Computes the CCS sequestration from the different sources

    Parameters
    ----------

    waste_BECCS : float
        Total maximum sequestration (in t CO2e / year) from food waste-origin BECCS
    overseas_BECCS : float
        Total maximum sequestration (in t CO2e / year) from overseas biomass BECCS
    DACCS : float
        Total maximum sequestration (in t CO2e / year) from DACCS
    biochar : float
        Total maximum sequestration (in t CO2e / year) from biochar and enhanced weathering
    """

    timescale = datablock["global_parameters"]["timescale"]
    food_orig = datablock["food"]["g/cap/day"]
    pctg = datablock["land"]["percentage_land_use"]
    area_per_pixel = datablock["land"]["area_per_pixel"]

    # Compute the total area of BECCS land used in hectares, and the total
    # sequestration in Mt CO2e / year

    pasture_BECCS_area = pctg.sel({"aggregate_class":"Bioenergy crops (pasture)"}).sum(dim=["x", "y"]).to_numpy() * area_per_pixel
    arable_BECCS_area = pctg.sel({"aggregate_class":"Bioenergy crops (arable)"}).sum(dim=["x", "y"]).to_numpy() * area_per_pixel
    land_BECCS = pasture_BECCS_area * datablock["advanced_settings"]["BECCS_pasture_tco2_ha_yr"]
    land_BECCS += arable_BECCS_area * datablock["advanced_settings"]["BECCS_arable_tco2_ha_yr"]

    logistic_0_val = logistic_food_supply(food_orig, timescale, 0, 1, years=food_orig.Year.values)

    waste_BECCS_seq_array = waste_BECCS * logistic_0_val
    overseas_BECCS_seq_array = overseas_BECCS * logistic_0_val
    DACCS_seq_array = DACCS * logistic_0_val
    biochar_seq_array = biochar * logistic_0_val
    land_BECCS_seq_array = land_BECCS * logistic_0_val

    # Create a dataset with the different sequestration sources
    seq_ds = xr.Dataset({"BECCS from waste": waste_BECCS_seq_array,
                         "BECCS from overseas biomass": overseas_BECCS_seq_array,
                         "BECCS from land": land_BECCS_seq_array,
                         "DACCS": DACCS_seq_array,
                         "Biochar": biochar_seq_array})

    seq_da = seq_ds.to_array(dim="Item", name="sequestration")

    if "co2e_sequestration" not in datablock["impact"]:
        datablock["impact"]["co2e_sequestration"] = seq_da
    else:
        # append sequestration to existing sequestration da
        seq_da_in = datablock["impact"]["co2e_sequestration"]
        seq_da = xr.concat([seq_da_in, seq_da], dim="Item")
        datablock["impact"]["co2e_sequestration"] = seq_da

    return datablock

def forest_sequestration_model(
        datablock,
        land_type,
        seq
        ):
    """Computes total annual sequestration from the different sources"""

    area_per_pixel = datablock["land"]["area_per_pixel"]

    if np.isscalar(land_type):
        land_type = [land_type]

    if np.isscalar(seq):
        seq = [seq]

    timescale = datablock["global_parameters"]["timescale"]
    food_orig = datablock["food"]["g/cap/day"]

    # Load the land use data from the datablock
    pctg = datablock["land"]["percentage_land_use"].copy(deep=True)
    logistic_0_val = logistic_food_supply(food_orig, timescale, 0, 1, years=food_orig.Year.values)

    for land_type_i, seq_i in zip(land_type, seq):

        # Compute forest area in ha, maximum anual sequestration, and growth curve
        area_land = pctg.loc[{"aggregate_class":land_type_i}].sum(dim=["x", "y"]) * area_per_pixel

        land_type_seq = area_land * seq_i

        # Create a dataset with the different sequestration sources
        seq_ds = xr.Dataset({land_type_i: land_type_seq})
        seq_da = seq_ds.to_array(dim="Item", name="sequestration")

        if "co2e_sequestration" not in datablock["impact"]:
            datablock["impact"]["co2e_sequestration"] = seq_da
        else:
            # append sequestration to existing sequestration da
            seq_da_in = datablock["impact"]["co2e_sequestration"]
            seq_da = xr.concat([seq_da_in, seq_da], dim="Item")
            datablock["impact"]["co2e_sequestration"] = seq_da

    # Compute agroecology sequestration

    return datablock

def scale_impact(
        datablock,
        scale_factor,
        items=None
        ):
    """ Scales the impact values for the selected items relative to the
    the baseline impact factors.
    """

    timescale = datablock["global_parameters"]["timescale"]
    # load quantities and impacts
    food_orig = datablock["food"]["g/cap/day"]
    impacts = datablock["impact"]["gco2e/gfood"].copy(deep=True)
    impacts_baseline = datablock["impact"]["baseline"].copy(deep=True)

    items = get_items(food_orig, items)

    # scale the impacts using the baseline values as reference
    scale = logistic_food_supply(food_orig, timescale, 0, scale_factor, years=food_orig.Year.values)
    delta = impacts_baseline.loc[{"Item": items}] * scale
    impacts.loc[{"Item": items}] = impacts.loc[{"Item": items}] - delta
    datablock["impact"]["gco2e/gfood"] = impacts

    return datablock