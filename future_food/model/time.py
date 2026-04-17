import copy
import numpy as np
import xarray as xr
from agrifoodpy.utils.scaling import linear_scale
from agrifoodpy.food.food import FoodBalanceSheet

def project_future(
        datablock,
        years,
        yield_change=None,
        ):
    """Project future food consumption based on scale

    Parameters
    ----------
    datablock : Dict
        Dictionary containing xarray datasets for population, food consumption,
        etc.

    scale : xarray.DataArray
        Scale to apply to food consumption
    yield_change : float
        Percentage change by the end of the projection period.

    Returns
    -------
    datablock : Dict
        New dictionary containinng projected food consumption data
    """

    pop = datablock["population"]["population"]

    scale = pop.sel(Region=826, Year=years) \
        / pop.sel(Region=826, Year=2020)

    # Per capita per day values remain constant
    g_cap_day = datablock["food"]["g/cap/day"]
    g_prot_cap_day = datablock["food"]["g_prot/cap/day"]
    g_fat_cap_day = datablock["food"]["g_fat/cap/day"]
    kcal_cap_day = datablock["food"]["kCal/cap/day"]

    years_past = g_cap_day.Year.values

    g_cap_day = g_cap_day.fbs.add_years(years, "constant")
    g_prot_cap_day = g_prot_cap_day.fbs.add_years(years, "constant")
    g_fat_cap_day = g_fat_cap_day.fbs.add_years(years, "constant")
    kcal_cap_day = kcal_cap_day.fbs.add_years(years, "constant")

    # Scale food production
    scale_past = xr.DataArray(
        np.ones(len(years_past)),
        dims=["Year"],
        coords={"Year": years_past}
        )
    scale_tot = xr.concat([scale_past, scale], dim="Year")

    cereal_items = g_cap_day.sel(Item=g_cap_day.Item_group == "Cereals - Excluding Beer").Item.values

    # If yield_change is not None, add a scaling factor to account for yield increase, only to vegetal items
    if yield_change is not None:
        scale_tot = scale_tot.expand_dims({"Item": g_cap_day.Item.values})
        scale_yield = xr.ones_like(scale_tot)
        scale_yield.loc[{"Item": cereal_items}] = linear_scale(2020, 2020, 2050, 2050, c_init=1, c_end=1+yield_change).sel(Year=scale_tot.Year)
        scale_tot = scale_tot / scale_yield

    # Scale food production and balance using imports
    g_cap_day = g_cap_day.fbs.scale_add(element_in="production", element_out="imports", scale=1/scale_tot, add=False)
    g_prot_cap_day = g_prot_cap_day.fbs.scale_add(element_in="production", element_out="imports", scale=1/scale_tot, add=False)
    g_fat_cap_day = g_fat_cap_day.fbs.scale_add(element_in="production", element_out="imports", scale=1/scale_tot, add=False)
    kcal_cap_day = kcal_cap_day.fbs.scale_add(element_in="production", element_out="imports", scale=1/scale_tot, add=False)

    # Do the same with exports, but this time add the change in exports to imports
    g_cap_day = g_cap_day.fbs.scale_add(element_in="exports", element_out="imports", scale=1/scale_tot)
    g_prot_cap_day = g_prot_cap_day.fbs.scale_add(element_in="exports", element_out="imports", scale=1/scale_tot)
    g_fat_cap_day = g_fat_cap_day.fbs.scale_add(element_in="exports", element_out="imports", scale=1/scale_tot)
    kcal_cap_day = kcal_cap_day.fbs.scale_add(element_in="exports", element_out="imports", scale=1/scale_tot)

    # Emissions per gram of food also remain constant
    g_co2e_g = datablock["impact"]["gco2e/gfood"]
    g_co2e_g_land = datablock["impact"]["gco2e/gfood_land"]
    g_co2e_g = g_co2e_g.fbs.add_years(years, "constant")
    g_co2e_g_land = g_co2e_g_land.fbs.add_years(years, "constant")

    datablock["food"]["g/cap/day"] = g_cap_day
    datablock["food"]["g_prot/cap/day"] = g_prot_cap_day
    datablock["food"]["g_fat/cap/day"] = g_fat_cap_day
    datablock["food"]["kCal/cap/day"] = kcal_cap_day
    datablock["impact"]["gco2e/gfood"] = g_co2e_g
    datablock["impact"]["gco2e/gfood_land"] = g_co2e_g_land
    datablock["impact"]["baseline"] = copy.deepcopy(datablock["impact"]["gco2e/gfood"])

    datablock["food"]["baseline_projected"] = copy.deepcopy(datablock["food"]["g/cap/day"])

    return datablock
