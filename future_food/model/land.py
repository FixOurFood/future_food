import numpy as np
import xarray as xr
from .food import get_items, check_negative_source, logistic_food_supply

def forest_land_model_new(
        datablock,
        forest_fraction,
        bdleaf_conif_ratio
        ):

    """Replaces arable and livestock land with forest land.
    If positive, spare_fraction only replaces pasture land and changes it to forest land.
    If negative, spare_fraction only replaces forest land and changes it to a mix of
    pasture land and arable land, which depends on the original land distribution.
    """

    timescale = datablock["global_parameters"]["timescale"]
    binning = datablock["land"]["area_per_pixel"]
    pctg = datablock["land"]["percentage_land_use"].copy(deep=True)

    old_use_arable = datablock["land"]["percentage_land_use"].sel({"aggregate_class":["Arable"]}).sum(dim=["x", "y", "aggregate_class"])

    total_uk_land = pctg.sum(dim=["x", "y", "aggregate_class"])

    # Fraction of forest to achieve area delta
    forest_xy = datablock["land"]["percentage_land_use"].sel({"aggregate_class":["Broadleaf woodland", "Coniferous woodland"]})
    total_forest = forest_xy.sum(dim=["x", "y", "aggregate_class"])

    forest_fraction_time = logistic_food_supply(pctg, timescale, .1317, forest_fraction, years=pctg.Year.values)

    # Required delta to forest = requested fraction - current fraction
    # delta_forest_land_percentage = forest_fraction - float(total_forest / total_uk_land)
    delta_forest_land_percentage = forest_fraction_time - (total_forest / total_uk_land)

    # Total area in hectares to be converted
    delta_forest_area = total_uk_land * delta_forest_land_percentage

    pasture_xy = datablock["land"]["percentage_land_use"].sel({"aggregate_class":["Improved grassland", "Semi-natural grassland"]})
    old_use_pasture = pasture_xy.sum(dim=["x", "y", "aggregate_class"])


    if delta_forest_land_percentage.isel(Year=-1) > 0: #OJO ACA
        # We only change pasture to forest

        # Only replace pasture
        delta_pasture_ratio = delta_forest_area / old_use_pasture
        delta_pasture_xy = pasture_xy * delta_pasture_ratio

        delta_forest_xy = delta_pasture_xy.sum(dim="aggregate_class") * forest_xy / forest_xy.sum(dim="aggregate_class")

        pctg.loc[{"aggregate_class":["Improved grassland", "Semi-natural grassland"]}] -= delta_pasture_xy.fillna(0)

        # Check if sum across aggregate_class equals 100 and adjust Broadleaf woodland if needed
        sum_across_classes = pctg.sum(dim="aggregate_class")
        difference = 100*binning - sum_across_classes
        pctg.loc[{"aggregate_class": "Broadleaf woodland"}] += difference.where(~np.isnan(pctg.sel(aggregate_class="Broadleaf woodland")), 0) * bdleaf_conif_ratio
        pctg.loc[{"aggregate_class": "Coniferous woodland"}] += difference.where(~np.isnan(pctg.sel(aggregate_class="Coniferous woodland")), 0) * (1 - bdleaf_conif_ratio)

    else:
        # We change forest to a mix of arable and forest
        agricultural_xy = datablock["land"]["percentage_land_use"].sel({"aggregate_class":["Improved grassland", "Semi-natural grassland", "Arable"]})

        # Per pixel percentage delta
        delta_forest_ratio = delta_forest_area / total_forest
        delta_forest_xy = forest_xy * delta_forest_ratio
        delta_agriculture_xy = delta_forest_xy.sum(dim="aggregate_class") * agricultural_xy / agricultural_xy.sum(dim="aggregate_class")

        pctg.loc[{"aggregate_class":["Broadleaf woodland", "Coniferous woodland"]}] += delta_forest_xy
        pctg.loc[{"aggregate_class":["Improved grassland", "Semi-natural grassland", "Arable"]}] -= delta_agriculture_xy

    # Add spared class to the land use map
    datablock["land"]["percentage_land_use"] = pctg

    # Scale food production and imports
    new_use_pasture = pctg.sel({"aggregate_class":["Improved grassland", "Semi-natural grassland"]}).sum(dim=["x", "y", "aggregate_class"])
    new_use_arable = pctg.sel({"aggregate_class":["Arable"]}).sum(dim=["x", "y", "aggregate_class"])

    scale_use_pasture = (new_use_pasture/old_use_pasture)
    scale_use_arable = (new_use_arable/old_use_arable)

    food_orig = datablock["food"]["g/cap/day"]
    # scale_forest_pasture = logistic_food_supply(food_orig, timescale, 1, scale_use_pasture, years=food_orig.Year.values)
    # scale_forest_arable = logistic_food_supply(food_orig, timescale, 1, scale_use_arable, years=food_orig.Year.values)

    scaled_items_pasture = food_orig.sel(Item=food_orig.Item_origin=="Animal Products").Item.values
    scaled_items_arable = food_orig.sel(Item=food_orig.Item_origin=="Vegetal Products").Item.values

    out = food_orig.fbs.scale_add(element_in="production",
                                  element_out="imports",
                                  scale=scale_use_pasture,
                                  items=scaled_items_pasture,
                                  add=False)

    out = out.fbs.scale_add(element_in="production",
                                  element_out="imports",
                                  scale=scale_use_arable,
                                  items=scaled_items_arable,
                                  add=False)

    out = check_negative_source(out, "production")
    out = check_negative_source(out, "imports")

    ratio = out / food_orig
    ratio = ratio.where(~np.isnan(ratio), 1)

    datablock["food"]["g/cap/day"] = out

    return datablock

def forest_land_model(
        datablock,
        forest_fraction,
        bdleaf_conif_ratio,
        map_mask=None,
        mask_vals=None
        ):
    """Replaces arable and livestock land with forest land.
    If positive, spare_fraction only replaces pasature land and changes it to forest land.
    If negative, spare_fraction only replaces forest land and changes it to a mix of
    pasture land and arable land, which depends on the original land distribution.
    """

    timescale = datablock["global_parameters"]["timescale"]
    pctg = datablock["land"]["percentage_land_use"].copy(deep=True)
    old_use_pasture = datablock["land"]["percentage_land_use"].sel({"aggregate_class":["Improved grassland", "Semi-natural grassland"]}).sum()
    old_use_arable = datablock["land"]["percentage_land_use"].sel({"aggregate_class":["Arable"]}).sum()

    # if no alc grade is provided, then use the whole map
    if mask_vals is not None or map_mask is not None:
        alc = datablock["land"][map_mask]
        alc_mask = np.isin(alc, mask_vals)
    else:
        alc_mask = np.ones_like(pctg, dtype=bool)

    total_uk_land = pctg.sum()

    total_forestable_pasture_land = pctg.where(alc_mask, other=0).sel({"aggregate_class":["Improved grassland", "Semi-natural grassland"]}).sum()
    total_forestable_arable_land = pctg.where(alc_mask, other=0).sel({"aggregate_class":["Arable"]}).sum()

    pasture_to_agricultural = total_forestable_pasture_land / (total_forestable_arable_land + total_forestable_pasture_land)

    forestable_pasture_ratio = total_forestable_pasture_land / total_uk_land
    forestable_arable_ratio = total_forestable_arable_land / total_uk_land

    to_forest_pasture = pctg.where(alc_mask, other=0).sel({"aggregate_class":["Improved grassland", "Semi-natural grassland"]})
    to_forest_arable = pctg.where(alc_mask, other=0).sel({"aggregate_class":"Arable"})

    # Spare the specified land type
    if forest_fraction >= 0:
        delta_forest_pasture = to_forest_pasture * forest_fraction / forestable_pasture_ratio

    else:
        delta_forest_pasture = to_forest_pasture * forest_fraction / forestable_pasture_ratio * pasture_to_agricultural
        delta_forest_arable = to_forest_arable * forest_fraction / forestable_arable_ratio * (1-pasture_to_agricultural)
        pctg.loc[{"aggregate_class":"Arable"}] -= delta_forest_arable
        pctg.loc[{"aggregate_class":"Broadleaf woodland"}] += delta_forest_arable*bdleaf_conif_ratio
        pctg.loc[{"aggregate_class":"Coniferous woodland"}] += delta_forest_arable*(1-bdleaf_conif_ratio)

    pctg.loc[{"aggregate_class":["Improved grassland", "Semi-natural grassland"]}] -= delta_forest_pasture
    pctg.loc[{"aggregate_class":"Broadleaf woodland"}] += delta_forest_pasture.sum(dim="aggregate_class")*bdleaf_conif_ratio
    pctg.loc[{"aggregate_class":"Coniferous woodland"}] += delta_forest_pasture.sum(dim="aggregate_class")*(1-bdleaf_conif_ratio)

    # Add spared class to the land use map
    datablock["land"]["percentage_land_use"] = pctg

    # Scale food production and imports
    new_use_pasture = pctg.sel({"aggregate_class":["Improved grassland", "Semi-natural grassland"]}).sum()
    new_use_arable = pctg.sel({"aggregate_class":"Arable"}).sum()

    scale_use_pasture = (new_use_pasture/old_use_pasture).to_numpy()
    scale_use_arable = (new_use_arable/old_use_arable).to_numpy()

    food_orig = datablock["food"]["g/cap/day"]
    scale_forest_pasture = logistic_food_supply(food_orig, timescale, 1, scale_use_pasture, years=food_orig.Year.values)
    scale_forest_arable = logistic_food_supply(food_orig, timescale, 1, scale_use_arable, years=food_orig.Year.values)

    scaled_items_pasture = food_orig.sel(Item=food_orig.Item_origin=="Animal Products").Item.values
    scaled_items_arable = food_orig.sel(Item=food_orig.Item_origin=="Vegetal Products").Item.values

    out = food_orig.fbs.scale_add(element_in="production",
                                  element_out="imports",
                                  scale=scale_forest_pasture,
                                  items=scaled_items_pasture,
                                  add=False)

    out = out.fbs.scale_add(element_in="production",
                                  element_out="imports",
                                  scale=scale_forest_arable,
                                  items=scaled_items_arable,
                                  add=False)

    out = check_negative_source(out, "production")
    out = check_negative_source(out, "imports")

    ratio = out / food_orig
    ratio = ratio.where(~np.isnan(ratio), 1)

    # Update per cap/day values and per year values using the same ratio, which
    # is independent of population growth
    qty_key = ["g/cap/day", "g_prot/cap/day", "g_fat/cap/day", "kCal/cap/day"]
    for key in qty_key:
        datablock["food"][key] *= ratio

    # datablock["food"]["g/cap/day"] = out

    return datablock

def peatland_restoration(
        datablock,
        restore_fraction,
        new_land_type,
        old_land_type,
        items,
        peat_map_key=None,
        mask_val=None
        ):
    """Replaces a specified land type fraction and sets it to a new type called
    'peatland'. Scales food production and imports to reflect the change in land
    use.
    """

    timescale = datablock["global_parameters"]["timescale"]
    pctg = datablock["land"]["percentage_land_use"].copy(deep=True)
    old_use = datablock["land"]["percentage_land_use"].sel({"aggregate_class":old_land_type}).sum(dim=["x", "y", "aggregate_class"])

    if peat_map_key is not None:
        peat_map_da = datablock["land"][peat_map_key]

        if mask_val is not None:
            peat_mask = np.isin(peat_map_da, mask_val)

    # if no mask is provided, then use the whole map
    else:
        peat_mask = np.ones_like(pctg, dtype=bool)

    to_spare = pctg.where(peat_mask, other=0).sel({"aggregate_class":old_land_type})

    restore_fraction_time = logistic_food_supply(pctg, timescale, 0, restore_fraction, years=pctg.Year.values)

    # Spare the specified land type
    delta_spared =  to_spare * restore_fraction_time
    pctg.loc[{"aggregate_class":old_land_type}] -= delta_spared

    if new_land_type not in pctg.aggregate_class.values:
        spared_new_class = xr.zeros_like(pctg.isel(aggregate_class=0)).where(np.isfinite(pctg.isel(aggregate_class=0)))
        spared_new_class["aggregate_class"] = new_land_type
        pctg = xr.concat([pctg, spared_new_class], dim="aggregate_class")

    pctg.loc[{"aggregate_class":new_land_type}] += delta_spared.sum(dim="aggregate_class")

    # Add spared class to the land use map
    datablock["land"]["percentage_land_use"] = pctg

    # Scale food production and imports
    new_use = pctg.sel({"aggregate_class":old_land_type}).sum(dim=["x", "y", "aggregate_class"])
    scale_use = (new_use/old_use).fillna(1)

    food_orig = datablock["food"]["g/cap/day"]
    # scale_spare = logistic_food_supply(food_orig, timescale, 1, scale_use, years=food_orig.Year.values)

    scaled_items = food_orig.sel(Item=food_orig.Item_origin==items).Item.values

    out = food_orig.fbs.scale_add(element_in="production",
                                  element_out="imports",
                                  scale=scale_use,
                                  items=scaled_items,
                                  add=False)

    ratio = out / food_orig
    ratio = ratio.where(~np.isnan(ratio), 1)
    datablock["food"]["g/cap/day"] = out

    return datablock


def BECCS_farm_land(
        datablock,
        farm_percentage,
        items,
        land_type="Arable",
        new_land_type="BECCS",
        mask_map=None,
        mask_values=None
        ):
    """Repurposes farm land for BECCS, reducing the amount of food production,
    and increasing the amount of CO2e sequestered.
    """

    timescale = datablock["global_parameters"]["timescale"]
    pctg = datablock["land"]["percentage_land_use"].copy(deep=True)
    old_use = datablock["land"]["percentage_land_use"].sel({"aggregate_class":land_type}).sum(dim=["x", "y", "aggregate_class"])

    if mask_map is not None:
        mask_map = datablock["land"][mask_map].copy(deep=True)

    # if no alc grade is provided, then use the whole map
        if mask_values is not None:
            peat_mask = np.isin(mask_map, mask_values)

    else:
        peat_mask = np.ones_like(pctg, dtype=bool)

    to_spare = pctg.where(peat_mask, other=0).sel({"aggregate_class":land_type})

    farm_percentage_time = logistic_food_supply(pctg, timescale, 0, farm_percentage, years=pctg.Year.values)

    # Spare the specified land type
    delta_spared =  to_spare * farm_percentage_time
    pctg.loc[{"aggregate_class":land_type}] -= delta_spared

    if new_land_type not in pctg.aggregate_class.values:
        spared_new_class = xr.zeros_like(pctg.isel(aggregate_class=0)).where(np.isfinite(pctg.isel(aggregate_class=0)))
        spared_new_class["aggregate_class"] = new_land_type
        pctg = xr.concat([pctg, spared_new_class], dim="aggregate_class")

    if "aggregate_class" in delta_spared.dims:
        pctg.loc[{"aggregate_class":new_land_type}] += delta_spared.sum(dim="aggregate_class")
    else:
        pctg.loc[{"aggregate_class":new_land_type}] += delta_spared

    # Add spared class to the land use map
    datablock["land"]["percentage_land_use"] = pctg

    # Scale food production and imports
    new_use = pctg.sel({"aggregate_class":land_type}).sum(dim=["x", "y", "aggregate_class"])
    scale_use = (new_use/old_use).fillna(1)

    food_orig = datablock["food"]["g/cap/day"]
    # scale_spare = logistic_food_supply(food_orig, timescale, 1, scale_use, years=food_orig.Year.values)

    # scaled_items = food_orig.sel(Item=food_orig.Item_origin=="Vegetal Products").Item.values
    scaled_items = get_items(food_orig, items)

    out = food_orig.fbs.scale_add(element_in="production",
                                  element_out="imports",
                                  scale=scale_use,
                                  items=scaled_items,
                                  add=False)

    ratio = out / food_orig
    ratio = ratio.where(~np.isnan(ratio), 1)
    datablock["food"]["g/cap/day"] = out

    return datablock


def agroecology_model(
        datablock,
        land_percentage,
        land_type,
        agroecology_class="Agroecology",
        tree_coverage=0.1,
        replaced_items=None,
        new_items=None,
        item_yield=None,
        seq_ha_yr=6.26
        ):
    """Changes traditional agricultural land use to agroecological land use.

    Parameters
    ----------
    datablock : dict
        The datablock dictionary, containing all the model parameters and
        datasets.
    land_type : list
        The type or types of land that will be converted to agroecology.
    land_percentage : list
        The percentage or percentages of land that will be converted to
        agroecology.
    tree_coverage : float
        The percentage of each land class that will be converted to trees. This
        also sets the production value of the land class, via a 1-tree_coverage
        factor.
    replaced_items : list
        The items that will be replaced by agroecological products.
    new_items : list
        The additional items that will be grown in agroecological land.
    item_yield : float
        The yield of the additional agroecological products in g/ha/day.
    seq_ha_yr : float
        CO2e sequestration of agroecological land in t CO2e/ha/year.

    Returns
    -------
    datablock : dict
        The updated datablock dictionary, containing all the model parameters
        and datasets.
    """

    # Load land use and food data from datablock
    pctg = datablock["land"]["percentage_land_use"].copy(deep=True)
    food_orig = datablock["food"]["g/cap/day"].copy(deep=True)
    old_use = pctg.sel({"aggregate_class":land_type}).sum()
    timescale = datablock["global_parameters"]["timescale"]

    # Compute land percentages to be converted to agroecology and remove them
    # from the land_type classes
    delta_agroecology = pctg.loc[{"aggregate_class":land_type}] * land_percentage
    pctg.loc[{"aggregate_class":land_type}] -= delta_agroecology

    # Add the agroecology percentage to the new agroecology class
    if agroecology_class not in pctg.aggregate_class.values:
        new_class = xr.zeros_like(pctg.isel(aggregate_class=0)).where(np.isfinite(pctg.isel(aggregate_class=0)))
        new_class["aggregate_class"] = agroecology_class
        pctg = xr.concat([pctg, new_class], dim="aggregate_class")

    delta_total = delta_agroecology.sum(dim="aggregate_class")
    pctg.loc[{"aggregate_class":agroecology_class}] += delta_total

    out = food_orig.copy(deep=True)

    # Reduce production of replaced items if they are provided
    if replaced_items is not None:
        new_use = pctg.sel({"aggregate_class":land_type}).sum()
        scale_use = (new_use/old_use) + (1-tree_coverage) * (1-new_use/old_use)
        scale_use = scale_use.to_numpy()

        scale_arr = logistic_food_supply(out, timescale, 1, scale_use, years=food_orig.Year.values)

        out = out.fbs.scale_add(element_in="production",
                                element_out="imports",
                                scale=scale_arr,
                                items=replaced_items,
                                add=False)

        out = check_negative_source(out, "production", "imports")
        out = check_negative_source(out, "imports", "production")

    # Add new items by scaling production from current values to future values
    if new_items is not None:
        if np.isscalar(new_items):
            new_items = [new_items]
        if np.isscalar(item_yield):
            item_yield = [item_yield]

        pop = datablock["population"]["population"].isel(Year=-1, Region=0)

        for item, yld in zip(new_items, item_yield):
            old_production = food_orig["production"].sel({"Item":item}).isel(Year=-1)
            new_production = old_production + yld * delta_agroecology.sum()/pop
            production_scale = (new_production / old_production).to_numpy()
            production_scale_array = logistic_food_supply(food_orig, timescale, 1, production_scale, years=food_orig.Year.values)

            out = out.fbs.scale_add(element_in="production",
                                element_out="imports",
                                scale=production_scale_array,
                                items=item,
                                add=False)

    # Compute forest area in ha, maximum anual sequestration, and growth curve
    area_agroecology = pctg.loc[{"aggregate_class":agroecology_class}].sum().to_numpy()
    max_seq_agroecology = area_agroecology * seq_ha_yr

    agroecology_seq = logistic_food_supply(food_orig, timescale, 1, c_end=max_seq_agroecology, years=food_orig.Year.values)

    # Create a dataset with the different sequestration sources
    seq_ds = xr.Dataset({agroecology_class: agroecology_seq})

    seq_da = seq_ds.to_array(dim="Item", name="sequestration")
    if "co2e_sequestration" not in datablock["impact"]:
        datablock["impact"]["co2e_sequestration"] = seq_da

    else:
        # append sequestration to existing sequestration da
        seq_da_in = datablock["impact"]["co2e_sequestration"]
        seq_da = xr.concat([seq_da_in, seq_da], dim="Item")
        datablock["impact"]["co2e_sequestration"] = seq_da

    # Rewrite land use data to datablock
    datablock["land"]["percentage_land_use"] = pctg

    ratio = out / food_orig
    ratio = ratio.where(~np.isnan(ratio), 1)

    # Update per cap/day values and per year values using the same ratio, which
    # is independent of population growth

    datablock["food"]["g/cap/day" ] *= ratio

    return datablock

def production_land_scale(
        datablock,
        bdleaf_conif_ratio
        ):
    """Scales land based on the relative production change of livestock and
    arable crops"""

    land = datablock["land"]["percentage_land_use"].copy(deep=True)
    obs = datablock["food"]["g/cap/day"].copy(deep=True)
    ref = datablock["food"]["baseline_projected"].copy(deep=True)

    # Obtain reference and observed production values
    ref_livest = ref["production"].sel(Item=ref.Item_origin=="Animal Products").sum(dim="Item")
    ref_arable = ref["production"].sel(Item=ref.Item_origin=="Vegetal Products").sum(dim="Item")

    obs_livest = obs["production"].sel(Item=obs.Item_origin=="Animal Products").sum(dim="Item")
    obs_arable = obs["production"].sel(Item=obs.Item_origin=="Vegetal Products").sum(dim="Item")

    # Compute ratios
    livest_ratio = obs_livest / ref_livest
    arable_ratio = obs_arable / ref_arable

    # Scale land use types
    delta_pasture = land.loc[{"aggregate_class":["Improved grassland", "Semi-natural grassland"]}] * (1-livest_ratio)
    land.loc[{"aggregate_class":["Improved grassland", "Semi-natural grassland"]}] -= delta_pasture

    delta_arable = land.loc[{"aggregate_class":"Arable"}] * (1-arable_ratio)
    land.loc[{"aggregate_class":"Arable"}] -= delta_arable

    # Remaining or excess land is allocated to or from forest
    # Calculate total percentage
    total = land.sum(dim="aggregate_class")

    # Check if total differs from 100
    delta = 100 - total
    delta = delta.where(np.isfinite(land.isel(aggregate_class=0)))

    # Adjust Broadleaf woodland to maintain 100% total
    if "Broadleaf woodland" in land.aggregate_class:
        land.loc[{"aggregate_class":"Broadleaf woodland"}] += delta*bdleaf_conif_ratio
    # If Broadleaf woodland doesn't exist, create it
    else:
        land.loc[{"aggregate_class":"Broadleaf woodland"}] = delta*bdleaf_conif_ratio

    # Adjust Coniforus woodland to maintain 100% total
    if "Coniferous woodland" in land.aggregate_class:
        land.loc[{"aggregate_class":"Coniferous woodland"}] += delta*(1-bdleaf_conif_ratio)
    # If Coniferous woodland doesn't exist, create it
    else:
        land.loc[{"aggregate_class":"Coniferous woodland"}] = delta*(1-bdleaf_conif_ratio)

    datablock["land"]["percentage_land_use"] = land

    return datablock

def managed_agricultural_land_carbon_model(
        datablock,
        fraction,
        managed_class,
        old_class
        ):
    """Replaces a fraction of "arable" and "pasture" land types with "managed
    arable" and "managed pasture" respectively.
    """

    timescale = datablock["global_parameters"]["timescale"]

    if np.isscalar(managed_class):
        managed_class = [managed_class]

    if np.isscalar(old_class):
        old_class = [old_class]

    # Load land use data from datablock
    pctg = datablock["land"]["percentage_land_use"].copy(deep=True)

    # Create new category for "managed arable" land
    for new_class_name in managed_class:
        if new_class_name not in pctg.aggregate_class.values:
            _new_class = xr.zeros_like(pctg.isel(aggregate_class=0)).where(np.isfinite(pctg.isel(aggregate_class=0)))
            _new_class["aggregate_class"] = new_class_name
            pctg = xr.concat([pctg, _new_class], dim="aggregate_class")

    # # Compute arable fraction to be managed and remove from the arable
    # delta_arable = pctg.loc[{"aggregate_class":"Arable"}] * fraction
    # pctg.loc[{"aggregate_class":"Arable"}] -= delta_arable
    # pctg.loc[{"aggregate_class":"Managed arable"}] += delta_arable

    # Compute pasture fraction to be managed and remove from the pasture classes

    fraction_time = logistic_food_supply(pctg, timescale, 0, fraction, years=pctg.Year.values)

    delta_arable = pctg.loc[{"aggregate_class":old_class}] * fraction_time
    pctg.loc[{"aggregate_class":old_class}] -= delta_arable
    pctg.loc[{"aggregate_class":managed_class}] += delta_arable.sum(dim="aggregate_class")

    # Rewrite land use data to datablock
    datablock["land"]["percentage_land_use"] = pctg
    return datablock

def zero_land_farming_model(
        datablock,
        fraction,
        items,
        land_type="Arable",
        bdleaf_conif_ratio=0.5
        ):
    """Reduces arable land proportional to the fraction of produced food
    assumed to be farmed in vertical / urban farms. Production remains constant.
    This ignores any item passed which is not a Vegetal Product.
    """

    food_orig = datablock["food"]["g/cap/day"].copy(deep=True)

    items = get_items(food_orig, items)

    timescale = datablock["global_parameters"]["timescale"]

    # Load land use data from datablock
    pctg = datablock["land"]["percentage_land_use"].copy(deep=True)

    # Load production data from datablock
    plant_items = food_orig.sel(Item=food_orig.Item_origin=="Vegetal Products").Item.values

    # Filter items to only include plant items
    items = [item for item in items if item in plant_items]

    # Create scaling array
    scale = logistic_food_supply(food_orig, timescale, 1, fraction, years=food_orig.Year.values)

    # Compute ratio of plant products now being produced in urban/vertical farms
    food_to_shift = food_orig["production"].sel(Item=items).sum(dim="Item") * scale

    shift_ratio_da =  food_to_shift / food_orig["production"].sel(Item=plant_items).sum(dim="Item")
    shift_ratio = shift_ratio_da.isel(Year=-1).values

    # Compute delta land use
    delta_arable = pctg.loc[{"aggregate_class":land_type}] * shift_ratio
    pctg.loc[{"aggregate_class":land_type}] -= delta_arable
    # Rewrite land use data to datablock

    # Add forested percentage to the land use map
    pctg.loc[{"aggregate_class":"Broadleaf woodland"}] += delta_arable * bdleaf_conif_ratio
    pctg.loc[{"aggregate_class":"Coniferous woodland"}] += delta_arable * (1-bdleaf_conif_ratio)

    datablock["land"]["percentage_land_use"] = pctg

    return datablock

def mixed_farming_model(
        datablock,
        fraction,
        prod_scale_factor,
        items,
        secondary_items,
        secondary_prod_scale_factor,
        land_type=["Arable",
                   "Managed arable"],
        secondary_land_type=["Improved grassland",
                             "Semi-natural grassland",
                             "Managed pasture"],
        new_land_type="Mixed farming"
        ):

    """Converts arable land to mixed farming.

    In mix farms, primary crops have a small decrease in production of specified
    items, set by primary_items and primary_production_scale.
    Secondary items are increased by a specified amount, set by secondary_items,
    secondary_production_scale and the current land used primarily for the secondary
    items.
    """

    # Load land use data from datablock
    old_land = datablock["land"]["percentage_land_use"]
    pctg = datablock["land"]["percentage_land_use"].copy(deep=True)
    food_orig = datablock["food"]["g/cap/day"].copy(deep=True)
    timescale = datablock["global_parameters"]["timescale"]

    converted_fraction_time = logistic_food_supply(pctg, timescale, 0, fraction, years=pctg.Year.values)

    # Create new category for "mixed farming" land
    if new_land_type not in pctg.aggregate_class.values:
        _new_class = xr.zeros_like(pctg.isel(aggregate_class=0)).where(np.isfinite(pctg.isel(aggregate_class=0)))
        _new_class["aggregate_class"] = new_land_type
        pctg = xr.concat([pctg, _new_class], dim="aggregate_class")

    # Compute arable fraction to be converted to mixed farming
    delta_arable = pctg.loc[{"aggregate_class":land_type}] * converted_fraction_time
    pctg.loc[{"aggregate_class":land_type}] -= delta_arable
    pctg.loc[{"aggregate_class":new_land_type}] += delta_arable.sum(dim="aggregate_class")

    # Compute relative change in arable land
    mixed_farm_frac = delta_arable.sum(dim=["x", "y", "aggregate_class"]) / old_land.loc[{"aggregate_class":land_type}].sum(dim=["x", "y", "aggregate_class"])
    arable_scale = 1 - mixed_farm_frac + mixed_farm_frac * prod_scale_factor

    # Get items
    items = get_items(food_orig, items)
    secondary_items = get_items(food_orig, secondary_items)

    out = food_orig.fbs.scale_add(element_in="production",
                                  element_out="imports",
                                  scale=arable_scale,
                                  items=items,
                                  add=False)

    # Compute relative change in secondary items
    # Get relative new area of mixed farming to secondary producing area
    total_area_secondary = pctg.loc[{"aggregate_class":secondary_land_type}].sum(dim=["x", "y", "aggregate_class"])
    mixed_farm_to_secondary_ratio = delta_arable.sum(dim=["x", "y", "aggregate_class"]) / total_area_secondary
    secondary_ratio = 1 + mixed_farm_to_secondary_ratio * secondary_prod_scale_factor

    out = out.fbs.scale_add(element_in="production",
                                  element_out="exports",
                                  scale=secondary_ratio,
                                  items=secondary_items,
                                  add=True)


    # Update land use data to datablock
    datablock["land"]["percentage_land_use"] = pctg

    # Rewrite food data datablock
    datablock["food"]["g/cap/day"] = out

    return datablock

def label_new_forest(
        datablock
        ):

    land = datablock["land"]["percentage_land_use"].copy(deep=True)
    land_baseline = datablock["land"]["baseline"].copy(deep=True)

    if "New Broadleaf woodland" not in land.aggregate_class.values:
        new_class = xr.zeros_like(land.isel(aggregate_class=0)).where(np.isfinite(land.isel(aggregate_class=0)))
        new_class["aggregate_class"] = "New Broadleaf woodland"
        land = xr.concat([land.isel(aggregate_class=slice(0, 2)), new_class, land.isel(aggregate_class=slice(2, None))], dim="aggregate_class")

    if "New Coniferous woodland" not in land.aggregate_class.values:
        new_class = xr.zeros_like(land.isel(aggregate_class=0)).where(np.isfinite(land.isel(aggregate_class=0)))
        new_class["aggregate_class"] = "New Coniferous woodland"
        land = xr.concat([land.isel(aggregate_class=slice(0, 3)), new_class, land.isel(aggregate_class=slice(3, None))], dim="aggregate_class")

    for w_type in ["Broadleaf woodland", "Coniferous woodland"]:
        # Compute the difference between current and baseline woodland
        delta_w = land.sel(aggregate_class=w_type) - land_baseline.sel(aggregate_class=w_type)

        # Identify where the difference is positive (indicating new woodland)
        new_w_mask = delta_w > 0

        # Assign the positive difference to "New Broadleaf woodland"
        land.loc[{"aggregate_class": "New "+w_type}] += delta_w.where(new_w_mask, 0)

        # Limit "Broadleaf woodland" to the baseline model
        land.loc[{"aggregate_class": w_type}] = land_baseline.sel(aggregate_class=w_type).where(new_w_mask, land.sel(aggregate_class=w_type))

    datablock["land"]["percentage_land_use"] = land

    return datablock