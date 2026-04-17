import numpy as np
import xarray as xr
import warnings
from agrifoodpy.utils.scaling import logistic_scale
import copy

def get_items(
        fbs,
        items
        ):
    """Get items from food data."""
    if isinstance(items, tuple):
        items = fbs.sel(Item=np.isin(fbs[items[0]], items[1])).Item.values
    elif np.isscalar(items):
        items = [items]
    return items


def logistic_food_supply(
        fbs,
        timescale,
        c_init,
        c_end,
        years
        ):
    """Creates a logistic curve using the year range of the input food balance
    supply"""

    y0 = fbs.Year.values[0]
    y1 = 2021
    y2 = 2021 + timescale
    y3 = fbs.Year.values[-1]

    scale = logistic_scale(y0, y1, y2, y3, c_init=c_init, c_end=c_end).sel(Year=years)

    return scale


def check_negative_source(
        fbs,
        source,
        fallback=None,
        add=True
        ):
    """Checks for negative values in the source element and adds the difference
    to the fallback element"""

    if fallback is None:
        if source == "production":
            fallback = "imports"
        elif source == "imports":
            fallback = "production"
        elif source == "exports":
            fallback = "production"

    delta_neg = fbs[source].where(fbs[source] < 0, other=0)
    fbs[source] -= delta_neg

    if add:
        fbs[fallback] += delta_neg
    else:
        fbs[fallback] -= delta_neg

    return fbs


def feed_scale(
        fbs,
        ref,
        elasticity=None,
        source="production"
        ):
    """Scales the feed, seed and processing quantities according to the change
    in production of animal and vegetal products"""

    # Obtain reference production values
    ref_feed_arr = ref["production"].sel(Item=ref.Item_origin=="Animal Products").sum(dim="Item")
    ref_seed_arr = ref["production"].sel(Item=ref.Item_origin=="Vegetal Products").sum(dim="Item")

    # Compute scaling factors for feed and seed based on proportional production
    feed_scale = fbs["production"].sel(Item=fbs.Item_origin=="Animal Products").sum(dim="Item") \
                / ref_feed_arr
    seed_scale = fbs["production"].sel(Item=fbs.Item_origin=="Vegetal Products").sum(dim="Item") \
                / ref_seed_arr

    # Set feed_scale and seed_scale to 1 where ref arrays are close or equal to zero
    feed_scale = xr.where(np.isclose(ref_feed_arr, 0), 1, feed_scale)
    seed_scale = xr.where(np.isclose(ref_seed_arr, 0), 1, seed_scale)

    processing_scale = fbs["production"].sum(dim="Item") \
                / ref["production"].sum(dim="Item")

    if elasticity is not None:
        out = fbs.fbs.scale_add(element_in="feed", element_out=source,
                                scale=feed_scale, elasticity=elasticity)

        out = out.fbs.scale_add(element_in="seed",element_out=source,
                                scale=seed_scale, elasticity=elasticity)

        out = out.fbs.scale_add(element_in="processing",element_out=source,
                                scale=processing_scale, elasticity=elasticity)

    else:
        out = fbs.fbs.scale_add(element_in="feed", element_out=source,
                                scale=feed_scale)

        out = out.fbs.scale_add(element_in="seed",element_out=source,
                                scale=seed_scale)

        out = out.fbs.scale_add(element_in="processing",element_out=source,
                                scale=processing_scale)

    return out


def scale_kcal_feed(
        obs,
        ref,
        items
        ):
    """Scales the feed quantities according to the difference in production of
    specified items, on a calorie by calorie basis"""

    # Obtain reference and observed production values
    ref_prod = ref["production"].sel(Item=items).expand_dims(dim="Item").sum(dim="Item")
    obs_prod = obs["production"].sel(Item=items).expand_dims(dim="Item").sum(dim="Item")

    # Compute difference
    delta = obs_prod - ref_prod

    # Compute scaling factors for feed based on new required quantities
    ref_feed = ref["feed"].sum(dim="Item")
    obs_feed = obs["feed"].sum(dim="Item")

    # Scaling factor to be applied to old feed quantities
    feed_scale = (obs_feed + delta) / obs_feed

    # Adjust feed quantities
    out = obs.fbs.scale_add(element_in="feed",
                            element_out="production",
                            scale=feed_scale)

    return out


def item_scaling_multiple(
        datablock,
        scale,
        source,
        scaling_nutrient,
        element="food",
        elasticity=None,
        items=None,
        add=True,
        constant=True,
        non_sel_items=None
        ):
    """Reduces per capita intake quantities and replaces them by other items
    keeping the overall consumption constant. Scales land use if production
    changes

    Parameters
    ----------
    scale : float, arr, or xarray.DataArray
        Scaling factor to apply to the selected items. If a DataArray, it must
        have a "Year" dimension with the years to scale.

    items : arr, tuple
        if an array,
    """

    timescale = datablock["global_parameters"]["timescale"]
    # We can use any quantity here, either per cap/day or per year. The ratio
    # will cancel out the population growth
    food_orig = datablock["food"][scaling_nutrient]

    if np.isscalar(source):
        source = [source]

    out = food_orig.copy(deep=True)

    non_sel_items = get_items(food_orig, non_sel_items)

    # Balanced scaling. Reduce food, reduce imports, keep kCal constant
    for sc, it in zip(scale, items):

        it_arr = get_items(out, it)
        # if no items are specified, do nothing
        if items is None:
            return datablock

        out = balanced_scaling(fbs=out,
                               items=it_arr,
                               element=element,
                               timescale=timescale,
                               year=2020,
                               scale=sc,
                               adoption="logistic",
                               origin=source,
                               add=add,
                               elasticity=elasticity,
                               constant=constant,
                               non_sel_items=non_sel_items)

    # Scale feed, seed and processing
    out = feed_scale(out, food_orig)

    # out = check_negative_source(out, "production", "imports")
    out = check_negative_source(out, "imports", "exports", add=False)

    ratio = out / food_orig
    ratio = ratio.where(~np.isnan(ratio), 1)

    # Update per cap/day values and per year values using the same ratio, which
    # is independent of population growth
    datablock["food"]["g/cap/day"] *= ratio

    return datablock

def scale_production(
        datablock,
        scale_factor,
        items=None
        ):
    """ Scales the production values for the selected items by multiplying them by
    a multiplicative factor.
    """

    timescale = datablock["global_parameters"]["timescale"]

    # load quantities and impacts
    food_orig = datablock["food"]["g/cap/day"].copy(deep=True)

    # if no items are specified, do nothing
    items = get_items(food_orig, items)

    scale_prod = logistic_food_supply(food_orig, timescale, 1, scale_factor, years=food_orig.Year.values)

    out = food_orig.fbs.scale_add(element_in="production",
                                element_out="imports",
                                scale=scale_prod,
                                items=items,
                                add=False)

    # Reduce feed and seed
    out = feed_scale(out, food_orig, source = "imports")

    out = check_negative_source(out, "production", "imports")
    out = check_negative_source(out, "imports", "exports", add=False)

    ratio = out / food_orig
    ratio = ratio.where(~np.isnan(ratio), 1)

    # Update per cap/day values and per year values using the same ratio, which
    # is independent of population growth
    # qty_key = ["g/cap/day", "g_prot/cap/day", "g_fat/cap/day", "kCal/cap/day"]
    # for key in qty_key:
    #     datablock["food"][key] *= ratio

    datablock["food"]["g/cap/day" ] *= ratio

    return datablock

def balanced_scaling(
        fbs,
        items,
        scale,
        element,
        year=None,
        adoption=None,
        timescale=10,
        origin=None,
        add=True,
        elasticity=None,
        constant=False,
        non_sel_items=None,
        fallback=None,
        add_fallback=True
        ):
    """Scale items quantities across multiple elements in a FoodBalanceSheet
    Dataset

    Scales selected item quantities on a food balance sheet and with the
    posibility to keep the sum of selected elements constant.
    Optionally, produce an Dataset with a sequence of quantities over the years
    following a smooth scaling according to the selected functional form.

    The elements used to supply the modified quantities can be selected to keep
    a balanced food balance sheet.

    Parameters
    ----------
    fbs : xarray.Dataset
        Input food balance sheet Dataset.
    items : list
        List of items to scale in the food balance sheet.
    element : string
        Name of the DataArray to scale.
    scale : float
        Scaling parameter after full adoption.
    adoption : string, optional
        Shape of the scaling adoption curve. "logistic" uses a logistic model
        for a slow-fast-slow adoption. "linear" uses a constant slope adoption
        during the the "timescale period"
    year : int, optional
        Year of the Food Balance Sheet to use as pivot. If not set, the last
        year of the array is used
    timescale : int, optional
        Timescale for the scaling to be applied completely.  If "year" +
        "timescale" is greater than the last year in the array, it is extended
        to accomodate the extra years.
    origin : string, optional
        Name of the DataArray which will be used to balance the food balance
        sheets. Any change to the "element" DataArray will be reflected in this
        DataArray.
    add : bool, optional
        If set to True, the scaled element difference is added to the "origin"
        DataArray. If False, it is subtracted.
    elasticity : float, float array_like optional
        Fractional percentage of the difference that is added to each
        element in origin.
    constant : bool, optional
        If set to True, the sum of element remains constant by scaling the non
        selected items accordingly.
    non_sel_items : list, optional
        List of items to scale to achieve constant quantity sum when constant
        is set to True.
    fallback : string, optional
        Name of the DataArray used to provide the excess required to balance
        the food balance sheet in case the "origin" falls below zero.
    add_fallback : bool, optional
        If set to True, the excessis added to the fallback DataArray. If False,
        it is subtracted.

    Returns
    -------
    data : xarray.Dataarray
        Food balance sheet Dataset with scaled "food" values.
    """

    # Check for single item inputs
    if np.isscalar(items):
        items = [items]

    if np.isscalar(origin):
        origin = [origin]

    if np.isscalar(add):
        add = [add]*len(origin)

    # Check for single item list fbs
    input_item_list = fbs.Item.values
    if np.isscalar(input_item_list):
        input_item_list = [input_item_list]
        if constant:
            warnings.warn("Constant set to true but input only has a single item.")
            constant = False

    # If no items are provided, we scale all of them.
    if items is None or np.sort(items) is np.sort(input_item_list):
        items = fbs.Item.values
        if constant:
            warnings.warn("Cannot keep food constant when scaling all items.")
            constant = False

    # Define Dataarray to use as pivot
    if "Year" in fbs.dims:
        if year is None:
            if np.isscalar(fbs.Year.values):
                year = fbs.Year.values
                fbs_toscale = fbs
            else:
                year = fbs.Year.values[-1]
                fbs_toscale = fbs.isel(Year=-1)
        else:
            fbs_toscale = fbs.sel(Year=year)

    else:
        fbs_toscale = fbs
        try:
            year = fbs.Year.values
        except AttributeError:
            year = 0

    # Define scale array based on year range
    if adoption is not None:
        if adoption == "linear":
            from agrifoodpy.utils.scaling import linear_scale as scale_func
        elif adoption == "logistic":
            from agrifoodpy.utils.scaling import logistic_scale as scale_func
        else:
            raise ValueError("Adoption must be one of 'linear' or 'logistic'")

        y0 = fbs.Year.values[0]
        y1 = year
        y2 = np.min([year + timescale, fbs.Year.values[-1]])
        y3 = fbs.Year.values[-1]

        scale_arr = scale_func(y0, y1, y2, y3, c_init=1, c_end=scale)

        # # Extend the dataset to include all the years of the array
        # fbs_toscale = fbs_toscale * xr.ones_like(scale_arr)

    else:
        scale_arr = scale

    # Modify and return
    out = fbs.fbs.scale_add(
        element,
        origin,
        scale_arr,
        items,
        add=add,
        elasticity=elasticity)

    if constant:

        delta = out[element] - fbs[element]

        # Scale non selected items
        if non_sel_items is None:
            non_sel_items = np.setdiff1d(fbs.Item.values, items)

        non_sel_scale = (fbs.sel(Item=non_sel_items)[element].sum(dim="Item") - delta.sum(dim="Item")) / fbs.sel(Item=non_sel_items)[element].sum(dim="Item")

        # Make sure inf and nan values are not scaled
        non_sel_scale = non_sel_scale.where(np.isfinite(non_sel_scale)).fillna(1.0)

        if np.any(non_sel_scale < 0):
            warnings.warn("Additional consumption cannot be compensated by \
                        reduction of non-selected items")

        out = out.fbs.scale_add(
            element,
            origin,
            non_sel_scale,
            non_sel_items,
            add=add,
            elasticity=elasticity
            )

        # If fallback is defined, adjust to prevent negative values
        if fallback is not None:
            df = sum(out[org].where(out[org] < 0).fillna(0) for org in origin)
            out[fallback] -= np.where(add_fallback, -1, 1)*df
            for org in origin:
                out[org] = out[org].where(out[org] > 0, 0)

    return out

def food_waste_model(
        datablock,
        waste_scale,
        kcal_rda,
        source,
        elasticity=None
        ):
    """Reduces daily per capita per day intake energy above a set threshold.
    """

    timescale = datablock["global_parameters"]["timescale"]
    kcal_fact = datablock["food"]["kCal/g_food"]
    food_orig = copy.deepcopy(datablock["food"]["g/cap/day"])*kcal_fact
    datablock["food"]["rda_kcal"] = kcal_rda

    # This is the maximum factor we can multiply food by to achieve consumption
    # equal to rda_kcal, multiplied by the ambition level
    waste_factor = (food_orig["food"].isel(Year=-1).sum(dim="Item") - kcal_rda) \
                 / food_orig["food"].isel(Year=-1).sum(dim="Item") \
                 * (waste_scale / 100)

    waste_factor = waste_factor.to_numpy()

    # Create a logistic curve starting at 1, ending at 1-waste_factor
    scale_waste = logistic_food_supply(food_orig, timescale, 1, 1-waste_factor, years=food_orig.Year.values)

    # Set to "imports" or "production" to choose which element of the food system supplies the change in consumption
    # Scale food and subtract difference from production
    out = food_orig.fbs.scale_add(element_in="food",
                                  element_out=source,
                                  scale=scale_waste,
                                  elasticity=elasticity)

    # Scale feed, seed and processing
    out = feed_scale(out, food_orig)

    # If supply element is negative, set to zero and add the negative delta to imports
    out = check_negative_source(out, "imports", "exports", add=False)

    # Scale all per capita qantities proportionally
    ratio = out / food_orig
    ratio = ratio.where(~np.isnan(ratio), 1)

    datablock["food"]["g/cap/day"] *= ratio

    return datablock

def alternative_food_model(
        datablock,
        cultured_scale,
        labmeat_co2e,
        baseline_items,
        copy_from,
        new_items,
        new_item_name,
        replaced_items,
        source,
        elasticity=None
        ):
    """Replaces selected items by alternative products on a weight by weight
    basis, compared to a baseline array.
    A list of replaced items is adjusted to keep calories constant.
    """

    timescale = datablock["global_parameters"]["timescale"]
    baseline_items = get_items(datablock["food"]["g/cap/day"], baseline_items)
    items_to_replace = get_items(datablock["food"]["g/cap/day"], replaced_items)

    nutrition_keys = ["g_prot/g_food", "g_fat/g_food", "kCal/g_food"]
    # Add new items to the food dataset
    datablock["food"]["g/cap/day"] = datablock["food"]["g/cap/day"].fbs.add_items(new_items)
    datablock["food"]["g/cap/day"]["Item_name"].loc[{"Item":new_items}] = new_item_name
    datablock["food"]["g/cap/day"]["Item_origin"].loc[{"Item":new_items}] = "Alternative Food"
    datablock["food"]["g/cap/day"]["Item_group"].loc[{"Item":new_items}] = "Alternative Food"
    # Set values to zero to avoid issues
    datablock["food"]["g/cap/day"].loc[{"Item":new_items}] = 0

    # Add nutrition values for new products to the food dataset
    for key in nutrition_keys:
        datablock["food"][key] = datablock["food"][key].fbs.add_items(new_items, copy_from=[copy_from])
        datablock["food"][key]["Item_name"].loc[{"Item":new_items}] = new_item_name
        datablock["food"][key]["Item_origin"].loc[{"Item":new_items}] = "Alternative Food"
        datablock["food"][key]["Item_group"].loc[{"Item":new_items}] = "Alternative Food"

    # Scale products by cultured_scale
    food_orig = copy.deepcopy(datablock["food"]["g/cap/day"])
    kcal_fact = datablock["food"]["kCal/g_food"]
    kcal_orig = food_orig * kcal_fact
    food_base = copy.deepcopy(datablock["food"]["baseline_projected"])

    scale_alternative = logistic_food_supply(food_orig, timescale, 0, cultured_scale, years=food_orig.Year.values)

    # This is the new alternative food consumption
    delta_alternative = (food_base["food"].sel(Item=baseline_items) * scale_alternative).sum(dim="Item")

    out = food_orig.copy(deep=True)
    out["food"].loc[{"Item":new_items}] += delta_alternative

    # If no item elasticity is provided, divide elasticity equally
    if elasticity is None:
        elasticity = [1.0/len(source)] * len(source)
    elif np.isscalar(elasticity):
        elasticity = [elasticity] * len(source)

    # Adjust source elements based on elasticity
    for src, elst in zip(source, elasticity):
        out[src].loc[{"Item":new_items}] += delta_alternative*elst

    # Reduce cereals to compensate additional kCal from alternative food
    delta_kcal_alternative = delta_alternative * kcal_fact.sel(Item=new_items)
    orig_target_calories = kcal_orig["food"].sel(Item=items_to_replace).sum(dim="Item")
    final_target_calories = kcal_orig["food"].sel(Item=items_to_replace).sum(dim="Item") - delta_kcal_alternative
    scale_target_calories = final_target_calories / orig_target_calories

    # out["food"].loc[{"Item":items_to_replace}] *= scale_target_calories
    out = out.fbs.scale_add(element_in="food",
                            element_out=source,
                            items=items_to_replace,
                            scale=scale_target_calories,
                            elasticity=elasticity)

    kcal_cap_day = kcal_orig.fbs.scale_add(element_in="food",
                            element_out=source,
                            items=items_to_replace,
                            scale=scale_target_calories,
                            elasticity=elasticity)

    # Check negative source elements
    out = check_negative_source(out, "production")
    out = check_negative_source(out, "imports", "exports", add=False)

    # Adjust feed and seed from animal production
    out = feed_scale(out, food_orig)
    datablock["food"]["g/cap/day"] = out

    # Add emissions factor for cultured meat
    datablock["impact"]["gco2e/gfood"] = datablock["impact"]["gco2e/gfood"].fbs.add_items(new_items)
    datablock["impact"]["gco2e/gfood"].loc[{"Item":new_items}] = labmeat_co2e

    out_kcal_cap_day = scale_kcal_feed(kcal_cap_day, kcal_orig, new_items)
    ratio = out_kcal_cap_day / kcal_cap_day
    ratio = ratio.where(~np.isnan(ratio), 1)

    datablock["food"]["g/cap/day"] *= ratio

    return datablock

def cultured_meat_model(
        datablock,
        cultured_scale,
        labmeat_co2e,
        items,
        copy_from,
        new_items,
        new_item_name,
        source,
        elasticity=None
        ):
    """Replaces selected items by cultured products on a weight by weight
    basis.
    """

    timescale = datablock["global_parameters"]["timescale"]
    items_to_replace = items

    # Add cultured meat to the dataset
    qty_key = ["g/cap/day", "g_prot/cap/day", "g_fat/cap/day", "kCal/cap/day"]

    for key in qty_key:
        datablock["food"][key] = datablock["food"][key].fbs.add_items(new_items)
        datablock["food"][key]["Item_name"].loc[{"Item":new_items}] = new_item_name
        datablock["food"][key]["Item_origin"].loc[{"Item":new_items}] = "Alternative Food"
        datablock["food"][key]["Item_group"].loc[{"Item":new_items}] = "Alternative Food"
        # Set values to zero to avoid issues
        datablock["food"][key].loc[{"Item":new_items}] = 0

    # Scale products by cultured_scale
    food_orig = copy.deepcopy(datablock["food"]["g/cap/day"])
    kcal_orig = copy.deepcopy(datablock["food"]["kCal/cap/day"])

    scale_labmeat = logistic_food_supply(food_orig, timescale, 1, 1-cultured_scale, years=food_orig.Year.values)

    # Scale and remove from suplying element
    out = food_orig.fbs.scale_add(element_in="food",
                                  element_out=source,
                                  scale=scale_labmeat,
                                  items=items_to_replace,
                                  add=True,
                                  elasticity=elasticity)

    # Add delta to cultured meat
    delta = (datablock["food"]["g/cap/day"]-out).sel(Item=items_to_replace).sum(dim="Item")
    out.loc[{"Item":new_items}] += delta

    # If production is negative, set to zero and add the negative delta to
    # imports
    out = check_negative_source(out, "production")
    out = check_negative_source(out, "imports", "exports", add=False)

    # Reduce feed and seed
    out = feed_scale(out, food_orig)

    datablock["food"]["g/cap/day"] = out

    # Add nutrition values for cultured meat
    nutrition_keys = ["g_prot/g_food", "g_fat/g_food", "kCal/g_food"]
    for key in nutrition_keys:
        datablock["food"][key] = datablock["food"][key].fbs.add_items(new_items, copy_from=[copy_from])
        datablock["food"][key]["Item_name"].loc[{"Item":new_items}] = new_item_name
        datablock["food"][key]["Item_origin"].loc[{"Item":new_items}] = "Alternative Food"
        datablock["food"][key]["Item_group"].loc[{"Item":new_items}] = "Alternative Food"

    # Add emissions factor for cultured meat
    datablock["impact"]["gco2e/gfood"] = datablock["impact"]["gco2e/gfood"].fbs.add_items(new_items)
    datablock["impact"]["gco2e/gfood"].loc[{"Item":new_items}] = labmeat_co2e

    # Recompute per capita values
    for key_pc, key_n in zip(qty_key[1:], nutrition_keys):
        datablock["food"][key_pc] = datablock["food"]["g/cap/day"] * datablock["food"][key_n]

    kcal_cap_day = datablock["food"]["kCal/cap/day"]

    out_kcal_cap_day = scale_kcal_feed(kcal_cap_day, kcal_orig, new_items)
    ratio = out_kcal_cap_day / kcal_cap_day
    ratio = ratio.where(~np.isnan(ratio), 1)

    for key in qty_key:
        datablock["food"][key] *= ratio

    return datablock

def extra_urban_farming(
        datablock,
        fraction,
        items
        ):
    """Increases production of certain items relative to the baseline production
    while keeping land utilization constant"""

    # Read datasets
    food_orig = datablock["food"]["g/cap/day"].copy(deep=True)
    food_base = datablock["food"]["baseline_projected"]

    # Read item lists
    items = get_items(food_orig, items)

    timescale = datablock["global_parameters"]["timescale"]

    # Create scaling array
    scale = logistic_food_supply(food_orig, timescale, 0, fraction, years=food_orig.Year.values)

    # Compute quantity of products now being produced in urban/vertical farms
    delta = food_base["production"].sel(Item=items) * scale
    delta = delta.fillna(0)

    # Add delta to productions and remove from imports
    food_orig["production"].loc[{"Item":items}] = food_orig["production"].loc[{"Item":items}] + delta
    food_orig["imports"].loc[{"Item":items}] = food_orig["imports"].loc[{"Item":items}] - delta

    # Check for negative sources and correct
    out = check_negative_source(food_orig, "imports", "exports", add=False)

    # Rewrite food data to datablock and return
    datablock["food"]["g/cap/day"] = out

    return datablock

def shift_production(
        datablock,
        scale,
        items,
        items_target,
        land_area_ratio
        ):

    """Scales production of selected items while adjusting target item list
    production according to the specified scaling factors to account for
    different yield rates.

    Parameters
    ----------
    datablock : dict
        The datablock dictionary, containing all the model parameters and
        datasets.
    scale : float
        The scale factor to be applied to the items being shifted.
    items : list
        The items to be shifted.
    items_target : list
        The items to which the production will be shifted.
    total_to_items : float
        Ratio of total productive land to land used for target items.
    target_to_items : float
        Ratio of total productive land of target items to productive land of
        items being scaled.
    """

    # Load food data from datablock
    food_orig = datablock["food"]["g/cap/day"].copy(deep=True)
    timescale = datablock["global_parameters"]["timescale"]

    items = get_items(food_orig, items)
    items_target = get_items(food_orig, items_target)

    # Create scaling array
    scale_items = logistic_food_supply(food_orig, timescale, 1, 1 + scale, years=food_orig.Year.values)

    scale_target = 1 - land_area_ratio * scale
    scale_target = logistic_food_supply(food_orig, timescale, 1, scale_target, years=food_orig.Year.values)

    # Scale production quantities

    out = food_orig.fbs.scale_add(element_in="production",
                                element_out="imports",
                                scale=scale_items,
                                items=items,
                                add=False)

    out = out.fbs.scale_add(element_in="production",
                            element_out="imports",
                            scale=scale_target,
                            items=items_target,
                            add=False)

    # Check for negative sources and correct
    out = check_negative_source(out, "imports", "exports", add=False)

    # Rewrite food data to datablock and return
    datablock["food"]["g/cap/day"] = out

    return datablock