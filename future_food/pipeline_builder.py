from .model import *

def pipeline_setup(
        pipeline,
        params,
        adv_settings,
        ):
    """
    pipeline builder 

    This function takes the Pipeline object, "food_system", which already
    contains the pre-built datablock and params, which is the dictionary of
    advanced settings and slider positions from the calculator.

    It returns the updated Pipeline object with the functions to be called and
    parameters to be used in the function calling
    """

    # Store run parameters in the datablock for later use
    pipeline.datablock["run_params"] = params

    # Global parameters
    pipeline.datablock_write(["global_parameters", "timescale"], adv_settings["n_scale"])

    # Consumer demand
    pipeline.add_node(project_future,
                            {"yield_change":adv_settings["yield_proj"]})
    
    pipeline.add_node(item_scaling_multiple,
                         {"scale":[1+params["ruminant"]/100,
                                   1+params["pig_poultry"]/100,
                                   1+params["fish_seafood"]/100,
                                   1+params["dairy"]/100,
                                   1+params["eggs"]/100,
                                   1+params["fruit_veg"]/100,
                                   1+params["pulses"]/100],

                          "items":[[2731, 2732],
                                   [2733, 2734],
                                   ("Item_group", "Fish, Seafood"),
                                   [2740, 2743, 2948],
                                   [2949],
                                   ("Item_group", ["Vegetables", "Fruits - Excluding Wine"]),
                                   ("Item_group", ["Pulses"])],

                          "source":["production", "imports"],
                          "elasticity":[adv_settings["elasticity"], 1-adv_settings["elasticity"]],
                          "scaling_nutrient":adv_settings["scaling_nutrient"],
                          "constant":True,
                          "non_sel_items":("Item_group", "Cereals - Excluding Beer")})
    
    pipeline.add_node(alternative_food_model,
                         {"cultured_scale":params["meat_alternatives"]/100,
                         "labmeat_co2e":adv_settings["labmeat_co2e"],
                         "baseline_items":[2731, 2732, 2733, 2734],
                         "replaced_items":("Item_group", "Cereals - Excluding Beer"),
                         "copy_from":2731,
                         "new_items":5000,
                         "new_item_name":"Alternative meat",
                         "source":["production", "imports"],
                         "elasticity":[adv_settings["elasticity"], 1-adv_settings["elasticity"]]})

    pipeline.add_node(alternative_food_model,
                            {"cultured_scale":params["dairy_alternatives"]/100,
                            "labmeat_co2e":adv_settings["dairy_alternatives_co2e"],
                            "baseline_items":[2948, 2743, 2740],
                            "replaced_items":("Item_group", "Cereals - Excluding Beer"),
                            "copy_from":2948,
                            "new_items":5001,
                            "new_item_name":"Alternative dairy",
                            "source":["production", "imports"],
                            "elasticity":[adv_settings["elasticity"], 1-adv_settings["elasticity"]]})
 
    pipeline.add_node(food_waste_model,
                            {"waste_scale":params["waste"],
                            "kcal_rda":adv_settings["rda_kcal"],
                            "source":["production", "imports"],
                            "elasticity":[adv_settings["elasticity"], 1-adv_settings["elasticity"]]})

    pipeline.add_node(production_land_scale,
                         {"bdleaf_conif_ratio":params["bdleaf_conif_ratio"]/100,}
                        )

    # Land management
    pipeline.add_node(forest_land_model_new,
                            {"forest_fraction":params["foresting_pasture"]/100,
                            "bdleaf_conif_ratio":params["bdleaf_conif_ratio"]/100,
                            })

    pipeline.add_node(BECCS_farm_land,
                        {"land_type": "Arable",
                         "farm_percentage":params["land_BECCS"]/100,
                         "items":("Item_origin", "Vegetal Products"),
                         "new_land_type":"Bioenergy crops (arable)",
                        })
    
    pipeline.add_node(BECCS_farm_land,
                        {"land_type": ["Improved grassland", "Semi-natural grassland"],
                         "farm_percentage":params["land_BECCS_pasture"]/100,
                         "new_land_type":"Bioenergy crops (pasture)",
                         "items":("Item_origin", "Animal Products"),
                        })

    pipeline.add_node(shift_production,
                         {"scale":params["horticulture"]/100,
                          
                          "items":[2617, 2775, 2615, 2532, 2614, 2560,
                                   2619, 2625, 2613, 2620, 2612, 2551,
                                   2563, 2602, 2611, 2640, 2641, 2618,
                                   2616, 2531, 2534, 2533, 2601, 2605,
                                   2535],

                          "items_target":[2659, 2513, 2546, 2656, 2658, 2657,
                                          2520, 2642, 2633, 2578, 2630, 2559,
                                          2575, 2572, 2745, 2514, 2582, 2552,
                                          2517, 2516, 2586, 2570, 2580, 2562,
                                          2577, 2576, 2547, 2549, 2574, 2558,
                                          2581, 2515, 2561, 2579, 2518, 2807, 
                                          2571, 2555, 2645, 2542, 2537, 2536,
                                          2541, 2557, 2573, 2543, 2635, 2511,
                                          2655],

                          "land_area_ratio":0.08650301817
                          })
    
    pipeline.add_node(shift_production,
                         {"scale":params["pulse_production"]/100,
                          "items":("Item_group", "Pulses"),
                          "items_target":("Item_group", ["Cereals - Excluding Beer",
                                                         "Vegetables",
                                                         "Fruits - Excluding Wine",
                                                         "Vegetables Oils",
                                                         "Spices",
                                                         "Starchy Roots",
                                                         "Sugar Crops",
                                                         "Oilcrops",
                                                         "Treenuts",                                                      
                                                         ]),
                          "land_area_ratio":0.03327492402
                          })

    pipeline.add_node(peatland_restoration,
                        {"restore_fraction":0.0475*params["lowland_peatland"]/100,
                         "new_land_type":"Restored lowland peat",
                         "old_land_type":["Arable"],
                         "items":"Vegetal Products",
                         })
    
    pipeline.add_node(peatland_restoration,
                        {"restore_fraction":0.0273*params["upland_peatland"]/100,
                         "new_land_type":"Restored upland peat",
                         "old_land_type":["Improved grassland", "Semi-natural grassland"],
                         "items":"Animal Products",
                         })
    
    pipeline.add_node(managed_agricultural_land_carbon_model,
                        {"fraction":params["pasture_soil_carbon"]/100,
                         "managed_class":"Managed pasture",
                         "old_class":["Improved grassland", "Semi-natural grassland"]})
    
    pipeline.add_node(managed_agricultural_land_carbon_model,
                        {"fraction":params["arable_soil_carbon"]/100,
                         "managed_class":"Managed arable",
                         "old_class":"Arable"})


    pipeline.add_node(mixed_farming_model,
                        {"fraction":params["mixed_farming"]/100,
                         "prod_scale_factor":adv_settings["mixed_farming_production_scale"],
                         "items":("Item_origin","Vegetal Products"),
                         "secondary_prod_scale_factor":adv_settings["mixed_farming_secondary_production_scale"],
                         "secondary_items":("Item_origin","Animal Products")})

    # Livestock farming practices        
    
    pipeline.add_node(agroecology_model,
                            {"land_percentage":params["silvopasture"]/100.,
                            "agroecology_class":"Silvopasture",
                            "land_type":["Improved grassland",
                                         "Semi-natural grassland",
                                         "Managed pasture"],
                            "tree_coverage":adv_settings["agroecology_tree_coverage"],
                            "replaced_items":[2731, 2732],
                            "seq_ha_yr":adv_settings["agroecology_tree_coverage"]*(params["bdleaf_conif_ratio"]/100 * adv_settings["bdleaf_seq_ha_yr"] \
                                        + (1 - params["bdleaf_conif_ratio"]/100) * adv_settings["conif_seq_ha_yr"]) \
                                        + (1 - adv_settings["agroecology_tree_coverage"]) * adv_settings["managed_pasture_seq_ha_yr"],
                            })
    
    pipeline.add_node(scale_impact,
                         {"items":("Item_origin","Vegetal Products"),
                          "scale_factor":adv_settings["nitrogen_ghg_factor"]*params["nitrogen"]/100})

    pipeline.add_node(scale_impact,
                            {"items":("Item_origin","Animal Products"),
                            "scale_factor":adv_settings["methane_ghg_factor"]*params["methane_inhibitor"]/100})
    
    pipeline.add_node(scale_production,
                            {"scale_factor":1+params["stock_density"]/100,
                             "items":[2731, 2732, 2733, 2735, 2948, 2740, 2743]})

    pipeline.add_node(scale_impact,
                            {"items":("Item_origin","Animal Products"),
                            "scale_factor":adv_settings["manure_ghg_factor"]*params["manure_management"]/100})

    pipeline.add_node(scale_impact,
                            {"items":("Item_origin","Animal Products"),
                            "scale_factor":adv_settings["breeding_ghg_factor"]*params["animal_breeding"]/100})

    pipeline.add_node(scale_impact,
                            {"items":("Item_origin","Animal Products"),
                            "scale_factor":adv_settings["fossil_livestock_ghg_factor"]*params["fossil_livestock"]/100})

    pipeline.add_node(scale_impact,
                            {"items":("Item_origin","Animal Products"),
                            "scale_factor":1-1/(params["livestock_yield"]/100)})
    
    pipeline.add_node(scale_production,
                            {"scale_factor":params["livestock_yield"]/100,
                            "items":("Item_origin","Animal Products")})


    # Arable farming practices
    pipeline.add_node(agroecology_model,
                            {"land_percentage":params["agroforestry"]/100.,
                            "agroecology_class":"Agroforestry",
                            "land_type":["Arable",
                                         "Managed arable"],
                            "tree_coverage":adv_settings["agroecology_tree_coverage"],
                            "replaced_items":2511,
                            "seq_ha_yr":adv_settings["agroecology_tree_coverage"]*(params["bdleaf_conif_ratio"]/100 * adv_settings["bdleaf_seq_ha_yr"] \
                                        + (1 - params["bdleaf_conif_ratio"]/100) * adv_settings["conif_seq_ha_yr"]) \
                                        + (1 - adv_settings["agroecology_tree_coverage"]) * adv_settings["managed_pasture_seq_ha_yr"],
                            
                            })
    
    pipeline.add_node(extra_urban_farming,
                         {"fraction":params["vertical_farming"]/100,
                          "items":("Item_group", ["Vegetables", "Fruits - Excluding Wine"])
                          })

    pipeline.add_node(scale_impact,
                            {"items":("Item_origin", "Vegetal Products"),
                            "scale_factor":adv_settings["fossil_arable_ghg_factor"]*params["fossil_arable"]/100})

    # Technology & Innovation    
    pipeline.add_node(ccs_model,
                            {"waste_BECCS":params["waste_BECCS"]*1e6,
                            "overseas_BECCS":params["overseas_BECCS"]*1e6,
                            "DACCS":params["DACCS"]*1e6,
                            "biochar":params["biochar"]*1e6})

    pipeline.add_node(label_new_forest)

    # Compute emissions and sequestration
    pipeline.add_node(forest_sequestration_model,
                            {"land_type":["Broadleaf woodland",
                                          "Coniferous woodland",
                                          "New Broadleaf woodland",
                                          "New Coniferous woodland",
                                          "Restored upland peat",
                                          "Restored lowland peat",
                                          "Managed arable",
                                          "Managed pasture",
                                          "Mixed farming",
                                          "Bioenergy crops (arable)",
                                          "Bioenergy crops (pasture)"
                                          ],
                            "seq":[adv_settings["bdleaf_seq_ha_yr"],
                                   adv_settings["conif_seq_ha_yr"],
                                   adv_settings["new_bdleaf_seq_ha_yr"],
                                   adv_settings["new_conif_seq_ha_yr"],
                                   adv_settings["peatland_seq_ha_yr"],
                                   adv_settings["peatland_seq_ha_yr"],
                                   adv_settings["managed_arable_seq_ha_yr"],
                                   adv_settings["managed_pasture_seq_ha_yr"],
                                   adv_settings["mixed_farming_seq_ha_yr"],
                                   adv_settings["beccs_crops_arable_seq_ha_yr"],
                                   adv_settings["beccs_crops_pasture_seq_ha_yr"]
                                   ]})
    # Compute emissions
    pipeline.add_node(compute_emissions)

    # Compute additional metrics 
    pipeline.add_node(compute_metrics)

    # Generate pathway URL
    pipeline.add_node(generate_API_url,
                         {"keys":[
                             "ruminant",
                             "pig_poultry",
                             "fish_seafood",
                             ]}
    )

    return pipeline
