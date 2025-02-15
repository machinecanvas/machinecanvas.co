-- MySQL dump 10.13  Distrib 8.0.40, for Win64 (x86_64)
--
-- Host: localhost    Database: machinecanvas
-- ------------------------------------------------------
-- Server version	8.0.40

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `images`
--

DROP TABLE IF EXISTS `images`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `images` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int DEFAULT NULL,
  `filename` varchar(255) NOT NULL,
  `category` varchar(255) NOT NULL,
  `price` decimal(10,2) NOT NULL,
  `upload_date` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `email` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`)
) ENGINE=InnoDB AUTO_INCREMENT=97 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `images`
--

LOCK TABLES `images` WRITE;
/*!40000 ALTER TABLE `images` DISABLE KEYS */;
INSERT INTO `images` VALUES (12,18,'DALLE_2024-09-23_21.59.13_-_A_beautifully_presented_plate_of_peanut_butter_fudge_sweetened_with_monk_fruit._The_fudge_is_cut_into_small_squares_with_a_creamy_smooth_texture._Th.webp','food_drink',5.00,'2024-11-03 00:19:46',NULL),(16,19,'DALLE_2024-10-16_13.40.08_-_A_representative_image_for_the_Nature_category_featuring_lush_green_leaves_and_trees_in_a_peaceful_forested_setting_with_sunlight_filtering_throug.webp','nature',5.00,'2024-11-06 09:36:44',NULL),(17,19,'DALLE_2024-10-18_13.45.06_-_A_bold_and_dynamic_abstract_composition_featuring_geometric_shapes_that_overlap_and_intersect_with_vibrant_color_gradients._The_design_should_convey_a.webp','abstract',5.00,'2024-11-06 09:44:22',NULL),(19,19,'DALLE_2024-10-18_13.45.18_-_A_chaotic_and_expressive_abstract_painting_with_splashes_of_bright_red_yellow_and_orange_paint._The_image_should_feature_thick_brushstrokes_and_ener.webp','abstract',5.00,'2024-11-06 10:03:57',NULL),(21,19,'DALLE_2024-10-21_20.28.05_-_A_captivating_abstract_image_featuring_a_mix_of_geometric_shapes_swirling_vibrant_colors_and_fluid_forms_blending_together._The_composition_has_a_se.webp','abstract',5.00,'2024-11-06 10:06:14',NULL),(22,19,'DALLE_2024-10-23_11.56.10_-_A_peaceful_forest_scene_with_sunlight_streaming_through_tall_trees_casting_shadows_on_the_forest_floor_covered_with_green_moss_and_fallen_leaves._The.webp','nature',5.00,'2024-11-06 10:06:48',NULL),(23,19,'DALLE_2024-10-23_11.56.20_-_A_mountain_landscape_with_a_clear_blue_sky_rugged_peaks_dusted_with_snow_and_a_crystal-clear_lake_reflecting_the_surrounding_scenery._The_foreground.webp','nature',5.00,'2024-11-06 10:07:20',NULL),(24,19,'DALLE_2024-10-23_11.56.34_-_A_dense_tropical_jungle_with_thick_vibrant_greenery_tall_palm_trees_and_vines_hanging_from_the_canopy._The_air_is_humid_and_there_are_bright_exot.webp','nature',5.00,'2024-11-06 10:07:45',NULL),(25,19,'DALLE_2024-10-23_11.56.38_-_A_serene_beach_scene_with_soft_golden_sand_gentle_ocean_waves_lapping_the_shore_and_a_clear_horizon_where_the_water_meets_the_sky._A_few_seashells_a.webp','nature',5.00,'2024-11-06 10:08:16',NULL),(26,19,'DALLE_2024-10-23_12.55.18_-_A_majestic_lion_standing_on_a_rock_in_the_savannah_its_golden_mane_blowing_in_the_wind._The_background_shows_a_vast_open_plain_with_a_few_acacia_tree.webp','animals',5.00,'2024-11-06 10:08:56',NULL),(27,19,'DALLE_2024-10-23_12.55.27_-_A_playful_dolphin_jumping_out_of_the_water_in_the_ocean_its_sleek_body_glistening_in_the_sunlight._The_background_features_gentle_waves_and_a_clear_b.webp','animals',5.00,'2024-11-06 10:09:16',NULL),(28,19,'DALLE_2024-10-23_12.55.31_-_A_close-up_of_a_curious_red_panda_climbing_a_tree_branch_its_soft_fur_and_striped_tail_visible._The_scene_is_set_in_a_lush_forest_with_leaves_and_bra.webp','animals',5.00,'2024-11-06 10:09:37',NULL),(29,19,'DALLE_2024-10-23_12.55.37_-_A_graceful_bald_eagle_soaring_high_in_the_sky_its_wings_fully_extended_and_sharp_eyes_focused_below._The_background_features_a_clear_blue_sky_with_so.webp','animals',5.00,'2024-11-06 10:09:58',NULL),(30,19,'DALLE_2024-10-23_13.00.08_-_A_futuristic_cityscape_with_towering_skyscrapers_made_of_glass_and_steel_advanced_transportation_systems_including_flying_cars_and_bright_holographi.webp','technology',5.00,'2024-11-06 10:20:43',NULL),(31,19,'DALLE_2024-10-23_13.00.10_-_A_sleek_modern_robotic_arm_working_in_a_factory_assembling_electronic_components_with_precision._The_background_shows_a_high-tech_manufacturing_floo.webp','technology',5.00,'2024-11-06 10:21:17',NULL),(32,19,'DALLE_2024-10-23_13.00.13_-_A_close-up_of_a_modern_smartphone_with_a_holographic_display_projecting_data_and_apps_in_3D._The_phone_is_resting_on_a_sleek_table_and_the_background.webp','technology',5.00,'2024-11-06 10:21:43',NULL),(33,19,'DALLE_2024-10-23_13.00.16_-_A_high-tech_data_center_with_rows_of_illuminated_server_racks_cables_and_cooling_systems._The_servers_are_organized_in_a_sleek_futuristic_design_wi.webp','technology',5.00,'2024-11-06 10:22:12',NULL),(34,19,'DALLE_2024-10-23_13.02.14_-_A_portrait_of_a_young_woman_sitting_in_a_cozy_cafe_smiling_and_looking_out_the_window._She_has_long_wavy_hair_and_is_wearing_casual_clothes_with_a_w.webp','people',5.00,'2024-11-06 10:22:54',NULL),(35,19,'DALLE_2024-10-23_13.02.19_-_A_group_of_friends_sitting_around_a_campfire_at_night_laughing_and_enjoying_each_others_company._The_warm_glow_of_the_fire_illuminates_their_faces_.webp','people',5.00,'2024-11-06 10:24:10',NULL),(36,19,'DALLE_2024-10-23_13.02.22_-_A_young_couple_walking_hand-in-hand_through_a_city_park_during_autumn_with_colorful_leaves_falling_around_them._They_are_bundled_up_in_warm_coats_an.webp','people',5.00,'2024-11-06 10:25:23',NULL),(37,19,'DALLE_2024-10-23_13.02.16_-_A_businessman_standing_in_a_modern_office_holding_a_tablet_and_smiling_confidently._He_is_dressed_in_a_sharp_suit_and_the_background_shows_large_win.webp','people',5.00,'2024-11-06 10:27:21',NULL),(38,19,'DALLE_2024-10-23_13.04.05_-_A_stunning_modern_skyscraper_with_a_sleek_glass_facade_towering_over_the_city._The_building_features_sharp_angles_and_a_reflective_surface_blending_.webp','architecture',5.00,'2024-11-06 10:28:12',NULL),(39,19,'DALLE_2024-10-23_13.04.09_-_An_ancient_Roman_amphitheater_with_massive_stone_arches_and_weathered_columns._The_structure_is_partially_in_ruins_but_the_grand_architecture_still_s.webp','architecture',5.00,'2024-11-06 10:28:40',NULL),(40,19,'DALLE_2024-10-23_13.04.17_-_A_traditional_Japanese_temple_surrounded_by_lush_gardens._The_temple_features_ornate_wooden_architecture_with_curved_tiled_roofs_and_the_peaceful_en.webp','architecture',5.00,'2024-11-06 10:29:13',NULL),(41,19,'DALLE_2024-10-23_13.04.27_-_A_gothic_cathedral_with_towering_spires_intricate_stained_glass_windows_and_flying_buttresses._The_stone_facade_is_adorned_with_detailed_carvings_a.webp','architecture',5.00,'2024-11-06 10:29:35',NULL),(42,19,'DALLE_2024-10-23_13.06.39_-_A_vibrant_abstract_painting_featuring_bold_swirling_colors_in_red_blue_and_yellow._The_shapes_are_fluid_and_dynamic_creating_a_sense_of_movement_a.webp','abstract',5.00,'2024-11-06 10:33:16',NULL),(43,19,'DALLE_2024-10-23_13.06.44_-_A_detailed_pencil_sketch_of_a_human_face_with_intricate_shading_showing_a_range_of_emotions._The_sketch_is_realistic_with_fine_lines_emphasizing_the.webp','art_design',5.00,'2024-11-06 10:33:51',NULL),(44,19,'DALLE_2024-10-23_13.06.46_-_An_elegant_digital_illustration_of_a_fashion_design_featuring_a_chic_evening_gown._The_gown_is_sleek_and_modern_with_flowing_fabric_and_intricate_pa.webp','art_design',5.00,'2024-11-06 10:34:18',NULL),(45,19,'DALLE_2024-10-23_13.07.53_-_A_creative_workspace_filled_with_art_supplies_sketches_and_a_half-finished_painting_on_an_easel._The_room_has_a_cozy_artistic_vibe_with_brushes_pa.webp','art_design',5.00,'2024-11-06 10:34:40',NULL),(46,19,'DALLE_2024-10-23_13.08.06_-_A_large_public_mural_on_a_city_wall_featuring_vibrant_colors_and_abstract_shapes_that_blend_into_each_other._The_mural_is_surrounded_by_an_urban_land.webp','art_design',5.00,'2024-11-06 10:35:07',NULL),(47,19,'DALLE_2024-10-23_13.12.31_-_A_busy_office_with_professionals_working_at_their_desks_using_computers_and_discussing_projects._The_office_features_modern_design_with_glass_partiti.webp','business',5.00,'2024-11-06 10:36:26',NULL),(50,19,'DALLE_2024-10-23_13.12.37_-_A_business_meeting_room_with_executives_gathered_around_a_large_table_reviewing_documents_and_discussing_a_strategy._The_room_has_a_professional_atmo_1.webp','business',5.00,'2024-11-06 10:38:30',NULL),(51,19,'DALLE_2024-10-23_22.15.49_-_A_professional_modern_office_environment_with_sleek_desks_laptops_and_city_skyline_view_through_large_windows_representing_a_business_workspace._T.webp','business',5.00,'2024-11-06 10:39:32',NULL),(52,19,'DALLE_2024-10-23_22.17.49_-_A_stylish_fashion_photoshoot_in_an_urban_setting_featuring_a_model_wearing_high-fashion_clothing_with_modern_chic_accessories._The_setting_includes_.webp','fashion',5.00,'2024-11-06 10:40:53',NULL),(53,19,'DALLE_2024-10-23_22.17.54_-_A_high-end_fashion_runway_show_with_models_wearing_bold_and_elegant_designer_outfits_walking_confidently_down_the_runway._The_scene_features_dramatic.webp','fashion',5.00,'2024-11-06 10:41:19',NULL),(54,19,'DALLE_2024-10-23_22.18.03_-_A_casual_streetwear_fashion_scene_featuring_models_walking_along_a_trendy_neighborhood_with_contemporary_urban_outfits._The_clothing_includes_stylish.webp','fashion',5.00,'2024-11-06 10:41:40',NULL),(55,19,'DALLE_2024-10-23_22.17.58_-_A_glamorous_fashion_editorial_photoshoot_set_in_an_elegant_indoor_space_featuring_a_model_dressed_in_avant-garde_designer_attire_with_bold_colors_an.webp','fashion',5.00,'2024-11-06 10:42:11',NULL),(56,19,'DALLE_2024-10-23_22.22.04_-_A_close-up_of_a_gourmet_plate_of_food_in_a_high-end_restaurant_featuring_a_carefully_arranged_dish_with_vibrant_colors_garnished_with_microgreens_an.webp','food_drink',5.00,'2024-11-06 10:42:50',NULL),(57,19,'DALLE_2024-10-23_22.22.07_-_A_vibrant_smoothie_bowl_with_fresh_fruits_seeds_and_colorful_toppings_set_on_a_rustic_wooden_table_in_a_cozy_cafe_environment.webp','food_drink',5.00,'2024-11-06 10:43:33',NULL),(58,19,'DALLE_2024-10-23_22.22.09_-_A_table_set_with_various_colorful_cocktails_featuring_fruit_garnishes_and_served_in_elegant_glassware_with_a_lively_bar_setting_in_the_background.webp','food_drink',5.00,'2024-11-06 10:43:55',NULL),(59,19,'DALLE_2024-10-23_22.22.12_-_A_platter_of_assorted_sushi_rolls_elegantly_presented_on_a_dark_slate_plate_with_chopsticks_and_soy_sauce_on_the_side_in_a_modern_Japanese_restauran.webp','food_drink',5.00,'2024-11-06 10:44:20',NULL),(60,19,'DALLE_2024-10-23_22.26.02_-_A_sleek_sports_car_on_a_highway_with_a_modern_city_skyline_in_the_background_showing_speed_and_elegance.webp','vehicles',5.00,'2024-11-06 10:45:06',NULL),(61,19,'DALLE_2024-10-23_22.26.07_-_A_rugged_off-road_SUV_in_a_desert_landscape_climbing_over_rocky_terrain_with_dust_in_the_air.webp','vehicles',5.00,'2024-11-06 10:50:44',NULL),(62,19,'DALLE_2024-10-23_22.26.10_-_A_futuristic_electric_motorcycle_in_a_sleek_urban_environment_showcasing_modern_design_and_technology.webp','vehicles',5.00,'2024-11-06 10:51:18',NULL),(63,19,'DALLE_2024-10-23_22.26.17_-_A_luxury_yacht_sailing_on_a_calm_ocean_during_sunset_with_a_sleek_design_and_modern_features.webp','vehicles',5.00,'2024-11-06 10:51:47',NULL),(64,19,'DALLE_2024-10-23_22.28.36_-_A_colorful_animated_scene_featuring_a_futuristic_cityscape_with_flying_cars_and_robots_in_a_vibrant_and_lively_environment.webp','animated',5.00,'2024-11-06 10:52:19',NULL),(65,19,'DALLE_2024-10-23_22.28.40_-_An_animated_woodland_scene_with_cute_animals_and_bright_vivid_colors_set_in_a_magical_forest_with_glowing_plants_and_trees.webp','animated',5.00,'2024-11-06 10:52:41',NULL),(66,19,'DALLE_2024-10-23_22.36.17_-_A_vibrant_animated_fantasy_landscape_featuring_floating_islands_waterfalls_and_magical_creatures_flying_in_the_sky_with_a_colorful_sunset_in_the_ba.webp','animated',5.00,'2024-11-06 10:53:01',NULL),(67,19,'DALLE_2024-10-23_22.36.53_-_An_animated_cityscape_at_night_with_bright_neon_lights_futuristic_buildings_and_flying_cars_in_the_sky._The_city_is_vibrant_glowing_with_various_c.webp','animated',5.00,'2024-11-06 10:53:26',NULL),(68,19,'DALLE_2024-10-23_23.02.19_-_A_modern_living_room_with_stylish_furniture_featuring_a_sleek_sofa_coffee_table_and_elegant_lighting_fixtures_with_tasteful_decor_accents_like_pla.webp','home_decor',5.00,'2024-11-06 10:53:45',NULL),(69,19,'DALLE_2024-10-23_23.02.23_-_A_cozy_bedroom_decorated_in_a_minimalist_style_featuring_a_comfortable_bed_with_soft_bedding_a_small_bedside_table_with_a_lamp_and_framed_artwork_o.webp','home_decor',5.00,'2024-11-06 10:54:03',NULL),(70,19,'DALLE_2024-10-23_23.02.30_-_A_well-decorated_dining_room_with_a_stylish_dining_table_modern_chairs_elegant_lighting_fixtures_and_tasteful_decor_accents_like_a_vase_of_flowers_.webp','home_decor',5.00,'2024-11-06 10:54:26',NULL),(71,19,'DALLE_2024-10-23_23.02.36_-_A_modern_kitchen_with_sleek_cabinetry_a_marble_countertop_decorative_lighting_and_plants_adding_a_natural_touch_designed_with_stylish_decor.webp','home_decor',5.00,'2024-11-06 10:54:47',NULL),(72,19,'DALLE_2024-10-23_23.07.31_-_A_whimsical_hot_air_balloon_floating_above_a_serene_landscape_with_mountains_and_lakes_featuring_vibrant_and_unique_designs_on_the_balloon.webp','miscellaneous',5.00,'2024-11-06 10:55:15',NULL),(74,19,'DALLE_2024-10-23_23.08.06_-_An_artistic_mosaic_pattern_made_of_colorful_tiles_in_various_shapes_and_sizes_displayed_on_a_wall_with_a_textured_surface._The_design_includes_abstra.webp','art_design',5.00,'2024-11-06 10:56:11',NULL),(75,19,'DALLE_2024-10-23_23.08.45_-_A_surreal_floating_island_suspended_in_the_sky_with_a_waterfall_flowing_off_the_edge_surrounded_by_clouds_and_a_magical_dreamlike_atmosphere.webp','miscellaneous',5.00,'2024-11-06 10:56:31',NULL),(76,19,'DALLE_2024-10-23_23.07.36_-_An_abstract_sculpture_made_of_intertwined_metallic_shapes_located_in_an_open_urban_plaza_reflecting_light_in_various_colors_1.webp','miscellaneous',5.00,'2024-11-06 10:57:18',NULL),(77,19,'DALLE_2024-10-23_23.07.41_-_A_futuristic_glowing_orb_hovering_above_a_reflective_surface_casting_bright_and_ethereal_light_with_a_dark_background.webp','miscellaneous',5.00,'2024-11-06 10:57:49',NULL),(78,19,'DALLE_2024-11-08_14.07.15_-_A_dramatic_scene_of_Torquay_England_during_a_storm._Dark_clouds_fill_the_sky_casting_an_ominous_shadow_over_the_coastal_town._Large_waves_crash_agai.webp','miscellaneous',5.00,'2024-11-08 14:12:05',NULL),(79,19,'DALLE_2024-11-13_19.17.01_-_A_dynamic_football_scene_featuring_a_player_in_motion_on_a_grassy_field_showcasing_the_action_and_energy_of_the_game_with_vivid_details_realistic_te.webp','sports',5.00,'2024-11-13 21:48:24',NULL),(80,19,'DALLE_2024-11-13_19.17.04_-_A_basketball_player_mid-jump_for_a_slam_dunk_in_a_vibrant_stadium_capturing_the_energy_and_excitement_of_the_game._The_image_showcases_realistic_text.webp','sports',5.00,'2024-11-13 21:48:43',NULL),(81,19,'DALLE_2024-11-13_19.17.06_-_A_tennis_player_preparing_to_serve_on_a_vibrant_tennis_court_capturing_the_precision_and_focus_of_the_sport._The_image_highlights_realistic_textures_.webp','sports',5.00,'2024-11-13 21:48:59',NULL),(83,19,'DALLE_2024-11-13_19.17.06_-_A_tennis_player_preparing_to_serve_on_a_vibrant_tennis_court_capturing_the_precision_and_focus_of_the_sport._The_image_highlights_realistic_textures__1.webp','sports',5.00,'2024-11-13 21:50:11',NULL),(84,19,'DALLE_2024-11-14_16.10.48_-_An_AI-generated_image_of_famous_global_landmarks_featuring_icons_such_as_the_Eiffel_Tower_the_Great_Wall_of_China_the_Statue_of_Liberty_and_the_Ta.webp','world',5.00,'2024-11-14 16:14:10',NULL),(85,19,'DALLE_2024-11-14_16.10.57_-_An_AI-generated_image_showcasing_traditional_cultural_clothing_from_diverse_regions_of_the_world_featuring_attire_from_various_cultures_such_as_Japan.webp','world',5.00,'2024-11-14 16:14:23',NULL),(86,19,'DALLE_2024-11-14_16.11.04_-_An_AI-generated_image_of_traditional_cuisine_from_around_the_world_featuring_diverse_dishes_such_as_Italian_pasta_Japanese_sushi_Mexican_tacos_and.webp','world',5.00,'2024-11-14 16:14:37',NULL),(87,19,'DALLE_2024-11-14_16.12.25_-_An_AI-generated_image_showcasing_people_from_different_cultures_engaging_in_traditional_celebrations_and_festivals._The_scene_includes_colorful_attire.webp','world',5.00,'2024-11-14 16:14:52',NULL),(88,19,'DALLE_2024-11-14_16.16.32_-_An_AI-generated_image_depicting_ancient_historical_architecture_featuring_iconic_structures_like_the_Pyramids_of_Giza_the_Colosseum_in_Rome_and_the.webp','history',5.00,'2024-11-14 16:20:22',NULL),(89,19,'DALLE_2024-11-14_16.16.41_-_An_AI-generated_image_illustrating_ancient_artifacts_and_relics_from_various_civilizations_including_items_like_Egyptian_statues_Roman_coins_and_Gr.webp','history',5.00,'2024-11-14 16:20:38',NULL),(90,19,'DALLE_2024-11-14_16.16.46_-_An_AI-generated_image_portraying_a_historical_battle_scene_with_soldiers_from_ancient_times_engaged_in_combat._The_image_includes_elements_like_armor.webp','history',5.00,'2024-11-14 16:20:58',NULL),(91,19,'DALLE_2024-11-14_16.16.54_-_An_AI-generated_image_of_a_historical_manuscript_or_ancient_map_featuring_intricate_designs_handwritten_script_and_faded_ink._The_image_should_have.webp','history',5.00,'2024-11-14 16:21:16',NULL),(92,19,'DALLE_2024-11-14_16.24.16_-_An_AI-generated_image_representing_the_finance_category_as_a_whole._The_image_should_include_symbolic_elements_like_stock_market_charts_coins_paper_.webp','finance',5.00,'2024-11-14 16:25:25',NULL),(93,19,'DALLE_2024-11-14_16.24.24_-_An_AI-generated_image_focusing_on_the_concept_of_investment_in_finance_featuring_a_stock_market_chart_with_rising_and_falling_lines_a_laptop_and_a_.webp','finance',5.00,'2024-11-14 16:25:39',NULL),(94,19,'DALLE_2024-11-14_16.24.27_-_An_AI-generated_image_illustrating_personal_finance_management_featuring_elements_like_a_household_budget_planner_cash_credit_cards_and_a_piggy_ba.webp','finance',5.00,'2024-11-14 16:25:57',NULL),(95,19,'DALLE_2024-11-14_16.24.30_-_An_AI-generated_image_representing_financial_growth_and_wealth_building_featuring_elements_like_stacked_coins_a_small_plant_growing_from_the_coins_.webp','finance',5.00,'2024-11-14 16:26:19',NULL),(96,19,'DALLE_2024-09-22_10.29.21_-_Two_halves_of_a_bell_pepper_filled_with_avocado_and_baked_egg_resting_on_a_white_plate._The_egg_yolk_is_still_slightly_runny_and_the_avocado_looks_c.webp','food_drink',5.00,'2024-11-20 21:26:49',NULL);
/*!40000 ALTER TABLE `images` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `pricing_plans`
--

DROP TABLE IF EXISTS `pricing_plans`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `pricing_plans` (
  `id` int NOT NULL AUTO_INCREMENT,
  `plan_name` varchar(50) NOT NULL,
  `price` decimal(10,2) NOT NULL,
  `upload_limit` int DEFAULT NULL,
  `revenue_share` decimal(5,2) DEFAULT '0.10',
  `description` text,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `pricing_plans`
--

LOCK TABLES `pricing_plans` WRITE;
/*!40000 ALTER TABLE `pricing_plans` DISABLE KEYS */;
INSERT INTO `pricing_plans` VALUES (1,'Developer Plan',0.00,99999,0.00,'Temporary plan for development testing');
/*!40000 ALTER TABLE `pricing_plans` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `purchases`
--

DROP TABLE IF EXISTS `purchases`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `purchases` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_email` varchar(255) NOT NULL,
  `image_id` int NOT NULL,
  `purchase_date` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `amount` decimal(10,2) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `purchases`
--

LOCK TABLES `purchases` WRITE;
/*!40000 ALTER TABLE `purchases` DISABLE KEYS */;
/*!40000 ALTER TABLE `purchases` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `saved_images`
--

DROP TABLE IF EXISTS `saved_images`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `saved_images` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `image_id` int NOT NULL,
  `saved_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  KEY `image_id` (`image_id`),
  CONSTRAINT `saved_images_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
  CONSTRAINT `saved_images_ibfk_2` FOREIGN KEY (`image_id`) REFERENCES `images` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=33 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `saved_images`
--

LOCK TABLES `saved_images` WRITE;
/*!40000 ALTER TABLE `saved_images` DISABLE KEYS */;
INSERT INTO `saved_images` VALUES (29,19,60,'2024-11-11 13:45:48'),(30,19,17,'2024-11-12 21:53:41'),(32,19,96,'2024-11-20 21:26:54');
/*!40000 ALTER TABLE `saved_images` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `user_credits`
--

DROP TABLE IF EXISTS `user_credits`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `user_credits` (
  `user_id` int NOT NULL,
  `credits` int DEFAULT '0',
  PRIMARY KEY (`user_id`),
  CONSTRAINT `user_credits_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `user_credits`
--

LOCK TABLES `user_credits` WRITE;
/*!40000 ALTER TABLE `user_credits` DISABLE KEYS */;
INSERT INTO `user_credits` VALUES (19,10);
/*!40000 ALTER TABLE `user_credits` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `user_subscriptions`
--

DROP TABLE IF EXISTS `user_subscriptions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `user_subscriptions` (
  `user_id` int NOT NULL,
  `plan_id` int NOT NULL,
  `start_date` date DEFAULT NULL,
  `end_date` date DEFAULT NULL,
  `credits_remaining` int DEFAULT '3',
  `uploads_this_month` int DEFAULT '0',
  `upload_limit` int DEFAULT '3',
  PRIMARY KEY (`user_id`,`plan_id`),
  KEY `plan_id` (`plan_id`),
  CONSTRAINT `user_subscriptions_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`),
  CONSTRAINT `user_subscriptions_ibfk_2` FOREIGN KEY (`plan_id`) REFERENCES `pricing_plans` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `user_subscriptions`
--

LOCK TABLES `user_subscriptions` WRITE;
/*!40000 ALTER TABLE `user_subscriptions` DISABLE KEYS */;
INSERT INTO `user_subscriptions` VALUES (19,1,'2024-11-03',NULL,1,2,1);
/*!40000 ALTER TABLE `user_subscriptions` ENABLE KEYS */;
UNLOCK TABLES;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = cp850 */ ;
/*!50003 SET character_set_results = cp850 */ ;
/*!50003 SET collation_connection  = cp850_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
/*!50003 CREATE*/ /*!50017 DEFINER=`root`@`localhost`*/ /*!50003 TRIGGER `set_start_date_before_insert` BEFORE INSERT ON `user_subscriptions` FOR EACH ROW BEGIN
    IF NEW.start_date IS NULL THEN
        SET NEW.start_date = CURDATE();
    END IF;
END */;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;

--
-- Table structure for table `users`
--

DROP TABLE IF EXISTS `users`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `users` (
  `id` int NOT NULL AUTO_INCREMENT,
  `email` varchar(255) DEFAULT NULL,
  `password` varchar(255) DEFAULT NULL,
  `username` varchar(255) DEFAULT NULL,
  `dob` date DEFAULT NULL,
  `confirmed` tinyint(1) DEFAULT '0',
  `subscription` enum('free','paid') DEFAULT 'free',
  `user_id` varchar(255) DEFAULT NULL,
  `credits` int DEFAULT '0',
  `upload_limit` int DEFAULT '0',
  `account_holder_name` varchar(255) DEFAULT NULL,
  `account_number` varchar(20) DEFAULT NULL,
  `sort_code` varchar(10) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `user_id` (`user_id`)
) ENGINE=InnoDB AUTO_INCREMENT=20 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `users`
--

LOCK TABLES `users` WRITE;
/*!40000 ALTER TABLE `users` DISABLE KEYS */;
INSERT INTO `users` VALUES (19,'jonsimm82@outlook.com','scrypt:32768:8:1$MhwMmbLI1LP5p0ck$c59a16df85223c83b685b631ad0a45c06dc58b9e055c679402629ab65488c5e9e56303f6a25cc6d66a9984ff52341bad8d53e78fe6f0a071ec3c947fbbd99a18',NULL,NULL,1,'free','jonsimm82',3,0,NULL,NULL,NULL);
/*!40000 ALTER TABLE `users` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-11-24 14:35:28
