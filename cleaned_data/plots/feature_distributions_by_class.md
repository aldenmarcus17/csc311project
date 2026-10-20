# Feature distributions by class

This report uses 738 train+validation rows only. The test set was not read.

## Class balance

| Class | Rows | Percentage |
|---|---:|---:|
| Baseball | 246 | 33.3% |
| Basketball | 246 | 33.3% |
| Soccer | 246 | 33.3% |

## Feature distributions by class

For numeric and ordinal features, values are means. For binary features, values are the percentage of rows in each class with the feature active.

| Feature | Type | Baseball | Basketball | Soccer |
|---|---|---:|---:|---:|
| ease | numeric | 4.16 | 5.98 | 6.23 |
| strategy | ordinal mean | 3.49 | 3.88 | 3.92 |
| injury | ordinal mean | 2.97 | 3.81 | 3.80 |
| brilliance | ordinal mean | 3.74 | 3.57 | 2.89 |
| where_indoor_facility | binary activation | 23.6% | 95.5% | 62.2% |
| where_outdoor_public_space | binary activation | 86.2% | 88.6% | 92.3% |
| where_school_or_university_facility | binary activation | 75.6% | 93.9% | 89.4% |
| where_community_recreation_facility | binary activation | 48.8% | 88.6% | 75.6% |
| where_private_sports_facility | binary activation | 54.5% | 82.9% | 68.7% |
| where_at_home_or_nearby | binary activation | 13.8% | 64.6% | 47.6% |
| where_others | binary activation | 3.7% | 9.8% | 15.4% |
| season_winter | binary activation | 5.7% | 58.9% | 7.7% |
| season_spring | binary activation | 74.0% | 85.0% | 70.7% |
| season_summer | binary activation | 91.1% | 91.9% | 91.5% |
| season_fall | binary activation | 71.1% | 87.4% | 67.9% |
| music_hip_hop_or_rap | binary activation | 24.4% | 82.9% | 39.0% |
| music_latin | binary activation | 3.7% | 0.0% | 39.0% |
| music_folk_or_traditional | binary activation | 11.4% | 1.6% | 2.4% |
| music_pop | binary activation | 34.6% | 48.0% | 39.4% |
| music_rock | binary activation | 22.8% | 28.9% | 25.2% |
| music_electronic_or_dance | binary activation | 13.8% | 19.9% | 35.4% |
| music_no_particular_type | binary activation | 28.0% | 8.1% | 17.5% |
| music_classical_or_jazz | binary activation | 7.3% | 2.0% | 1.2% |
| music_country | binary activation | 31.7% | 3.3% | 4.1% |
| music_other | binary activation | 1.6% | 2.8% | 3.7% |
| food_hotdog | binary activation | 64.2% | 17.1% | 21.1% |
| food_popcorn | binary activation | 18.3% | 18.3% | 8.1% |
| food_beer | binary activation | 32.5% | 15.9% | 51.2% |
| food_soda | binary activation | 32.9% | 39.4% | 38.6% |
| food_pizza | binary activation | 4.9% | 17.9% | 15.9% |
| food_fries | binary activation | 10.6% | 15.0% | 11.8% |
| food_chips | binary activation | 7.3% | 9.3% | 20.3% |
| food_nachos | binary activation | 6.5% | 15.4% | 7.3% |
| food_burger | binary activation | 8.5% | 11.8% | 13.0% |
| food_wings | binary activation | 1.6% | 18.7% | 7.7% |
| food_chicken | binary activation | 3.7% | 6.5% | 6.9% |
| food_peanuts | binary activation | 12.6% | 0.8% | 2.0% |
| food_sunflower | binary activation | 2.8% | 0.0% | 0.8% |
| food_pretzel | binary activation | 1.6% | 1.6% | 1.2% |
| food_water | binary activation | 1.6% | 2.4% | 2.0% |
| food_sports_drink | binary activation | 0.0% | 6.9% | 2.0% |
| food_poutine | binary activation | 1.2% | 0.4% | 1.2% |
| food_sandwich | binary activation | 0.8% | 0.4% | 2.4% |
| food_fish | binary activation | 0.0% | 0.0% | 3.7% |
| food_taco | binary activation | 0.4% | 0.4% | 2.0% |
| food_bbq | binary activation | 0.0% | 0.8% | 4.1% |
| food_sausage | binary activation | 0.0% | 0.0% | 2.4% |
| food_snacks | binary activation | 2.8% | 3.7% | 8.9% |
| food_none | binary activation | 2.4% | 3.3% | 2.4% |
| colour_red | binary activation | 16.3% | 31.7% | 22.0% |
| colour_blue | binary activation | 50.4% | 2.8% | 22.8% |
| colour_green | binary activation | 4.1% | 0.0% | 51.6% |
| colour_white | binary activation | 27.2% | 2.4% | 22.4% |
| colour_black | binary activation | 1.2% | 8.5% | 14.2% |
| colour_orange | binary activation | 1.2% | 53.7% | 0.8% |
| colour_yellow | binary activation | 1.6% | 6.9% | 6.5% |
| colour_brown | binary activation | 10.6% | 9.3% | 0.8% |
| colour_purple | binary activation | 0.0% | 4.1% | 0.0% |
| colour_grey | binary activation | 2.0% | 0.0% | 0.0% |
| song_waka | binary activation | 0.0% | 0.0% | 58.1% |
| song_baseball | binary activation | 32.5% | 0.0% | 0.0% |
| song_hiphop | binary activation | 2.0% | 19.1% | 0.0% |
| song_named | binary activation | 67.1% | 77.6% | 87.0% |
| pay_log | numeric | 3.70 | 4.58 | 4.64 |
| feel_excited | binary activation | 19.9% | 57.7% | 63.8% |
| feel_bored | binary activation | 30.5% | 6.1% | 10.6% |
| feel_tense | binary activation | 6.5% | 12.6% | 17.1% |
| feel_happy | binary activation | 9.3% | 11.4% | 8.9% |
| feel_calm | binary activation | 11.4% | 1.2% | 1.2% |
| feel_indifferent | binary activation | 7.7% | 5.3% | 4.1% |
| feel_curious | binary activation | 11.4% | 10.6% | 8.1% |
| feel_confused | binary activation | 7.7% | 0.8% | 2.4% |
| feel_none | binary activation | 7.7% | 2.4% | 3.7% |
