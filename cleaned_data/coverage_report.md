# Coverage report

Rows per split: train=606, val=132, train+val=738

## % of rows with at least one keyword matched
```
                   train   val  train+val
food (>=1 item)     95.0  90.2       94.2
colour (>=1)        96.4  97.0       96.5
feel (>=1)          89.1  93.9       90.0
song (recognised)   35.5  45.5       37.3
song (named any)    77.1  78.0       77.2
```

## % of rows setting each feature
```
                   train   val  train+val
food_hotdog         32.7  40.9       34.1
food_popcorn        15.5  12.1       14.9
food_beer           31.2  42.4       33.2
food_soda           37.0  37.1       37.0
food_pizza          13.2  11.4       12.9
food_fries          12.2  13.6       12.5
food_chips          13.0   9.1       12.3
food_nachos         10.1   8.3        9.8
food_burger         10.6  13.6       11.1
food_wings           8.9  11.4        9.3
food_chicken         6.6   1.5        5.7
food_peanuts         5.3   4.5        5.1
food_sunflower       1.5   0.0        1.2
food_pretzel         1.7   0.8        1.5
food_water           1.8   3.0        2.0
food_sports_drink    3.5   0.8        3.0
food_poutine         0.8   1.5        0.9
food_sandwich        1.5   0.0        1.2
food_fish            1.3   0.8        1.2
food_taco            0.7   2.3        0.9
food_bbq             1.8   0.8        1.6
food_sausage         0.8   0.8        0.8
food_snacks          5.3   4.5        5.1
food_none            3.1   0.8        2.7
colour_red          24.8  16.7       23.3
colour_blue         24.9  27.3       25.3
colour_green        19.3  15.2       18.6
colour_white        18.5  12.1       17.3
colour_black         8.3   6.8        8.0
colour_orange       17.7  22.7       18.6
colour_yellow        5.3   3.8        5.0
colour_brown         7.6   3.8        6.9
colour_purple        1.7   0.0        1.4
colour_grey          0.5   1.5        0.7
feel_excited        46.9  48.5       47.2
feel_bored          15.3  17.4       15.7
feel_tense          11.7  13.6       12.1
feel_happy          10.1   9.1        9.9
feel_calm            4.5   5.3        4.6
feel_indifferent     5.3   7.6        5.7
feel_curious        10.2   9.1       10.0
feel_confused        3.8   3.0        3.7
feel_none            4.5   5.3        4.6
song_waka           19.1  20.5       19.4
song_baseball       10.2  13.6       10.8
song_hiphop          6.1  11.4        7.0
song_named          77.1  78.0       77.2
```

## Pay answers: how each was handled (% of rows)
```
                        train   val  train+val
pay                                           
blank (median)            1.2   1.5        1.2
numeric                  92.9  92.4       92.8
refusal (0)               3.6   3.0        3.5
unparsed text (median)    2.3   3.0        2.4
```

## Blank (missing) raw answers (% of rows)
```
            train  val  train+val
ease          0.8  1.5        0.9
strategy      0.8  1.5        0.9
injury        0.8  1.5        0.9
brilliance    0.8  1.5        0.9
where         0.8  1.5        0.9
seasons       0.8  1.5        0.9
music         0.8  1.5        0.9
food          2.3  6.1        3.0
colour        1.5  2.3        1.6
song          9.1  9.1        9.1
feel          1.0  1.5        1.1
pay           1.2  1.5        1.2
```

## TRAIN answers with no food match (top 15)
```
                                                                                                                  count
food                                                                                                                   
<blank>                                                                                                              14
monster drink                                                                                                         1
i haven't ever watched it before.                                                                                     1
shawarma and cold drinks                                                                                              1
donuts                                                                                                                1
perhaps some refreshments for the athletes but i do not associate any food or drink with the viewing experience.      1
scran                                                                                                                 1
takeout/doordash anything                                                                                             1
classic arena food staples.                                                                                           1
no                                                                                                                    1
iced tea                                                                                                              1
power drinks, protein, carbs..                                                                                        1
takoyaki                                                                                                              1
vegetable                                                                                                             1
chewing gum                                                                                                           1
```

## TRAIN answers with no colour match (top 15)
```
                                           count
colour                                          
<blank>                                        9
dark major colours of the rainbow              2
n/a, since i know nothing about it at all      1
all colours                                    1
#name?                                         1
navy                                           1
no idea                                        1
no particular color                            1
wood                                           1
unsure                                         1
beige                                          1
wood-toned                                     1
none                                           1
```

## TRAIN answers with no feel match (top 15)
```
                                        count
feel                                         
<blank>                                     6
not my type                                 3
dopamine release                            3
none                                        2
cool                                        2
electric                                    1
waiting for an advertisement to finish      1
cool, but will be tired very quickly.       1
introspective                               1
no sense                                    1
giddy                                       1
patience                                    1
not sure                                    1
hard to see the ball                        1
rooting for my team.                        1
```
