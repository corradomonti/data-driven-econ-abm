;This is a complete version of the ABM

;these are the parameters and quantities useful on a global level
globals [number-of-apartments-per-location number-of-agents number-of-categories initial-price max-attractiveness R
         Y0 delta beta alpha gamma agents-list mu lambda tau nu additional-flux xiklist initial-distribution

         number-of-patches-at-distance-d patches-at-distance-d distances filename endtransient

         filename-prices filename-agents filename-transactions filename-buyers
         ]

;these quantities are useful at the level of each location
patches-own [attractiveness  utility-for-k-agents probability-for-k-agents
  market-price mean-income-here

  all-market-prices prices-of-transactions-this-step nb ns housed-k-agents
  ]

;these quantities are useful at the level of each agent
turtles-own [income income-category state proposed-movement reservation-price time-on-sale time-born]
;state = 0: buyer; state = 1: housed; state = 2: seller

to setup-parameters

  ;setup of the parameters
  clear-all
  ;parameters of the model
  set number-of-apartments-per-location 1000;
  set number-of-categories 3;
  set R 0 ;steepness of intrinsic attractiveness

  ;parameters of the agents
  set Y0 10 ;  minimum income level
  set delta 40 ; difference in income between categories
  set beta 0.5 ; weight given to the attractiveness in the utility function

  set mu 0.10 ;initial markup on market price
  set lambda 0.95 ;price reduction if unsuccessful sale
  set tau 2 ;time steps of unsuccessful sale required to reduce price
  set nu 0.1 ;parameter quantifying the bargaining parameter of sellers

  set additional-flux false
  ;parameters for agents coming and leaving
  set alpha 0.1 ; probability to put the apartment on sale
  set gamma 500 ; number of incoming agents any time step
  set agents-list (list (5 / 10) (4 / 10) (1 / 10) )
  set xiklist (list (1) (1) (1))

  set initial-distribution [[310 690   0]
 [ 85 915   0]
 [851 149   0]
 [741  82 177]
 [995   5   0]]
  set filename-prices "traces/prices18.txt"
  set filename-agents "traces/distribution-agents18.txt"
  set filename-transactions "traces/transactions18.txt"
  set filename-buyers "traces/buyers18.txt"

  set endtransient 50

end


to setup-patches-turtles
  random-seed 3

  ;setup technical variables related to locations
  set number-of-agents number-of-apartments-per-location * (count patches)
  set initial-price 5 ;
  set max-attractiveness 1 ;

  ;setup of the agents
  let k 0
  let l 0
  repeat count patches [
    set k 0
  repeat number-of-categories [
      create-turtles item k (item l initial-distribution)
                                      [set reservation-price initial-price
                                        move-to patch l 0
                                        set income Y0 + k * delta
                                        set shape "person"
                                        set color grey
                                        set income-category k
                                        set state 1
                                        set time-born 0
      ]
      set k k + 1
  ]
    set l l + 1
  ]



  ;setup of the patches
  let initialize-price-proportional-to-income true

  let mean-income mean [income] of turtles
  ask patches [

    set mean-income-here mean [income] of turtles-here
    set market-price initial-price
    if initialize-price-proportional-to-income [
      set market-price mean-income-here
    ]
    set attractiveness max-attractiveness
    set utility-for-k-agents []
    set probability-for-k-agents []
    set housed-k-agents []
    set k 0
    repeat number-of-categories [
      set housed-k-agents lput [] housed-k-agents
      set housed-k-agents (replace-item k housed-k-agents (lput (count turtles-here with [income-category = k and state != 0]) (item k housed-k-agents) ) )
      set utility-for-k-agents lput  ( compute-utility ( Y0 + k * delta ) market-price (item k xiklist) attractiveness (mean-income-here / mean-income) ) utility-for-k-agents
      set probability-for-k-agents lput 0 probability-for-k-agents

      set k k + 1
      ]

    set all-market-prices []
    set prices-of-transactions-this-step []

  ]

  file-close-all
  if file-exists? filename-prices [file-delete filename-prices]
  if file-exists? filename-transactions [file-delete filename-transactions]
  if file-exists? filename-agents [file-delete filename-agents]
  if file-exists? filename-buyers [file-delete filename-buyers]

  reset-ticks

  print-agents-per-patch
  print-prices-per-patch
  print-transactions-per-patch
  print-buyers-per-patch

end

to setup

  setup-parameters
  setup-patches-turtles

end

to go

  ;create the agents at the beginning of each time step
  create-agents
  ;compute the probabilities for k-agents to visit any location
  compute-probabilities
  ;some agents decide to put their house on sale
  become-seller
  ;incoming agents search for an apartment with a probability proportional to the utility of going there
  choose-location
  ;matching happens through a continuos double auction
  do-continuos-double-auction
  ;computes "patch utility"
  update-utility

  ;removes the turtles which were not successful in finding an apartment
  ask turtles with [state = 0] [die]

  tick

  print-prices-per-patch
  print-transactions-per-patch
  print-agents-per-patch
  print-buyers-per-patch

  if ticks = 24 [stop]

end



to create-agents

  let k 0

  repeat number-of-categories [
    create-turtles gamma * (item k agents-list) [
      set income Y0 + k * delta
      set shape "person"
      set color grey
      set income-category k
      set xcor min-pxcor
      set ycor min-pycor
      set state 0
      set time-born ticks + 1
    ]
    set k k + 1
  ]


end


to compute-probabilities

  let normalizationk []
  let k 0

  repeat number-of-categories [
    set normalizationk lput sum ( [item k utility-for-k-agents] of patches ) normalizationk
    if item k normalizationk = 0 [set normalizationk replace-item k normalizationk 1]
    set k k + 1
  ]

  ask patches [
    set k 0
    repeat number-of-categories [
      set probability-for-k-agents replace-item k probability-for-k-agents ( item k utility-for-k-agents / ( item k normalizationk ) )
      set k k + 1
    ]
  ]

end

to become-seller

  ask turtles with [state = 1]    [
    let u random-float 1
    if u < alpha [set state 2 set time-on-sale ticks set reservation-price (1 + mu)*([market-price] of patch-here)]
  ]

end



to choose-location

 let k 0
 let L (count patches)

 repeat number-of-categories [

   let ordered-probabilities-list [] ;from topleft to topright, until bottomleft to bottomright
   let i 0
   repeat  (count patches) [
     set ordered-probabilities-list lput ([item k probability-for-k-agents] of patch ( min-pxcor + i mod L ) ( 0 ) )  ordered-probabilities-list
     set i i + 1
   ]

   let cumulated-list []
   set cumulated-list lput 0 cumulated-list
   set i 1
   repeat (length ordered-probabilities-list) [set cumulated-list lput (item (i - 1) ordered-probabilities-list + item (i - 1) cumulated-list) cumulated-list set i i + 1 ]

     ask turtles with [state = 0 and income-category = k] [  ;those looking for a new house
       let chosen-location-id inverse-transform cumulated-list
       set proposed-movement patch ( min-pxcor + chosen-location-id mod L ) ( 0 )
   ]
   set k k + 1

 ]

  if additional-flux [
    ask turtles with [state = 0 and income-category = k] [
      ;this is if the richest only look for a location in the center
      let chosen-location-id one-of patches with [pxcor ^ 2 + pycor ^ 2 <= 3 ^ 2]
      ;this is if the richest only look for a location in intermediate distance
      set proposed-movement chosen-location-id
]
  ]

end


to set-price
  if (ticks - time-on-sale) mod tau = 0 [set reservation-price reservation-price * lambda]

end


to do-continuos-double-auction

  ask patches [

   set prices-of-transactions-this-step []

   let buyers turtles with [proposed-movement = myself and state = 0]

   let sellers turtles-here with [state = 2]

   ;if self = patch 0 0 [print sort [income] of buyers print sort [reservation-price] of sellers]

   set nb count buyers
   set ns count sellers

   ask buyers [set reservation-price income / (item income-category xiklist)]
   ask sellers [set-price]


   if not any? sellers or not any? buyers [
     update-market-price
     stop]

   let logB []
   let logS []

   if not any? sellers or not any? buyers or max [reservation-price] of buyers < min [reservation-price] of sellers
     [update-market-price   stop]

   ask (turtle-set buyers sellers ) [


       let tmp[]
       set tmp lput reservation-price tmp
       set tmp lput who tmp

       ifelse state = 0 [
       set logB lput tmp logB
       set logB sort-by [[?1 ?2] -> item 0 ?1 > item 0 ?2] logB
       ]

       [
       set logS lput tmp logS
       set logS sort-by [[?1 ?2] -> item 0 ?1 < item 0 ?2] logS
       ]

     if (not empty? logB and not empty? logS) and item 0 (item 0 logB) >= item 0 (item 0 logS) [

       let transaction-price nu * (item 0 (item 0 logB)) + (1 - nu) * (item 0 (item 0 logS))


       let seller turtle (item 1 (item 0 logS))
       let buyer turtle (item 1 (item 0 logB))

       ask myself [
         set prices-of-transactions-this-step lput transaction-price prices-of-transactions-this-step
         if ticks > endtransient [set all-market-prices lput transaction-price all-market-prices]
       ]

       set logB but-first logB
       set logS but-first logS

       ask buyer [set state 1 move-to proposed-movement set reservation-price transaction-price]
       ask seller [die]
     ]


   ]


    update-market-price

  ]

end

to update-market-price  ;to be run by a patch


  if prices-of-transactions-this-step != []
    [set market-price (mean prices-of-transactions-this-step) ]
  ;if pxcor ^ 2 + pycor ^ 2 = 5  [file-open "price-record.txt" file-print market-price file-close]

end


to update-utility

  let mean-income mean [income] of turtles with [state != 0]

  ask patches [
    set mean-income-here mean [income] of turtles-here with [state != 0]

    let k 0
    repeat number-of-categories [
      set housed-k-agents (replace-item k housed-k-agents (lput (count turtles-here with [income-category = k and state != 0]) (item k housed-k-agents) ) )
      set utility-for-k-agents replace-item k utility-for-k-agents  ( compute-utility ( Y0 + k * delta ) market-price (item k xiklist) attractiveness (mean-income-here / mean-income) )
      set k k + 1
      ]
  ]

end





to-report compute-utility [Y PX xik A0 Ak]
  ifelse Y - xik * PX >= 0
    [ report (Y - xik * PX) ^ (1 - beta) * (A0 * Ak) ^ beta]
    [ report 0 ]
end


to-report inverse-transform [lista]

  let u random-float 1
  let l1 0
  let l2 length lista

  while [l2 - l1 > 1] [

    let l12  int ( ( l1 + l2 ) / 2 )
    ifelse u < ( item l12 lista ) [set l2 l12] [set l1 l12]

  ]

  report l1
end







to print-prices-per-patch
  file-open filename-prices
  let l 0
  let temp []
  repeat count patches [
    ;ifelse ([prices-of-transactions-this-step] of patch l 0) != [] [
    ;  set temp lput (mean [prices-of-transactions-this-step] of patch l 0) temp
    ;] [
    set temp lput ([market-price] of patch l 0) temp
    ;]
    set l l + 1
  ]
  file-print temp
  file-close
end

to print-transactions-per-patch
  file-open filename-transactions
  let l 0
  let temp []
  repeat count patches [
    set temp lput (length ([prices-of-transactions-this-step] of patch l 0)) temp
    set l l + 1
  ]
  file-print temp
  file-close
end


to print-agents-per-patch
  file-open filename-agents
  let l 0
  let temp []
  repeat count patches [
    let k 0
    repeat number-of-categories [
      set temp lput (count turtles with [income-category = k and xcor = l and ycor = 0]) temp
      set k k + 1
    ]
    set l l + 1
  ]
  file-print temp
  file-close
end

to print-buyers-per-patch
  file-open filename-buyers
  let l 0
  let temp []
  repeat count patches [
    let k 0
    repeat number-of-categories [
      ifelse ticks = 0 [
      set temp lput 0 temp
      ] [
      set temp lput (count turtles with [income-category = k and xcor = l and ycor = 0 and time-born = ticks]) temp
      ]
      set k k + 1
    ]
    set l l + 1
  ]
  file-print temp
  file-close
end
@#$#@#$#@
GRAPHICS-WINDOW
210
10
418
59
-1
-1
40.0
1
10
1
1
1
0
1
1
1
0
4
0
0
1
1
1
ticks
30.0

BUTTON
39
30
102
63
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
40
87
103
120
NIL
go
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
41
154
104
187
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

PLOT
170
94
534
327
states
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -2674135 true "" "plot count turtles with [state = 1]"
"pen-1" 1.0 0 -13791810 true "" "plot count turtles with [state = 2]"
"pen-2" 1.0 0 -16448764 true "" "plot count turtles with [state = 0]"

PLOT
164
336
537
543
market price
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -13791810 true "" "plot [market-price] of patch 0 0"
"pen-1" 1.0 0 -13345367 true "" "plot [market-price] of patch 1 0"
"pen-2" 1.0 0 -8630108 true "" "plot [market-price] of patch 2 0"
"pen-3" 1.0 0 -5825686 true "" "plot [market-price] of patch 3 0"
"pen-4" 1.0 0 -2064490 true "" "plot [market-price] of patch 4 0"

PLOT
559
10
874
185
social composition at 0 0
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -2674135 true "" "plot count turtles with [income-category = 0 and xcor = 0 and ycor = 0]"
"pen-1" 1.0 0 -13840069 true "" "plot count turtles with [income-category = 1 and xcor = 0 and ycor = 0]"
"pen-2" 1.0 0 -13345367 true "" "plot count turtles with [income-category = 2 and xcor = 0 and ycor = 0]"

PLOT
888
10
1201
186
social composition at 1 0
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -2674135 true "" "plot count turtles with [income-category = 0 and xcor = 1 and ycor = 0]"
"pen-1" 1.0 0 -13840069 true "" "plot count turtles with [income-category = 1 and xcor = 1 and ycor = 0]"
"pen-2" 1.0 0 -13345367 true "" "plot count turtles with [income-category = 2 and xcor = 1 and ycor = 0]"

PLOT
561
206
875
389
social composition at 2 0
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -2674135 true "" "plot count turtles with [income-category = 0 and xcor = 2 and ycor = 0]"
"pen-1" 1.0 0 -13840069 true "" "plot count turtles with [income-category = 1 and xcor = 2 and ycor = 0]"
"pen-2" 1.0 0 -13345367 true "" "plot count turtles with [income-category = 2 and xcor = 2 and ycor = 0]"

PLOT
888
206
1204
388
social composition at 3 0
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -2674135 true "" "plot count turtles with [income-category = 0 and xcor = 3 and ycor = 0]"
"pen-1" 1.0 0 -13840069 true "" "plot count turtles with [income-category = 1 and xcor = 3 and ycor = 0]"
"pen-2" 1.0 0 -13345367 true "" "plot count turtles with [income-category = 2 and xcor = 3 and ycor = 0]"

PLOT
560
404
873
584
social composition at 4 0
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -2674135 true "" "plot count turtles with [income-category = 0 and xcor = 4 and ycor = 0]"
"pen-1" 1.0 0 -13840069 true "" "plot count turtles with [income-category = 1 and xcor = 4 and ycor = 0]"
"pen-2" 1.0 0 -13345367 true "" "plot count turtles with [income-category = 2 and xcor = 4 and ycor = 0]"

@#$#@#$#@
## WHAT IS IT?

(a general understanding of what the model is trying to show or explain)

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.2.1
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
