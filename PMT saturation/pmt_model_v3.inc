; pmt_model.inc

.param Vmin=1m
.param msc=2

.func vclamp(V) { max(V, Vmin) }

; Stage gain parameters
.param a   = 0.7885
.param k   = 0.1286
.param V_0 = 0

.func k_gen( V ) { k/sqrt(1+V_0/vclamp(V)) }
.func gain_0( V ) { k_gen(V) * vclamp(V)**a}

; Collection efficiency parameters
.param VkneeAnode = 5
.param VkneeOther = -100
.param Vslope = 2

.func eta( V, Vknee ) { 1/(1+exp((Vknee - max(V,0))/Vslope)) }

; Space charge parameters
.param IscDefault = 1e9
.param Isc8=150m
.param Isc9=60m
.param Isc10=20m
.param m =1.0

; .func space_charge(Iin, Isc) { 1 / (1 + (abs(Iin)/Isc)**m) }
.func space_charge(Iin, Vgap, Isc) { 1 / (1 + (abs(Iin*gain_0(Vgap))/Isc)**m) }

; Complete model of current behavior
; The idea here is that k, a, V_0, Vslope, and m are parameters shared across all stages.
; The collection efficiency can be set to 1 by choosing "VkneeOther=-100", which guarantees that
; the exponent is very small. The space charge can be modified through changing the Isc parameter to 
; the value relevant for that particular dynode; for most dynodes we give it 1e9 to guarantee
; that the correction factor is 1. So, only for the last few dynodes and anode are the collection
; and space charge effects relevant.

.func g_effective(Vgap, Vcollect, Iin, Vknee, Isc) { gain_0(Vgap)*eta(Vcollect,Vknee)*space_charge(Iin,Vgap,Isc) }
