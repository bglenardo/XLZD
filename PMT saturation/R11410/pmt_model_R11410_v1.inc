; pmt_model.inc

.param Vmin=1m
.param msc=2

.func vclamp(V) { max(V, Vmin) }

; Stage gain parameters
.param a   = 0.8113804291
.param k   = 0.09069767165
.param V_0 = 0

.func k_gen( V ) { k/sqrt(1+V_0/vclamp(V)) }
.func gain_0( V ) { k_gen(V) * vclamp(V)**a}

; Collection efficiency parameters -- single Vknee shared by every dynode gap including
; the anode (2026-08-16 decision: VkneeAnode/VkneeOther used to be separate, but sweeping
; VkneeAnode independently showed it has essentially no effect since the anode gap voltage
; barely sags under load, so there's no reason to fit it separately from the rest of the chain)
.param VkneeAnode = 35
.param VkneeOther = 35
.param Vslope = 5

.func eta( V, Vknee ) { 1/(1+exp((Vknee - max(V,0))/Vslope)) }

; Space charge parameters
.param IscDefault = 1e9
.param Isc8=0.8405294962
.param Isc9=0.2027021462
.param Isc10=0.1540646705
.param Isc11=0.1
.param Isc12=0.1
.param m =1.797281632

; .func space_charge(Iin, Isc) { 1 / (1 + (abs(Iin)/Isc)**m) }
.func space_charge(Iin, Vgap, Isc) { 1 / (1 + (abs(Iin*gain_0(Vgap))/Isc)**m) }

; Complete model of current behavior
; The idea here is that k, a, V_0, Vknee, Vslope, and m are parameters shared across all
; stages. The collection efficiency can be set to 1 by choosing "Vknee=-100", which
; guarantees that the exponent is very small. The space charge can be modified through
; changing the Isc parameter to the value relevant for that particular dynode; for most
; dynodes we give it 1e9 to guarantee that the correction factor is 1. So, only for the
; last few dynodes are the space-charge effects relevant, while collection efficiency
; (via the shared Vknee) applies to every stage.

.func g_effective(Vgap, Vcollect, Iin, Vknee, Isc) { gain_0(Vgap)*eta(Vcollect,Vknee)*space_charge(Iin,Vgap,Isc) }
