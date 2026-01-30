 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 19:08:31 2025

@author: ole
"""

"""
#pseudo-kode for iterativ DATANALYSE. 

lese inn mappene. 
lese opp filene. 

velge å behandle utvalge filer, slik at jeg kan ha de i Pandas-minnet.

bare behandle de utvalgte filene.

1. lese inn signalet. 
2. finne stilltillstand (i begynnelsen eller i referansedokument)
3. smoothe signalet. (?) Knapp[av/på]
4. finne topper og bunner
5. avkorte signalet (basert på smart matematisk modell?)
6. lese av (digitalt) amplitude 
7. regne ut andre
8. plotte. 
    bonus - plotte avkortet signal oppå hele signalet.
9.lagre plot og tabell

så på nytt, jeg ønsker å variere plottet, og variere beregnet output (fudge?) derfor må hele tidsserien være i minnet.
slik at jeg kan printe deler, gå tilbake og printe andre deler.

"""


""" STILLTILSTAND
finne stilltilstand - bruke den jeg allerede har.
"""

""" GLATTE - MOVING AVERAGE
Glatte ut signalet.
"""

"""AMPLITUDE - TOPP OG BUNN
Finne amplituden ved å plukke ut 5-10 topper, og regne fra bunnen før dem?.
Regne Hs?

"""



""" KLIPPE KLIPPE KLIPPE
klippe til signalet... først å fremst kan jeg gjøre manuelt, nå som jeg vet at alle dagens kjøringer er på 15 sek.

smart matematisk modell - parametere å jobbe ut i fra:
    filstørrelse (indikativ lengde)
    lengde på hele signalet (1 kolonne trengs)
    input frekvens, input amplitude utregnet hastighet. 
    mstop #kutte fra mstop og bakover? avhengig av utregnet gruppehastighet.
"""

