Ny høgde. 100. evt. også ny lateral-posisjon på probe 1. .. nå er den vel 3-5mm nærmere midten...pga at den gule probeholdern til Olav bygger utover.

en nowave. en run amp0100-fre1300 for good-measure. 

så, amp0300-freq1500 - bølgen i front er brytende.. resten klarer seg.  

okei. 

løfta takplate og fiksa panel, men lot vinden stå på. interessant. pitot, som er helt foran ved inngangen til vinden viser 28-29 pa. fra pitottup til bakre probe er det 59cm. dette 
Nu er det egentlig for å se om vi ser noe spennende mtp vind. de fremre probene viser faktisk lavere vannstand på 102-103. Må sjekke stilltilstand med en gang. Flytter alt vannet i tanken seg bak? 

"experimental-4roof50cm-fraMaxtil0tilMaxwin-depth580-mstop30-run1"

wat! probe 1 droppa ut igjen! nappe den ut funka ikke. 

Dro på masterseminar.
Nå er hele boksen på 0.198 igjen! Helt feil!!
Nappa ut alle 4 probeledningene fra boksen - nå funker de. 

Ny stilltilstandsmåling gjort.

Neste, coldstart amp0100-freq1300-per240. 
så, en rekke med amp0100-freq13-4-5-6-7

Så fylle inn noen amp0200-freq17-8 som ikke ble gjort i stad. 
fortsatt svære dropouts på probe 1. Ok, men det som er interessant er at amp0200-freq1800-per40 er ikke veldig representativ bølge. den starter med en brytende i front. så ender den dårlig, ser ut til å miste kraft, kanskje pga refleksjon?

ja, her er det definitivt refleksjon. men den ser nesten ikke ut til å nå frem til fremste probe, fordi panelet er jo utstrukket typ 20 cm fra parallellprobene 9373, men den refleksjonen når altså ikke frem til "8805-proben"

WOW! I bølge amp0300-freq1800 ser faktisk en artefakt av refleksjon RETT FØR TOPPEN når frem til 9373!!. Men det ser ut som det trenger litt tid på seg. la meg ta en lang serie.. kanskje ikke fullt 240, hva med 80?. 
Okei, amp0300-freq1600-per80 ser ut som fremste probe 8805 gir dropouts... men ser faktisk også ut som refleksjonen bygges opp.  
La oss prøve ett hakk ned. 
bølgen amp0300-freq1600-per80. filmet i slow mo og vanlig. 

Nå, på med vind. vente noen minutt. sample 6 minutter. sjekke at alt er ok. 
Første problem - ser ut som pitotrøret bare viser 19-20 Pa... skulle vært oppe i 21 helst. 
Når vinden står på så er de første 6-7 platene under vann. dypere enn hviletilstanden uten vann. 

wind-only run done. 
nå blir det standardbølgen amp0100-freq1300. 

er jo litt interessant å se at det ser ut som bølgen bak panelet rykker til. panelet rykker jo til, men gjør egentlig bølgen det?

fylt på fullwind-amp0100 målinger fra i stad. 
nå er det fullwind-amp0200 målinger. fra 1300-1800. 

USELESS: fullpanel-fullwind-amp0200-freq1500-per40-depth580-mstop30-run1-P2malfunction.

okei, funker nå. 
i bølgen amp0200-freq1600-per40 synes jeg å se refleksjon fra panelet. og nevner at bølgen ser firkanta ut

fullwind-amp0200-freq1800-per40 er jo faktisk en brytende bølge... 

kjører amp0300-serie nå. 

Arg! channel 2 er NAN! den er strengt nødvendig. Må ta 1400 og 1500 på nytt. kaller de for run 2. 

disse bølgene er jo i grenseland brytende..
nå, på amp0300-freq17/18 har vi mulig brytning. ..

tror jeg vil se mere av amp0200. 

fascinerende, på ett vis er jo de amp0300 steile bølgene mere uniforme, fordi de bestemmer over vinden...

oppå denne, amp0200-freq1500, så ser vi en liten minibølgetopp som reiser oppå den store bølgen..kan bli veldig rare data.

kjører en laang serie med amp0200-freq1600, og her er det jo en minibølge som reiser på topppen.

vinden er nede i 18 og oppei 20.. må sjekke etter på hva som er null. 

nå, på amp0200-freq1400-per240 så er det tydelig at vindbølgene får reise mere sidelengs, overflaten er jo mere ujevn.. 

en eksperimentkjøring til der jeg skrur av vinden og måler 10minutter.

steikje. nå, nesten 10 minutt senere ser jo ut som det er bevegelse oppstrøms ved probene der, etter jeg nå skrudde av vinden for disse 10 minuttene siden. 
uh-oh - labview -error, den sier at samples requestted have not yet been aquired. men ULS holder på fortsatt. muligens bølgemakern som lager kvalm? 
aha. mitt forsøk på å måle fra MaxToZeroWin(d) ble avbrutt. jeg fikk heller bare en dårlig stillvannstillstand.

fullpanel-nowind-ULSonly-withLabviewError er altså med avslått pumpe mot slutten av kjøringen. 

nå sliter ULS med at den fortsatt har en labview-trigger gående. uavhengig av at jeg har kjørt min egen sample 
