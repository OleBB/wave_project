%Postprosessering av master-data paa nytt. 
%Samme program som med tilsvarende tittel i mappa "Master_PhDtillegg".
%RENSER FOR DROPOUTS FØR(!) SENTRERING RUNDT MEAN.

%Tar inn raafil med 6 kolonner: 
%[dato og tid , probe1, probe2, probe3, probe4, lydhastighet]
%Returnerer filen probe_runX_renset_sentrert (X er run nr) med 4 kolonner:
%[probe1, probe2, probe3, probe4]

clear

phd_settings

for run_nr = 3:53 %gaar gjennom alle runs
    %Laster inn fil (fra mastermaalinger):
    datafile = sprintf('%s/Kjoring/probe_run%d.csv', kadingirKarenroot, run_nr);
    fid = fopen(datafile, 'r');
    data = textscan(fid, '%s %f %f %f %f %f', 'Delimiter', ',');
    fclose(fid);
    run = cell2mat(data(2:end)); %Tar vekk kol. med dato+klokkeslett

    overflate = zeros(length(run(:,1)),4);
    for probe = 1:4 %gaar gjennom alle prober

        %SNUR DATA:
        overflate(:,probe) = -run(:,probe);

        %DROPOUTS:
        eta_in = overflate(:,probe);
        Threshold = 0.0009; %stigningstall-grense
        eta_out=InterpolateSpikes(eta_in,Threshold);
    
        %SENTRERER OM MEAN:
        likevekt = mean(eta_out); %finner overflaten som tilsvarer 0,52m dybde
        overflate(:,probe) = eta_out-likevekt; %finner overflaten ifht likevektsdybden
    end
    
    %Lagrer (i phd-mappa paa kadingir):
    filename = sprintf('%s/Overflatedata/probe_run%d_renset_sentrert',kadingirNyPostprosroot,run_nr); 
    save(filename, 'overflate')
end