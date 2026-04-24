close all
clearvars


% isotropic surfaces 
% N = 8;
% l = 513;
% H1 = 0.8;
% H2 = 0.8;
%geon = 50;

% anisotropic surfaces
N = 8;
l = 513;
H1 = 0.6;
H2 = 0.8;
geon = 20;

zamps = [10 160 280 400];

amins = [1 2 5 10];

rmsl = nan(length(zamps), geon);

for ai=1:length(zamps)
    zamp = zamps(ai);

    psurf = ['l' num2str(l) 'H' num2str(H1*10) num2str(H2*10) 'A' num2str(zamp)];
    mat_save = ['mats/synf_unm_uz_' psurf 'g' num2str(geon) '.mat'];
    
    di=1;
    bst = nan(1e6, 17);
    
    ri=1;
    rngs = nan(1e6, 5);
    
    fils = {}; 
    gi=1;
    while gi<=geon
        %pstrg = [psurf 'g' num2str(gi)];
    

        fault = Synthetic2DFault(N, H1, H2);
        dim = size(fault);
        xd = dim(1);
        [x, y] = meshgrid((1:xd), (1:xd));  

        xsub = x;
        ysub = y;

        xsub = xsub-min(xsub(:))+1;
        ysub = ysub-min(ysub(:))+1;
    
        z = fault;
        zmin = min(z(:));
        zmax = max(z(:));
        zn = (fault-zmin)./(zmax-zmin);
        zno1 = zn.*zamp;
    
        n = length(zno1(:));
        mu = mean(zno1(:));
        delz = zno1(:)-mu;
        rms = sqrt((1/n)*sum(delz.^2));

        rmsl(ai, gi) = rms/xd;

        fault2 = Synthetic2DFault(N, H1, H2);
        z = fault2;
        zmin = min(z(:));
        zmax = max(z(:));
        zn2 = (fault2-zmin)./(zmax-zmin);
        zno2 = zn2.*zamp;
    
        zslipc = zno2;
        zbot = zno1;
        ztopc = zslipc+zamp;

        zdels = linspace(0.4*zamp, 0.9*zamp, 30);

        % figure(1)
        % subplot(2,1,1)
        % hold on
        % 
        % surf(zbot, 'EdgeColor', 'none')
        % 
        % axis equal tight
        % colorbar
        % %clim([0 35])
        % 
        % subplot(2,1,2)
        % hold on
        % 
        % surf(ztopc, 'EdgeColor', 'none')
        % 
        % axis equal tight
        % colorbar
        % %clim([0 35])
        % return

        for zi=1:length(zdels)
            zdel = zdels(zi);

            pstrc = [psurf 'geo' num2str(di)];
            
            fname = ['txt/bm_syn_unm_gmax' num2str(geon) '_' pstrc '.txt'];
            fnamep = ['txt/bm_syn_unm_perpen_gmax' num2str(geon) '_' pstrc '.txt'];
            
            fils{di} = pstrc;

            ztop = ztopc-zdel;
            
            contact = zeros(size(zbot));
            icon = ztop<=zbot;
            contact(icon) = 1;
            conv = contact(:);
            nval = numel(conv);
            contpt = sum(conv);
            ar = 100*(contpt/nval);
            
            bs = ztop-zbot;
            bps = bs;
            bps(bps<0) = 0; % make the negative aperture equal to zero aperture

            bmu = mean(bps(:));
            bstd = std(bps(:));

            cc = bwconncomp(contact);
            p = regionprops('table', cc, 'Area', 'EquivDiameter', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'PixelList');
            
            ares = p.Area;
            d = [ares (1:length(ares))'];
            dsort = sortrows(d, 1, 'descend');

            ilg = ares>=amins(1);
            alg = d(ilg, :);

            nlg = length(alg(:,1));

            props = ones(1, 9);
            if nlg>0
                sizc = size(contact);

                sta = nan(nlg, 3);
                sta2 = nan(nlg, 2);
                sta3 = nan(nlg, 2);
                sta4 = nan(nlg, 2);
                pri=0;
                for ci=1:nlg
                    ii = alg(ci, 2);
    
                    a = p(ii, :).Area;
                    
                    peri = p(ii, :).Perimeter;
                    de = p(ii, :).EquivDiameter;

                    % ratio of true perimeter/equivalent perimeter
                    % if a tortuos shape, then number is higher
                    eqperi = (de*(22/7));
                    peq = peri/eqperi;

                    alpha = (p(ii, :).MinorAxisLength)/(p(ii, :).MajorAxisLength);

                    if peq<1
                        peq = 1;
                    end

                    sta(ci, :) = [peq alpha a];

                    if a>amins(2)
                        sta2(ci, :) = [peq alpha];
                    end
                    if a>amins(3)
                        sta3(ci, :) = [peq alpha];
                    end
                    if a>amins(4)
                        sta4(ci, :) = [peq alpha];
                    end
                end

 
                sta2 = sta2(~isnan(sta2(:,1)), :);
                p2 = [1 1];
                if ~isempty(sta2)
                    p2 = [mean(sta2(:,1)) mean(sta2(:,2))];
                end
                sta3 = sta3(~isnan(sta3(:,1)), :);
                p3 = [1 1];
                if ~isempty(sta3)
                    p3 = [mean(sta3(:,1)) mean(sta3(:,2))];
                end
                sta4 = sta4(~isnan(sta4(:,1)), :);
                p4 = [1 1];
                if ~isempty(sta4)
                    p4 = [mean(sta4(:,1)) mean(sta4(:,2))];
                end
                

                props = [mean(sta(:,1)) mean(sta(:,2)) p2 p3 p4 mean(sta(:,3))];

            end

            % % reverse the order so that axes are as expected in the python script
            binv = bps';
            dat = [xsub(:) ysub(:) binv(:)];
            writematrix(dat, fname)
            disp(fname)

            if H2>H1
                dat = [xsub(:) ysub(:) bps(:)];
                writematrix(dat, fnamep)
                disp(fnamep)
            end

            %             1  2   3    4            5  6   7    8   
            bst(di, :) = [gi zi zdel mean(ztop(:)) ar bmu bstd bmu/bstd props];
            di=di+1;

        end

        stcurr = bst(di-length(zdels):di-1, :);

        % rngs, 2-3= ar, get max ar=40
        % rngs, 4-5= bmu/bstd, get lower ranges of bm/bstd
        rngs(ri, :) = [gi min(stcurr(:,5)) max(stcurr(:,5)) min(stcurr(:,8)) max(stcurr(:,8))];
        ri=ri+1;

        arrng = rngs(ri-1, 2:3);
        bmstdrng = rngs(ri-1, 4:5);

        gi=gi+1;

    end



    bst = bst(~isnan(bst(:,1)), :);
    rngs = rngs(~isnan(rngs(:,1)), :);

    ntotgeos = length(bst(:,1));


    save(mat_save, 'bst', 'rngs', 'fils')
    disp(mat_save)

end

