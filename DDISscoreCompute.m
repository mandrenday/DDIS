function DDIS = DDISscoreCompute(nnf, sT, h, fastDiversity)
%DDISscore�������targetImg T��SourceImg I�е����ֵ����Ϊ�������������;�Ǽ򵥵�T��������I����ͬ��������С��I
%��������ֻ��Ҫʹ��һ��max������������С�ڣ���Ӧ��ʹ��˫��max
    DDIS = max(max(DDIS_nnf_scan_matlab(nnf, double(sT(1:2)),h)));
end

function [DDIS,DIS] = DDIS_nnf_scan_matlab(nnf,sT,h)

% Expand nnf
[mnnf,nnnf] = size(nnf);
mnnf = mnnf - sT(1) + 1;
nnnf = nnnf - sT(2) + 1;

% Create dis
rows = 0:(sT(1)-1);
cols = 0:(sT(2)-1);
DIS = zeros(mnnf,nnnf);
DDIS = zeros(mnnf,nnnf);

% Apply fun to each neighborhood of a
for j=1:nnnf
    for i=1:mnnf
%rows��cols�Ĵ�С�ֱ�Ϊtemplate�����д�С
%����nnfw���Ǵ�source����λ�ÿ�ʼ�ĸ���box����ס��subimage
%���Կ��Կ���������ComputeDDISForWindow���Ǽ������box��DDIS score
%�������ֻ�Ǽ��㵥����Ӧ�ķ�������ֱ����ComputeDDISForWindow�������nnfw��Ϊ��Ӧ��nnf
        nnfw = nnf(i+rows,j+cols);                
        [DDIS(i,j)] = ComputeDDISForWindow(nnfw,sT,h);
        DIS(i,j) = numel(unique(nnfw));
    end
end
end

function [DDIS]= ComputeDDISForWindow(nnfw,sT,h)
    [yDest,xDest] = ind2sub(sT, nnfw(:));%-patchSize+1
    [ySrc,xSrc] = ind2sub(sT, (1:(sT(1)*sT(2)))');%-patchSize+1 
    %����������Ҫ�����nnf�����СΪsT(1)*sT(2)����Ҳ����֪�����ڴ����ͼƬ��Ҫ����sourceImg����targetImg
    u = xDest-xSrc;
    v = yDest-ySrc;
    [~,r] = cart2pol(u,v);
    
    [uniqueIndices, chosenIndexesInNnfw,indTrasform] = unique(nnfw);          %provides sorted unique list of elements
    if numel(uniqueIndices) == 1
        useCountForUniqueIdx = numel(nnfw);
    else
        useCountForUniqueIdx = hist(nnfw(:) ,uniqueIndices)';    %provides a count of each element's occurrence
    end
    diversityPerPatch = useCountForUniqueIdx(indTrasform);
    DIw = exp((1-diversityPerPatch) / h);
    DDIS = sum( DIw' ./ (r'+1) );

end