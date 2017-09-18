function DDIS = DDISscoreCompute(nnf, sT, h, fastDiversity)
%DDISscore计算的是targetImg T在SourceImg I中的最大值，因为我们是这里的用途是简单的T的列数与I的相同，而行数小于I
%所以我们只需要使用一个max，若是两个都小于，则应该使用双重max
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
%rows和cols的大小分别为template的行列大小
%所以nnfw就是从source各个位置开始的各个box所框住的subimage
%所以可以看出来就是ComputeDDISForWindow就是计算各个box的DDIS score
%所以如果只是计算单个对应的分数，就直接用ComputeDDISForWindow，传入的nnfw即为对应的nnf
        nnfw = nnf(i+rows,j+cols);                
        [DDIS(i,j)] = ComputeDDISForWindow(nnfw,sT,h);
        DIS(i,j) = numel(unique(nnfw));
    end
end
end

function [DDIS]= ComputeDDISForWindow(nnfw,sT,h)
    [yDest,xDest] = ind2sub(sT, nnfw(:));%-patchSize+1
    [ySrc,xSrc] = ind2sub(sT, (1:(sT(1)*sT(2)))');%-patchSize+1 
    %上面这两步要求传入的nnf必须大小为sT(1)*sT(2)，则也可以知道对于传入的图片的要求是sourceImg大于targetImg
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