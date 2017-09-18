function [DDIS]= ComputeDDISForWindow(nnfw,sT,h)
    [yDest,xDest] = ind2sub(sT, nnfw(:));%-patchSize+1
    [ySrc,xSrc] = ind2sub(sT, (1:(sT(1)*sT(2)))');%-patchSize+1
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