function score=compute(I,T)
    patchSize=3;
    approximated = 1;
    conversion = @(I) I;
    fastDiversity = 1;
    h = 1;  % bendwidth parameter


% aproximated params for TreeCANN
    S_grid = 1;
    T_grid = 1;
    S_win = 3;    %must be odd
    T_win = 5;    %must be odd
    eps = 2;
    num_PCA_dims = 9;
    train_patches = 100;
    knn = 5;
    second_phase = 1;
    T = im2double(T);
    T = conversion(T);
    sT = size(T);
    T = im2uint8(T);
    I = im2double(I);
    I = conversion(I);
    [size11,size12,size13]=size(T);
    [size21,size22,size23]=size(I);
    row=max(size11,size21);
    column=max(size12,size22);
    T1=imresize(T,[row,column]);
    sT = size(T1);
    I=imresize(I,[row,column]);
    nnfApprox = zeros(size(I,1),size(I,2));
    nnf_dist = zeros(size(I,1),size(I,2));
    nnfExact = zeros(size(I,1),size(I,2));
    I = im2uint8(I);
    if approximated %using TreeCANN
        [nnf_dist_temp, nnf_X , nnf_Y] = run_TreeCANN(I,T1,patchSize,S_grid,T_grid,train_patches,num_PCA_dims,eps,knn,S_win,T_win,second_phase);
        nnf_X1 = nnf_X(1:end-patchSize+1,1:end-patchSize+1);
        nnf_Y1 = nnf_Y(1:end-patchSize+1,1:end-patchSize+1);
        %remove patchSize from end
        nnf_dist(1:end-patchSize+1,1:end-patchSize+1) = nnf_dist_temp(1:end-patchSize+1,1:end-patchSize+1);
        nnfApprox(1:end-patchSize+1,1:end-patchSize+1) = sub2ind(sT(1:2),nnf_Y1,nnf_X1);
        nnf=nnfApprox;
    end
    sT= double(sT(1:2));
    score=ComputeDDISForWindow(nnf,sT,h);
end