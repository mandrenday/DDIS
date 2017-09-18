clear all
LengthFiles = 200;
LengthFiles1 = 200;
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
DDISscore=zeros(LengthFiles,LengthFiles1);
for i = 1:LengthFiles
    display(i);
    targetImg = imread(strcat('D:\Documents\Tencent Files\1432874755\FileRecv\Ä¿±êÍ¼¿â\',num2str(i),'.png'));
    T=targetImg;
    T = im2double(T);
    T = conversion(T);
   %sT = size(T);
    T = im2uint8(T);
    T1=imresize(T,[64,64]);
    sT = size(T1);
    for j=1:LengthFiles1
        sourceImg=imread(strcat('D:\Documents\Tencent Files\1432874755\FileRecv\´ý¼ì²âÍ¼¿â\',num2str(j),'.png'));
        I=sourceImg;
        I = im2double(I);
        I = conversion(I);
        I=imresize(I,[64,64]);
       %% first step - NN fiels
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
        else
            nnf_XYD = ENN_matching(I, T, patchSize);
            nnf_X2 = nnf_XYD(1:end-patchSize+1,1:end-patchSize+1,1)+1;
            nnf_Y2 = nnf_XYD(1:end-patchSize+1,1:end-patchSize+1,2)+1;
            nnf_dist_temp = nnf_XYD(1:end-patchSize+1,1:end-patchSize+1,3)+1;
            %remove patchSize from end
            nnf_dist(1:end-patchSize+1,1:end-patchSize+1) = nnf_dist_temp(1:end-patchSize+1,1:end-patchSize+1);
            nnfExact(1:end-patchSize+1,1:end-patchSize+1) = sub2ind(sT(1:2),nnf_Y2,nnf_X2);
            nnf=nnfExact;
        end
        sT= double(sT(1:2));
        DDISscore(i,j)=ComputeDDISForWindow(nnf,sT,h);
    end
end