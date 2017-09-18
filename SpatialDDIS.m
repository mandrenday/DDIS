clear all;
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
DDISscore=zeros(200,200);
size1=[0,32,64];%设置各个部分的范围
weights=[1,3];%设置各个spatial的权值
for i =1:200
    img=imread(strcat('D:\Documents\Tencent Files\1432874755\FileRecv\目标图库\',num2str(i),'.png'));
    img=imresize(img,[64,64]);
    display(i);
    T = im2double(img);
    T = conversion(T);
   %sT = size(T);
    T = im2uint8(T);
    L1=Colorhistogram(T);
    [m,n]=sort(L1,'descend');
    a=sort(n(1:7));
    a1=zeros(1,256);%由Colorhistogram计算得到的L1的大小为1*256；
    a1(a)=1;
    for j =1:200
        img=imread(strcat('D:\Documents\Tencent Files\1432874755\FileRecv\待检测图库\',num2str(j),'.png'));
        img=imresize(img,[64,64]);
        I = im2double(img);
        I = conversion(I);
        I = im2uint8(I);
        L2=Colorhistogram(I);
        [m1,n1]=sort(L2,'descend');
        b=sort(n1(1:10));
        k=[b,b+1,b+2,b-1,b-2];
        k1=k(k>0 & k<257);
        b1=zeros(1,256);
        b1(k1)=1;
        a2=a1-b1;
        b3=a2(a2==1);
        num11=sum(b3);
        if(num11<4)
            nnfApprox = zeros(size(I,1),size(I,2));
            nnf_dist = zeros(size(I,1),size(I,2));
            nnfExact = zeros(size(I,1),size(I,2));
            score=0;
            for k=1:2
                if(k==2)
                    T1=T(size1(k)+1:size1(k+1),1:50,:);
                else
                    T1=T(size1(k)+1:size1(k+1),:,:);
                end
                sT=size(T1);
                if approximated %using TreeCANN
                    [nnf_dist_temp, nnf_X , nnf_Y] = run_TreeCANN(I,T1,patchSize,S_grid,T_grid,train_patches,num_PCA_dims,eps,knn,S_win,T_win,second_phase);
                    nnf_X1 = nnf_X(1:end-patchSize+1,1:end-patchSize+1);
                    nnf_Y1 = nnf_Y(1:end-patchSize+1,1:end-patchSize+1);
                %remove patchSize from end
                    nnf_dist(1:end-patchSize+1,1:end-patchSize+1) = nnf_dist_temp(1:end-patchSize+1,1:end-patchSize+1);
                    nnfApprox(1:end-patchSize+1,1:end-patchSize+1) = sub2ind(sT(1:2),nnf_Y1,nnf_X1);
                    nnf=nnfApprox;
                else
                    nnf_XYD = ENN_matching(I, T1, patchSize);
                    nnf_X2 = nnf_XYD(1:end-patchSize+1,1:end-patchSize+1,1)+1;
                    nnf_Y2 = nnf_XYD(1:end-patchSize+1,1:end-patchSize+1,2)+1;
                    nnf_dist_temp = nnf_XYD(1:end-patchSize+1,1:end-patchSize+1,3)+1;
                %remove patchSize from end
                    nnf_dist(1:end-patchSize+1,1:end-patchSize+1) = nnf_dist_temp(1:end-patchSize+1,1:end-patchSize+1);
                    nnfExact(1:end-patchSize+1,1:end-patchSize+1) = sub2ind(sT(1:2),nnf_Y2,nnf_X2);
                    nnf=nnfExact;
                end
                sT= double(sT(1:2));
                score=score+weights(k)*(DDISscoreCompute(nnf,sT,h,fastDiversity));
            end
            DDISscore(i,j)=score/4;
        else
            DDISscore(i,j)=0;       
        end   
    end
end