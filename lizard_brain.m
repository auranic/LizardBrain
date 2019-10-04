function [data,v,u,s] = lizard_brain()

number_of_branches = 10;
dimension = 10;
epsilon = 0.005;
add_noise = 0.001;
min_branch_points = 50;
k_forknngraph = 8;

make_knn_graph = 0;

x0 = zeros(1,dimension);
i1 = 1;
i2 = 2;
branch = [];
while size(branch,1)<min_branch_points
x0(i1) = rand();
x0(i2) = rand();
branch = make_branch(x0,i1,i2,epsilon);
end

data = branch;

k = 0;
while k<=number_of_branches
n = floor(size(branch,1)/2);
%x0 = branch(n,:);
x0 = data(floor(rand()*size(data,1)),:);

i1 = floor(rand()*dimension+1);
i2 = floor(rand()*dimension+1);
while(i2==i1)
    i2 = floor(rand()*dimension+1);
end
disp(sprintf('Dim (%i,%i)',i1,i2));

[newbranch] = make_branch(x0,i1,i2,epsilon);
n1 = size(data,1);
n2 = size(newbranch,1);
if n2>min_branch_points-1
data(n1+1:n1+n2,:) = newbranch(:,:);
branch = newbranch;
k = k+1;
end
% plot(branch(:,1),branch(:,2),'ko'); hold on;
%  plot([x0(:,1) x0(:,1)+v1(:,1)/20],[x0(:,2) x0(:,2)+v1(:,2)/20],'b-');
%  plot([x0(:,1) x0(:,1)+v2(:,1)/20],[x0(:,2) x0(:,2)+v2(:,2)/20],'b-');
end

if add_noise>0
    data = data + rand(size(data,1),size(data,2))*add_noise;
end

[v,u,s] = pca(data);
plot(u(:,1),u(:,2),'ko','MarkerSize',2); hold on; drawnow;
xlabel(sprintf('PC1 %2.2f%%',s(1)/sum(s)*100));
ylabel(sprintf('PC2 %2.2f%%',s(2)/sum(s)*100));
axis equal;

if make_knn_graph

knngraph = knnsearch(data,data,'k',k_forknngraph);
fid = fopen('knn1.sif','w');
for i=1:size(knngraph,1)
    for k=2:size(knngraph,2)
        fprintf(fid,'%i\tna\t%i\n',knngraph(i,1),knngraph(i,k));
        %plot([u(knngraph(i,1),1) u(knngraph(i,k),1)],[u(knngraph(i,1),2) u(knngraph(i,k),2)],'b-','MarkerSize',2); hold on;
    end
    %drawnow;
end
fclose(fid);

end

end

function [branch] = make_branch(x0,i1,i2,epsilon)
dimension = size(x0,2);
v1 = zeros(1,dimension);
v2 = zeros(1,dimension);
v1(i1) = rand()-0.5; 
v1(i2) = rand()-0.5; 
v1 = v1/norm(v1);
v2(i1) = -v1(i2); 
v2(i2) = v1(i1); 

[branch] = parabolic_branch(x0,v1,v2,epsilon);
end

function [x] = parabolic_branch(x0,v1,v2,epsilon)
x = [];
t = epsilon/1000;
i = 1;
irx1 = find(v1~=0);
irx2 = find(v2~=0);
    while 1
      xn = x0+t*v1+t*t*v2;
      if (max(xn(irx1))<1)&&(min(xn(irx2))>0)
          x(i,:) = xn;
          i = i+1;
          t = t+epsilon;
      else
          break;
      end
    end
end
