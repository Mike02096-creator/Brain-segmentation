clear all
close all

I = imread('CT_SubCortInf.jpg');
o=figure;
imshow(I);
title ("Tête entière avec axe de symétrie sur le cerveau");

% Coordonnées du centroîd
x1 = 309.8705;
y1 = 371.2801;
axis ;
hold on;
scatter(x1,y1);
plot([x1 290],[y1 750]);
plot([x1 330.741],[y1 0]);

J = rgb2gray(I);

tableau = (zeros(size(I,1),size(I,2)));
for i= 1:size(J,1)
    for j= 1:size(J,2)
        if I(i,j)>200
            tableau(i,j) = I(i,j);
        end
    end
end

%tableau = tour de tete
tableau = im2uint8(tableau);
se = strel('disk',5);
background3 = imopen(tableau,se);
figure;
imshow(background3,[]);
title('Boite cranienne');

%Affichage du cerveau seul

sou = I - tableau;
se = strel('disk',10);
background = imopen(sou,se);
%figure;
%imshow(background,[]);
%title("Cerveau seul");

test = imbinarize(background);
test = im2uint8(test);
se = strel('disk',5);
background4 = imopen(test,se);
%figure;
%imshow(test)
%title('test')

%figure;
test2 = ~test;
test2 = im2uint8(test2);
%figure
%imshow(test2)
se = strel('disk',15);
background10 = imopen(test2,se);
%figure;
%imshow(background10)
h = im2uint8(background10);
Cerveau = J-h;
figure;
imshow(Cerveau);
title('Cerveau seul');

BW = imbinarize(J,'adaptive','Sensitivity',0);
%figure
%imshow(BW)
%title("Image binarisée")

%se = strel('disk',5);
%background = imopen(BW,se);
%figure
%imshow(background,[])
%title("Boite cranienne")

BW2 = imbinarize(J,'global');
%figure
%imshow(BW2)

BW3 = imfill(BW2,'Holes');
%figure
%imshow(BW3)

se2 = strel('disk',5);
background2 = imopen(BW3,se);
figure;
%imshow(background2,[]);
title('Tête complète');

%BW4 = background2 - background
%figure
%imshow(BW4)

%se = strel('disk',8);
%background3 = imopen(BW4,se);
%figure
%imshow(background3,[])
%title("Cerveau")


%Centroîd du cerveau
f = regionprops('table',J,'Eccentricity','Area','Centroid')
p = cat(1,f.Centroid);
x = p(1);
y = p(2);
r=[x y]
scatter(x,y)


%K-means
[L,Centers] = imsegkmeans(Cerveau,4);
B = labeloverlay(Cerveau,L);
figure
imshow(B)
legend('Matière grise','Matiere blanche')
title('Labeled Image : K-means (en rouge : matière blanche et en bleu clair : matière grise)')



lab_I = rgb2lab(I);
ab = lab_I(:,:,1:3);
ab = im2single(ab);
nColors = 3;
% repeat the clustering 3 times to avoid local minima
pixel_labels = imsegkmeans(ab,nColors,'NumAttempts',5);

mask1 = pixel_labels==1;
cluster1 = Cerveau .* uint8(mask1);
%figure
%imshow(cluster1)
%title('Objects in Cluster 1 : matiere blanche');

mask2 = pixel_labels==2;
cluster2 = Cerveau .* uint8(mask2);
%figure
%imshow(cluster2)
%title('Objects in Cluster 2 : matiere grise');

mask3 = pixel_labels==3;
cluster3 = Cerveau .* uint8(mask3);
l = rgb2gray(cluster3);
k = imbinarize(l)
%figure
%imshow(k)
%title('Objects in Cluster 3 : Liquide cerebrospinal');
L = lab_I(:,:,1);
L_blue = L .* double(mask3);
L_blue = rescale(L_blue);
idx_light_blue = imbinarize(nonzeros(L_blue));

BWoutline = bwperim(cluster3);
Segout = Cerveau;
Segout(BWoutline) = 255;
figure
imshow(Segout);
title('Liquide cerebrospinal')

%Rotation de l'image
m = imrotate(Segout,90);
figure
imshow(m)
title('Rotation Cerveau')

new = rgb2gray(B);
%figure;
%imshow(new);
%title('Matiere grise')

kim = imadjust(new,[0.5 0.9],[])
%figure
%imshow(kim)

lom = imbinarize(kim);
figure;
imshow(lom);
title('Matière grise binarisé');

lom2 = ~lom;
lom2 = im2uint8(lom2);
figure;
imshow(lom2);
title('Matière grise en binairisation inversée');

se = strel('disk',15);
background25 = imopen(lom2,se);
hop = im2uint8(background25);
Matieregrise = J-hop;
figure;
imshow(Matieregrise);
title('AVC');

d = imdistline;
delete(d);
figure;
b=bwboundaries(Matieregrise);
imshow(Matieregrise);
title('Detection de l''AVC');
hold on;
for i=1:length(b);
    contour=b{i};
    plot(contour(:,2),contour(:,1),'g','LineWidth',3);
end

BWoutline = bwperim(lom2);
Segout = Cerveau;
Segout(BWoutline) = 255;
figure
imshow(Segout);
title('Matière blanche')

BWoutline = bwperim(kim);
Segout = Cerveau;
Segout(BWoutline) = 255;
figure
imshow(Segout);
title('Matière grise')

