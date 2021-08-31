function linRGB = PQEOTF(E)

m1 = 0.1593017578125;
m2 = 78.84375;
c1 = 0.8359375;
c2 = 18.8515625;
c3 = 18.6875;

linRGB = 10000 * ((max((E.^(1/m2))-c1, 0))./(c2-c3*(E.^(1/m2)))).^(1/m1);
linRGB = ClampImg(linRGB, 0, 10000);

end

addpath('./hdrvdp-3.0.6/');

LQ_img_path = '...';
GT_img_path = '...';
LQ_img = imread(LQ_img_path);
GT_img = imread(GT_img_path);
LQ_img = im2double(LQ_img);
GT_img = im2double(GT_img);
LQ_img = PQEOTF(LQ_img);
GT_img = PQEOTF(GT_img);

res = hdrvdp3('side-by-side', LQ_img, GT_img, 'rgb-bt.2020', ppd, {'rgb_display', 'led-lcd-wcg'});

output = res.Q