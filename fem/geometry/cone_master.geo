// Master mesh for all Fe-Cr compositions
lc_tip = 20;
lc_int = 40;
lc_far = 80;

Point(1) = {0, 0, 0, lc_tip};
Point(2) = {100, 0, 0, lc_tip};
Point(3) = {200, 0, 0, lc_int};
Point(4) = {500, 0, 0, lc_far};
Point(5) = {500, -500, 0, lc_far};
Point(6) = {0, -500, 0, lc_far};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};

Curve Loop(1) = {1, 2, 3, 4, 5, 6};
Plane Surface(1) = {1};

Physical Curve("contact") = {1};
Physical Curve("surface") = {2, 3};
Physical Curve("right") = {4};
Physical Curve("bottom") = {5};
Physical Curve("axis") = {6};
Physical Surface("substrate") = {1};

Mesh.ElementOrder = 1;
Mesh.Algorithm = 6;
Mesh.Optimize = 1;
Mesh.Smoothing = 3;
