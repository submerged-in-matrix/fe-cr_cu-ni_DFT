// Conical indenter nanoindentation — axisymmetric substrate
// Units: nm
// Half-included angle: 70.3 degrees (Berkovich-equivalent)
// Substrate: cylinder R=500nm, H=500nm

// Mesh size parameters
lc_tip = 2;    // near contact (nm)
lc_far = 25;   // far field (nm)

// Substrate corner points (r-z plane, z pointing up, surface at z=0)
Point(1) = {0,   0,  0, lc_tip};  // origin — symmetry axis, surface
Point(2) = {100, 0,  0, lc_tip};  // surface, near contact zone edge
Point(3) = {500, 0,  0, lc_far};  // surface, far edge
Point(4) = {500, -500, 0, lc_far}; // bottom right
Point(5) = {0,   -500, 0, lc_far}; // bottom left (axis)

// Lines
Line(1) = {1, 2};   // surface inner (contact zone)
Line(2) = {2, 3};   // surface outer
Line(3) = {3, 4};   // right side
Line(4) = {4, 5};   // bottom
Line(5) = {5, 1};   // symmetry axis (r=0)

// Surface
Curve Loop(1) = {1, 2, 3, 4, 5};
Plane Surface(1) = {1};

// Physical groups — these become node/element sets in the mesh
Physical Curve("contact", 1)  = {1};   // inner surface (indenter acts here)
Physical Curve("surface", 2)  = {2};   // outer surface (free)
Physical Curve("right", 3)    = {3};   // right boundary
Physical Curve("bottom", 4)   = {4};   // fixed bottom
Physical Curve("axis", 5)     = {5};   // symmetry axis r=0
Physical Surface("substrate", 6) = {1};

// Second-order elements for better contact accuracy
Mesh.ElementOrder = 2;
Mesh.Algorithm = 6;  // Frontal-Delaunay
