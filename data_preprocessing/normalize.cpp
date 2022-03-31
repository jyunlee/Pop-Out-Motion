#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <cmath>

struct Mesh
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
} surface;


int main(int argc, char *argv[])
{

    if (!igl::readOBJ(argv[1], surface.V, surface.F))
    {
        std::cout << "failed to load mesh" << std::endl;
    }
    int n = surface.V.rows();

    double min_x = 1e9, min_y = 1e9, min_z = 1e9;
    double max_x = -1e9, max_y = -1e9, max_z = -1e9;

    for (int i = 0; i < n; i++) {
        min_x = std::min(min_x, surface.V(i, 0));
        min_y = std::min(min_y, surface.V(i, 1));
        min_z = std::min(min_z, surface.V(i, 2));
        max_x = std::max(max_x, surface.V(i, 0));
        max_y = std::max(max_y, surface.V(i, 1));
        max_z = std::max(max_z, surface.V(i, 2));
    }
    for (int i = 0; i < n; i++) {
        surface.V(i, 0) -= (max_x + min_x) / 2;
        surface.V(i, 1) -= (max_y + min_y) / 2;
        surface.V(i, 2) -= (max_z + min_z) / 2;
    }
    double radius = 0;
    for (int i = 0; i < n; i++) {
        double x = surface.V(i, 0), y = surface.V(i, 1), z = surface.V(i, 2);
        radius = std::max(radius, sqrt(x * x + y * y + z * z));
    }

    for (int i = 0; i < n; i++) {
        surface.V(i, 0) /= radius;
        surface.V(i, 1) /= radius;
        surface.V(i, 2) /= radius;
    }
    igl::writeOBJ(argv[2], surface.V, surface.F);
    return 0;
}
