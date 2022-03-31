#include <igl/readMESH.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <iostream>
#include <fstream>
#include <string>


struct Mesh
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi T, F;
} mesh;

Eigen::SparseMatrix<double> L;
Eigen::SparseMatrix<double> M, Minv;


int main(int argc, char *argv[])
{
    using namespace Eigen;
    using namespace std;
    using namespace igl;

    if (!readMESH(argv[1], mesh.V, mesh.T, mesh.F)) //tet mesh
    {
        cout << "failed to load mesh" << endl;
    }

    cotmatrix(mesh.V, mesh.T, L);

    massmatrix(mesh.V, mesh.T, MASSMATRIX_TYPE_DEFAULT, M);

    // normalize
    M /= ((Matrix<double, Dynamic, 1>)M.diagonal()).array().abs().maxCoeff();
    Minv =
        ((Matrix<double, Dynamic, 1>)M.diagonal().array().inverse()).asDiagonal();

    // save V
    FILE * v_file;
    v_file = fopen((string(argv[2]) + "/V/" + string(argv[3]) + ".txt").c_str(), "w");

    for (int i = 0; i < mesh.V.rows(); i ++) {
        for (int j = 0; j < mesh.V.cols(); j ++)
            fprintf(v_file, "%f ", mesh.V.coeff(i, j));
        fprintf(v_file, "\n");  
    }
    fclose(v_file);
       
    // save L
    FILE * l_file;
    l_file = fopen((string(argv[2]) + "/L/" + string(argv[3]) + ".txt").c_str(), "w");  

    for (int i = 0; i < L.rows(); i ++) {
        for (int j = 0; j < L.cols(); j ++) {
	    if (std::abs (L.coeff(i, j)) > 1E-8)
		fprintf(l_file, "%d %d %.12lf\n", i, j, L.coeff(i, j));
	}
    }
    fclose(l_file);

    // save Minv
    FILE * minv_file;
    minv_file = fopen((string(argv[2]) + "/Minv/" + string(argv[3]) + ".txt").c_str(), "w");

    for (int i = 0; i < Minv.rows(); i ++) 
        fprintf(minv_file, "%.12lf ", Minv.coeff(i, i));

    fclose(minv_file);
    
}
