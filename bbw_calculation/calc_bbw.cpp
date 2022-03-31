#include <igl/boundary_conditions.h>
#include <igl/matrix_to_list.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/readOBJ.h>
#include <igl/slice.h>
#include <iostream>
#include <fstream>
#include <string>

#include "bbw.h"
#include "mosek.h"

struct Mesh
{
    Eigen::MatrixXd V, U;
    Eigen::MatrixXi T, F;
} low, high, surface;

Eigen::MatrixXd W;

int main(int argc, char *argv[])
{
    using namespace Eigen;
    using namespace std;
    using namespace igl;

    if (!readOBJ(argv[1], high.V, high.F)) // mesh
    {
        cout << "failed to load mesh" << endl;
    }

    if (!readOBJ(argv[2], low.V, low.F)) // point points
    {
        cout << "failed to load mesh" << endl;
    }

    {
        Eigen::VectorXi b;
        {
            Eigen::VectorXi J = Eigen::VectorXi::LinSpaced(high.V.rows(), 0, high.V.rows() - 1);
            Eigen::VectorXd sqrD;
            Eigen::MatrixXd _2;
	    cout << high.V.rows() << " " << high.V.cols() << endl;
	    cout << low.V.rows() << " " << low.V.cols() << endl;
            igl::point_mesh_squared_distance(low.V, high.V, J, sqrD, b, _2);
        }

        igl::slice(high.V, b, 1, low.V);

        std::vector<std::vector<int>> S;
        igl::matrix_to_list(b, S);

        cout << "Computing weights for " << b.size() << " handles at " << high.V.rows() << " vertices..." << endl;
        const int k = 2;
        
	Eigen::VectorXi bbb;
	Eigen::MatrixXd bc;

	Eigen::VectorXi P(low.V.rows());

	for (int i = 0; i < low.V.rows(); i ++) {
	    P[i] = i;
	}

	igl::boundary_conditions(high.V, high.F, low.V, P, Eigen::MatrixXi(), Eigen::MatrixXi(), bbb, bc);
	
	Eigen::MatrixXd J_(S.size(), S.size());
	J_.setIdentity();
         
	igl::BBWData bbw_data;
	igl::mosek::MosekData mosek_params;
	bbw_data.active_set_params.max_iter = 100;
	bbw_data.verbosity = 1;

        igl::mosek::bbw(high.V, high.F, bbb, bc, bbw_data, mosek_params, W, std::string(argv[3]));
	
        igl::normalize_row_sums(W, W);

    }
    
    freopen(argv[4], "w", stdout); // weights_mesh
    printf("%d %d\n", W.rows(), W.cols());
    for (int i = 0; i < W.rows(); i++)
    {
        for (int j = 0; j < W.cols(); j++)
            printf("%f ", W(i, j));
        printf("\n");
    }
}
