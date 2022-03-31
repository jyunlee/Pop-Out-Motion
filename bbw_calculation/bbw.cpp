// This file is part of libigl, a simple c++ geometry processing
// library.
//
// Copyright (C) 2016 Alec Jacobson <alecjacobson@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include "bbw.h"
#include <igl/mosek/mosek_quadprog.h>
#include <igl/harmonic.h>
#include <igl/slice_into.h>
#include <Eigen/Sparse>
#include <iostream>
#include <cstdio>
#include <string>

void read_matrix(std::string fileName, Eigen::MatrixXd &outputMat) {
  using namespace std;

  fstream cin; 

  cin.open(fileName.c_str()); 

  if (cin.fail()) { 
    std::cout << "Failed to open file: " << fileName << std::endl; 
    std::cin.get(); 
  }

  string s;
  vector <vector <long double> > matrix;

  while (getline(cin, s)) { 

    stringstream input(s); 
    long double temp; 
    vector <long double> currentLine; 

    while (input >> temp) 
      currentLine.push_back((long double)temp);     

    matrix.push_back(currentLine);
  }

  igl::list_to_matrix(matrix, outputMat);
}

template <
  typename DerivedV,
  typename DerivedEle,
  typename Derivedb,
  typename Derivedbc,
  typename DerivedW>
IGL_INLINE bool igl::mosek::bbw(
  const Eigen::PlainObjectBase<DerivedV> & V,
  const Eigen::PlainObjectBase<DerivedEle> & Ele,
  const Eigen::PlainObjectBase<Derivedb> & b,
  const Eigen::PlainObjectBase<Derivedbc> & bc,
  igl::BBWData & data,
  igl::mosek::MosekData & mosek_data,
  Eigen::PlainObjectBase<DerivedW> & W,
  std::string fname
  )
{
  using namespace std;
  using namespace Eigen;
  assert(!data.partition_unity && "partition_unity not implemented yet");
  // number of domain vertices
  int n = V.rows();
  // number of handles
  int m = bc.cols();
  // Build biharmonic operator
  //Eigen::SparseMatrix<typename DerivedV::Scalar> Q;
  //harmonic(V,Ele,2,Q);
  W.derived().resize(n,m);
  // No linear terms
  VectorXd c = VectorXd::Zero(n);
  // No linear constraints
  SparseMatrix<typename DerivedW::Scalar> A(0,n);
  VectorXd uc(0,1),lc(0,1);
  // Upper and lower box constraints (Constant bounds)
  cout << n << endl;
  VectorXd ux = VectorXd::Ones(n);
  VectorXd lx = VectorXd::Zero(n);

  MatrixXd pred_A;
  Eigen::SparseMatrix<typename DerivedV::Scalar> sparse_pred_A;
  read_matrix(fname.c_str(), pred_A);
  sparse_pred_A = pred_A.sparseView();

  // Loop over handles
  for(int i = 0;i<m;i++)
  {
    if(data.verbosity >= 1)
    {
      cout<<"BBW: Computing weight for handle "<<i+1<<" out of "<<m<<
        "."<<endl;
    }
    VectorXd bci = bc.col(i);
    VectorXd Wi;
    // impose boundary conditions via bounds
    slice_into(bci,b,ux);
    slice_into(bci,b,lx);

    //bool r = mosek_quadprog(Q,c,0,A,lc,uc,lx,ux,mosek_data,Wi);
    bool r = mosek_quadprog(sparse_pred_A,c,0,A,lc,uc,lx,ux,mosek_data,Wi);
    if(!r)
    {
      cout << "error occured" << endl; 
      FILE * error_log;
      error_log = fopen("./error_log.txt", "a+");
      fprintf(error_log, "error\n");
      fclose(error_log);
      return false;
    }
    W.col(i) = Wi;
  }
#ifndef NDEBUG
    const double min_rowsum = W.rowwise().sum().array().abs().minCoeff();
    if(min_rowsum < 0.1)
    {
      cerr<<"bbw.cpp: Warning, minimum row sum is very low. Consider more "
        "active set iterations or enforcing partition of unity."<<endl;
    }
#endif

  return true;
}

#ifdef IGL_STATIC_LIBRARY
// Explicit template instantiation
template bool igl::mosek::bbw<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, igl::BBWData&, igl::mosek::MosekData&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
#endif

