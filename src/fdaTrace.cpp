#include <RcppArmadillo.h>

// Arguments:
// left_limit_    : left  limit of sampling interval
// right_limit_   : right limit of sampling interval
// tau            : leading coefficient of L_lambda operator
// eta_minus_real : matrix of real      part of left most eigenvalues
// eta_minus_imag : matrix of imaginary part of left most eigenvalues 
// eta_plus_real  : matrix of real      part of right most eigenvalues
// eta_plus_imag  : matrix of imaginary part of right most eigenvalues
// M_minus_real   : matrix of real      part of left most eigenvector top
// M_minus_imag   : matrix of imaginary part of left most eigenvector top 
// M_plus_real    : matrix of real      part of right most eigenvector top
// M_plus_imag    : matrix of imaginary part of right most eigenvector top
// FleftMat       : matrix of left  boundary conditions
// FrightMat      : matrix of right boundary conditions
// N_             : sample length (number of rows in Y-matrix)
//
// Argument types, from which the dimensions (kk,order,NN,MM) are extracted:
// left_limit_    : double
// right_limit_   : double
// tau            : double
// eta_minus_real : matrix of dimension (J,k)
// eta_minus_imag : matrix of dimension (J,k)
// eta_plus_real  : matrix of dimension (J,k)
// eta_plus_imag  : matrix of dimension (J,k)
// M_minus_real   : matrix of dimension (J,k)
// M_minus_imag   : matrix of dimension (J,k)
// M_plus_real    : matrix of dimension (J,k)
// M_plus_imag    : matrix of dimension (J,k)
// FleftMat       : matrix of dimension (k,2*k)
// FrightMat      : matrix of dimension (k,2*k)
// N_             : double
//
// Value:
// vector (length J) of diagonal integrals of the Greens functions 
//
// Implicit assumptions:
// 1) left_limit_ < right_limit_
// 2) Sampling is equidistant as described in Markussen (2013)
//
// Remarks:
// 1) If eta_minus_real <=0 and eta_plus_real >= 0, then the algorithm is
//    numerically stable.


extern "C" SEXP fdaTrace(SEXP left_limit_, SEXP right_limit_, 
                         SEXP tau, 
                         SEXP eta_minus_real, SEXP eta_minus_imag,
                         SEXP eta_plus_real,  SEXP eta_plus_imag,
                         SEXP M_minus_real, SEXP M_minus_imag,
                         SEXP M_plus_real,  SEXP M_plus_imag,
                         SEXP FleftMat, SEXP FrightMat,
                         SEXP N_
                         ) {
  try{
    // copy data to armadillo structures
    double  left_limit = Rcpp::as<double>( left_limit_);
    double right_limit = Rcpp::as<double>(right_limit_);
    arma::cx_mat eta_minus = 
      arma::cx_mat(Rcpp::as<arma::mat>(eta_minus_real),
                   Rcpp::as<arma::mat>(eta_minus_imag));
    arma::cx_mat eta_plus = 
      arma::cx_mat(Rcpp::as<arma::mat>(eta_plus_real),
                   Rcpp::as<arma::mat>(eta_plus_imag));
    arma::cx_mat M_minus = 
      arma::cx_mat(Rcpp::as<arma::mat>(M_minus_real),
                   Rcpp::as<arma::mat>(M_minus_imag));
    arma::cx_mat M_plus = 
      arma::cx_mat(Rcpp::as<arma::mat>(M_plus_real),
                   Rcpp::as<arma::mat>(M_plus_imag));
    // remark: eta_left and eta_right defined as row vectors
    arma::mat Fleft    = Rcpp::as<arma::mat>(FleftMat);
    arma::mat Fright   = Rcpp::as<arma::mat>(FrightMat);
    int kk       = eta_minus.n_cols;
    int JJ       = eta_minus.n_rows;
    int NN       = Rcpp::as<int>(N_);
    double Delta = (right_limit-left_limit)/NN;

    // initialize vector to contain result
    arma::vec traceGreen = arma::zeros<arma::vec>(JJ);

    // initialize variables
    arma::cx_mat    Wminus       = arma::ones<arma::cx_mat>(2*kk,kk);
    arma::cx_mat    Wplus        = arma::ones<arma::cx_mat>(2*kk,kk);
    arma::cx_mat    W            = arma::zeros<arma::cx_mat>(2*kk,2*kk);
    arma::cx_mat    invW         = arma::zeros<arma::cx_mat>(2*kk,2*kk);
    arma::cx_vec    v2minus      = arma::zeros<arma::cx_vec>(kk);
    arma::cx_vec    v2plus       = arma::zeros<arma::cx_vec>(kk);
    arma::cx_rowvec exp_left     = arma::zeros<arma::cx_rowvec>(kk);
    arma::cx_rowvec exp_right    = arma::zeros<arma::cx_rowvec>(kk);
    arma::cx_mat    Aminus_minus = arma::zeros<arma::cx_mat>(kk,kk);
    arma::cx_mat    Aplus_plus   = arma::zeros<arma::cx_mat>(kk,kk);
    arma::cx_mat    Aminus_plus  = arma::zeros<arma::cx_mat>(kk,kk);
    arma::cx_mat    Aplus_minus  = arma::zeros<arma::cx_mat>(kk,kk);
    arma::cx_mat    Bmat         = arma::zeros<arma::cx_mat>(kk,kk);
    arma::cx_mat    Fleft_W      = arma::zeros<arma::cx_mat>(kk,kk);
    arma::cx_mat    Fright_W     = arma::zeros<arma::cx_mat>(kk,kk);

    // loop over sets of eigenvalues
    for (int jj=0; jj<JJ; jj++) {
      // define variables from Markussen (2013), Theorem 2
      Wminus.row(0) = M_minus.row(jj);
      Wplus.row(0)  = M_plus.row(jj);
      for (int ii=1; ii<2*kk; ii++) {
        Wminus.row(ii) = Wminus.row(ii-1) % eta_minus.row(jj);   
        Wplus.row(ii)  = Wplus.row(ii-1)  % eta_plus.row(jj);   
      }
      W       = arma::join_rows(Wminus,Wplus);
      invW    = arma::pinv(W);
      v2minus = invW.submat(0, 2*kk-1,  kk-1,2*kk-1);
      v2plus  = invW.submat(kk,2*kk-1,2*kk-1,2*kk-1);

      // make some pre-computations
      exp_left  = exp((right_limit-left_limit)*eta_minus.row(jj));
      exp_right = exp((left_limit-right_limit)*eta_plus.row(jj));
      Fleft_W   = arma::pinv(Fleft *Wminus)*Fleft *Wplus;
      Fright_W  = arma::pinv(Fright*Wplus) *Fright*Wminus;

      // compute Aminus_minus
      Aminus_minus = (
        arma::repmat(exp_left.st(),1,kk) - 
        arma::repmat(exp_left,kk,1)
        ) / (Delta*(
            arma::repmat(eta_minus.row(jj).st(),1,kk)-
            arma::repmat(eta_minus.row(jj),kk,1)
        ));
      Aminus_minus.diag() = NN*exp_left;

      // compute Aplus_plus
      Aplus_plus = (
        arma::repmat(exp_right.st(),1,kk) - 
        arma::repmat(exp_right,kk,1)
      ) / (Delta*(
          arma::repmat(eta_plus.row(jj),kk,1) -
          arma::repmat(eta_plus.row(jj).st(),1,kk)
      ));
      Aplus_plus.diag() = NN*exp_right;

      // compute Aminus_plus
      // remark: Bmat is temporarily used before it is computed below
      Bmat = arma::repmat(eta_minus.row(jj).st(),1,kk) - 
        arma::repmat(eta_plus.row(jj),kk,1);
      Aminus_plus = (exp((right_limit-left_limit)*Bmat) - 
        arma::ones<arma::cx_mat>(kk,kk)) / (Delta*Bmat);

      // compute Aplus_minus
      Aplus_minus = Aminus_plus.st();

      // compute Bmat
      Bmat = Fleft_W*arma::diagmat(exp_right)*Fright_W*
             arma::pinv(arma::eye<arma::cx_mat>(kk,kk)-
	                arma::diagmat(exp_left)*Fleft_W*arma::diagmat(exp_right)*Fright_W);

      // compute trace
      traceGreen(jj)  = arma::as_scalar(arma::real(NN*M_minus.row(jj)*v2minus));
      traceGreen(jj) += arma::as_scalar(arma::real(M_minus.row(jj)*(Fleft_W %Aminus_plus)*v2plus));
      traceGreen(jj) -= arma::as_scalar(arma::real(M_plus.row(jj)*(Fright_W%Aplus_minus)*v2minus));
      traceGreen(jj) -= arma::as_scalar(arma::real(M_plus.row(jj)*((Fright_W*arma::diagmat(exp_left)*
                                                        Fleft_W)%Aplus_plus)*v2plus));
      traceGreen(jj) += arma::as_scalar(arma::real(M_minus.row(jj)*(Bmat%Aminus_minus)*v2minus));
      traceGreen(jj) += arma::as_scalar(arma::real(M_minus.row(jj)*((Bmat*arma::diagmat(exp_left)*Fleft_W)%
                                                       Aminus_plus)*v2plus));
      traceGreen(jj) -= arma::as_scalar(arma::real(M_plus.row(jj)*((Fright_W*arma::diagmat(exp_left)*Bmat)%
                                                       Aplus_minus)*v2minus));
      traceGreen(jj) -= arma::as_scalar(arma::real(M_plus.row(jj)*((Fright_W*arma::diagmat(exp_left)*Bmat*
                                                       arma::diagmat(exp_left)*Fleft_W)%Aplus_plus)*v2plus));
    }

    // Rescale trace
    traceGreen = traceGreen/Rcpp::as<double>(tau);

    // return it to R
    return Rcpp::wrap(traceGreen);

    // Below possible exceptions are taken care off
  } catch( std::exception &ex ) {
      forward_exception_to_r( ex );
  } catch(...) { 
      ::Rf_error( "c++ exception (unknown reason)"); 
  }

  // return to R
  return R_NilValue;
}
