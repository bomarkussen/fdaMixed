#include <RcppArmadillo.h>

// Arguments:
//   left_limit_    : left  limit of sampling interval (including border)
//   right_limit_   : right limit of sampling interval (including border)
//   alpha_2k_      : leading coefficient of L-operator
//   eta_minus_real : vector of real      part of left most eigenvalues
//   eta_minus_imag : vector of imaginary part of left most eigenvalues 
//   eta_plus_real  : vector of real      part of right most eigenvalues
//   eta_plus_imag  : vector of imaginary part of right most eigenvalues
//   M_minus_real   : vector of real      part of left most eigenvector top
//   M_minus_imag   : vector of imaginary part of left most eigenvector top 
//   M_plus_real    : vector of real      part of right most eigenvector top
//   M_plus_imag    : vector of imaginary part of right most eigenvector top
//   FleftMat       : matrix of left  boundary conditions
//   FrightMat      : matrix of right boundary conditions
//   Gmat           : covariance matrix for random effect
//   Ymat           : matrix of observations
//   projMat        : matrix to contain projection of columns of Ymat
//   betaHat_       : vector to contain estimate of fixed effect
//   uBLUP_         : vector to contain prediction of random effect
//   xBLUP_         : matrix to contain prediction of serial effect
//   condRes_       : vector to contain conditional residuals
//   Cbeta_         : matrix to contain variance of betaHat-beta
//   Cu_            : matrix to contain variance of uBLUP-u
//   logLikVec      : vector to contain terms appearing in log likelihood
//
// Argument types, from which the dimensions (kk,order,pp,qq, NN,MM) are extracted:
//   left_limit_    : double
//   right_limit_   : double
//   alpha_2k_      : double
//   eta_minus_real : vector of length kk
//   eta_minus_imag : vector of length kk
//   eta_plus_real  : vector of length kk
//   eta_plus_imag  : vector of length kk
//   M_minus_real   : vector of length kk
//   M_minus_imag   : vector of length kk
//   M_plus_real    : vector of length kk
//   M_plus_imag    : vector of length kk
//   FleftMat       : matrix of dimension (kk,2*kk)
//   FrightMat      : matrix of dimension (kk,2*kk)
//   Gmat           : matrix of dimension (qq,qq)
//   Ymat           : matrix of dimension (NN,Ycols), Ycols=   MM*(1+pp+qq)
//   projMat        : matrix of dimension (NN,cols),   cols=Ycols*(order+1)
//   betaHat_       : vector of length pp
//   uBLUP_         : vector of length qq
//   xBLUP_         : matrix of dimension (NN*MM,1+order)
//   condRes_       : vector of length NN*MM
//   Cbeta_         : matrix of dimension (pp,pp)
//   Cu_            : matrix of dimension (qq,qq)
//   logLikVec      : vector of length 1+1+1+1+1+order = 5+order
//
// Value:
//   1) projections returned in variable projMat, which may 
//      be interpreted/reshaped as cube of dimensions (NN,cols,order+1)
//   2) estimates of fixed effect returned in variable betaHat
//   3) prediction of random effect returned in variable uBLUP
//   4) prediction of serial effect returned in variable xBLUP, which may 
//      be interpreted/reshaped as cube of dimensions (NN,MM,1+order)
//
// Implicit assumptions:
//   1) left_limit_ < right_limit_
//   2) Sampling is equidistant as described in Markussen (2013)
//   3) Dimensions are consistent (this is not checked)
//   4) Ymat has at least 3 rows, ie. NN >= 3 (this is not checked)
//
// Remarks:
//   1) If eta_minus_real <=0 and eta_plus_real >= 0, then the algorithm 
//      is numerically stable.
//   2) order larger than 2*k-1 gives linear dependent results
//
// Optimization ideas:
//   1) Introduce 2 matrices of dimension NN times kk to contain the needed 
//      exponentials. These matrices may be computed similar to 
//      prod_discounter_left.
//   2) If exp((right_limit-left_limit)*eta_left)==0 with double precision,
//      then the formulae simplifies considerably.
//      Similar when exp((left_limit-right_limit)*eta_right)==0. 
//   3) If eta_left = -eta_right, then simplifications also exists.

extern "C" SEXP fdaEngine(SEXP left_limit_, SEXP right_limit_, 
                          SEXP alpha_2k_, 
                          SEXP eta_minus_real, SEXP eta_minus_imag,
                          SEXP eta_plus_real,  SEXP eta_plus_imag,
                          SEXP M_minus_real, SEXP M_minus_imag,
                          SEXP M_plus_real,  SEXP M_plus_imag,
                          SEXP FleftMat, SEXP FrightMat,
                          SEXP Gmat,
                          SEXP Ymat,
                          SEXP projMat,
                          SEXP betaHat_,
                          SEXP uBLUP_,
                          SEXP xBLUP_,
                          SEXP condRes_,
                          SEXP Cbeta_,
                          SEXP Cu_,
                          SEXP logLikVec
                          ) {
  try{
    // copy data to armadillo structures
    double left_limit  = Rcpp::as<double>(left_limit_);
    double right_limit = Rcpp::as<double>(right_limit_);
    double alpha_2k    = Rcpp::as<double>(alpha_2k_);
    arma::cx_rowvec eta_minus = 
      arma::cx_rowvec(Rcpp::as<arma::rowvec>(eta_minus_real),
                      Rcpp::as<arma::rowvec>(eta_minus_imag));
    arma::cx_rowvec eta_plus = 
      arma::cx_rowvec(Rcpp::as<arma::rowvec>(eta_plus_real),
                      Rcpp::as<arma::rowvec>(eta_plus_imag));
    arma::cx_rowvec M_minus = 
      arma::cx_rowvec(Rcpp::as<arma::rowvec>(M_minus_real),
                      Rcpp::as<arma::rowvec>(M_minus_imag));
    arma::cx_rowvec M_plus = 
      arma::cx_rowvec(Rcpp::as<arma::rowvec>(M_plus_real),
                      Rcpp::as<arma::rowvec>(M_plus_imag));
    arma::mat Fleft  = Rcpp::as<arma::mat>(FleftMat);
    arma::mat Fright = Rcpp::as<arma::mat>(FrightMat);
    arma::mat G      = Rcpp::as<arma::mat>(Gmat);

    // creates Rcpp objects from SEXP's
    Rcpp::NumericMatrix Yr(Ymat);
    Rcpp::NumericMatrix projR(projMat);
    Rcpp::NumericVector betaR(betaHat_);
    Rcpp::NumericVector uR(uBLUP_);
    Rcpp::NumericMatrix xR(xBLUP_);
    Rcpp::NumericVector condResR(condRes_);
    Rcpp::NumericMatrix CbetaR(Cbeta_);
    Rcpp::NumericMatrix CuR(Cu_);
    Rcpp::NumericVector logLikR(logLikVec);

    // extract dimensions from defined objects
    int kk    = Fleft.n_rows;
    int NN    = Yr.nrow();
    int Ycols = Yr.ncol();
    int cols  = projR.ncol();
    int pp    = betaR.size();
    int qq    = uR.size();
    int MM    = Ycols/(1+pp+qq);
    int order = xR.ncol()-1;

    // introduce iterators to handle situation with pp=0 and/or qq=0
    arma::vec dummy_vec = arma::zeros<arma::vec>(1);
    arma::mat dummy_mat = arma::zeros<arma::mat>(1,1);
    arma::vec::iterator betaP  = dummy_vec.begin();
    arma::vec::iterator uP     = dummy_vec.begin();
    arma::mat::iterator CbetaP = dummy_mat.begin();
    arma::mat::iterator CuP    = dummy_mat.begin();
    if (pp > 0) {
      betaP  = betaR.begin();
      CbetaP = CbetaR.begin();
    }
    if (qq > 0) {
      uP  = uR.begin();
      CuP = CuR.begin();
    }
    
    // reuses memory and avoids extra copies
    arma::mat Y(Yr.begin(),NN,Ycols,false);
    arma::mat proj(projR.begin(),NN,cols,false);
    arma::vec betaHat(betaP,pp,false);
    arma::vec uBLUP(uP,qq,false);
    arma::mat xBLUP(xR.begin(),NN*MM,1+order,false);
    arma::vec condRes(condResR.begin(),NN*MM,false);
    arma::mat Cbeta(CbetaP,pp,pp,false);
    arma::mat Cu(CuP,qq,qq,false);
    arma::vec logLik(logLikR.begin(),5+order,false);
    
    // define variables from Markussen (2013), Proposition 2
    double Delta = (right_limit-left_limit)/NN;
    arma::cx_mat Wminus = arma::zeros<arma::cx_mat>(2*kk,kk);
    Wminus.row(0) = M_minus;
    for (int ii=1; ii<2*kk; ii++) {
      Wminus.row(ii) = Wminus.row(ii-1) % eta_minus;   
    }
    arma::cx_mat Wplus = arma::zeros<arma::cx_mat>(2*kk,kk);
    Wplus.row(0) = M_plus;
    for (int ii=1; ii<2*kk; ii++) {
      Wplus.row(ii) = Wplus.row(ii-1) % eta_plus;
    }
    arma::cx_mat W         = arma::join_rows(Wminus,Wplus);
    arma::cx_mat invW      = arma::pinv(W);
    arma::cx_vec v2minus   = invW.submat(0, 2*kk-1,  kk-1,2*kk-1);
    arma::cx_vec v2plus    = invW.submat(kk,2*kk-1,2*kk-1,2*kk-1);
    arma::cx_vec xi_minus  = arma::strans((exp(Delta*eta_minus/2)-1)/eta_minus);
    arma::cx_vec xi0_minus = arma::strans((1-(1-Delta*eta_minus)%
      exp(Delta*eta_minus))/(Delta*eta_minus%eta_minus));
    arma::cx_vec xi1_minus = arma::strans((exp(Delta*eta_minus)-1-Delta*eta_minus)/
      (Delta*eta_minus%eta_minus));
    arma::cx_vec xi_plus   = arma::strans((1-exp(-Delta*eta_plus/2))/eta_plus);
    arma::cx_vec xi0_plus  = arma::strans((exp(-Delta*eta_plus)-1+Delta*eta_plus)/
      (Delta*eta_plus%eta_plus));
    arma::cx_vec xi1_plus  = arma::strans((1-(1+Delta*eta_plus)%
      exp(-Delta*eta_plus))/(Delta*eta_plus%eta_plus));
    
    // make some pre-computations
    arma::cx_mat Fleft_W  = arma::pinv(Fleft *Wminus)*Fleft *Wplus;
    arma::cx_mat Fright_W = arma::pinv(Fright*Wplus) *Fright*Wminus;

    // compute discounters
    arma::cx_vec discounter_minus      = arma::strans(exp(     Delta*eta_minus));
    arma::cx_vec half_discounter_minus = arma::strans(exp( 0.5*Delta*eta_minus));
    arma::cx_vec discounter_plus       = arma::strans(exp(    -Delta*eta_plus));
    arma::cx_vec half_discounter_plus  = arma::strans(exp(-0.5*Delta*eta_plus));

    // initialize forward loop sums
    arma::cx_mat sum1 = arma::zeros<arma::cx_mat>(kk,Ycols);
    arma::cx_mat sum2 = (v2minus%xi_minus)*Y.row(0);
    arma::cx_mat sum3 = arma::zeros<arma::cx_mat>(kk,Ycols);
    arma::cx_mat sum4 = (v2plus%xi_plus)*Y.row(0);
    // initialize forward loop discounters
    arma::cx_vec prod_discounter_left  = half_discounter_minus;
    arma::cx_vec prod_discounter_right = half_discounter_plus;
    // initialize forward loop phi matrix
    arma::cx_mat exp_FW_exp = arma::diagmat(exp((0.5-NN)*Delta*eta_plus))*Fright_W*
      arma::diagmat(exp((NN-0.5)*Delta*eta_minus));
    arma::cx_mat exp_FW     = arma::diagmat(prod_discounter_left)*Fleft_W;
    arma::cx_mat phi_fac    = arma::pinv(arma::eye<arma::cx_mat>(kk,kk)-
      exp_FW*arma::diagmat(prod_discounter_right)*exp_FW_exp);
    arma::cx_mat phi        = arma::zeros<arma::cx_mat>(order+1,kk);
    for (int ii=0; ii<=order; ii++) {
      phi.row(ii) = (Wminus.row(ii)-Wplus.row(ii)*exp_FW_exp)*phi_fac;
    }
    // update conditional mean
    proj.row(0) = arma::reshape(arma::trans(arma::real(phi*(sum2+exp_FW*sum4))),1,cols);
    // forward loop
    for (int nn=1; nn<NN; nn++) {
      // update sums
      sum1 = arma::diagmat(discounter_minus)*sum1 + (v2minus%xi0_minus)*Y.row(nn-1);
      sum2 = arma::diagmat(discounter_minus)*sum2 + (v2minus%xi1_minus)*Y.row(nn);
      sum3 += (prod_discounter_right%v2plus%xi0_plus)*Y.row(nn-1);
      sum4 += (prod_discounter_right%v2plus%xi1_plus)*Y.row(nn);
      // update discounters
      prod_discounter_left  = prod_discounter_left  % discounter_minus;
      prod_discounter_right = prod_discounter_right % discounter_plus;
      // compute phi
      exp_FW_exp = arma::diagmat(exp((0.5+nn-NN)*Delta*eta_plus))*Fright_W*
                  arma::diagmat(exp((NN-nn-0.5)*Delta*eta_minus));
      exp_FW     = arma::diagmat(prod_discounter_left)*Fleft_W;
      phi_fac    = arma::pinv(arma::eye<arma::cx_mat>(kk,kk)-
        exp_FW*arma::diagmat(prod_discounter_right)*exp_FW_exp);
      for (int ii=0; ii<=order; ii++) {
        phi.row(ii) = (Wminus.row(ii)-Wplus.row(ii)*exp_FW_exp)*phi_fac;
      }
      // update conditional mean
      proj.row(nn) = arma::reshape(arma::trans(arma::real(phi*(sum1+sum2+exp_FW*(sum3+sum4)))),1,cols);
    }

    // initialize backward loop sums
    sum1 = (v2plus%xi_plus)*Y.row(NN-1);
    sum2 = arma::zeros<arma::cx_mat>(kk,Ycols);
    sum3 = (v2minus%xi_minus)*Y.row(NN-1);
    sum4 = arma::zeros<arma::cx_mat>(kk,Ycols);
    // initialize backward loop discounters
    prod_discounter_left  = half_discounter_minus;
    prod_discounter_right = half_discounter_plus;
    // initialize backward loop psi matrix, use variable phi
    exp_FW_exp = arma::diagmat(exp((NN-0.5)*Delta*eta_minus)) *Fleft_W*
      arma::diagmat(exp((0.5-NN)*Delta*eta_plus));
    exp_FW     = arma::diagmat(prod_discounter_right)*Fright_W;
    phi_fac    = arma::pinv(arma::eye<arma::cx_mat>(kk,kk)-
      exp_FW*arma::diagmat(prod_discounter_left)*exp_FW_exp);
    for (int ii=0; ii<=order; ii++) {
      phi.row(ii) = (Wplus.row(ii)-Wminus.row(ii)*exp_FW_exp)*phi_fac;
    }
    // update conditional mean
    proj.row(NN-1) -= arma::reshape(arma::trans(arma::real(phi*(sum1+exp_FW*sum3))),1,cols);
    // backward loop
    for (int nn=NN-1; nn>0; nn--) {
      // update sums
      sum1 = arma::diagmat(discounter_plus)*sum1 + (v2plus%xi0_plus)*Y.row(nn-1);
      sum2 = arma::diagmat(discounter_plus)*sum2 + (v2plus%xi1_plus)*Y.row(nn);
      sum3 += (prod_discounter_left%v2minus%xi0_minus)*Y.row(nn-1);
      sum4 += (prod_discounter_left%v2minus%xi1_minus)*Y.row(nn);
      // update discounters
      prod_discounter_left  = prod_discounter_left  % discounter_minus;
      prod_discounter_right = prod_discounter_right % discounter_plus;
      // compute phi
      exp_FW_exp = arma::diagmat(exp((nn-0.5   )*Delta*eta_minus))*Fleft_W*
        arma::diagmat(exp((0.5-nn   )*Delta*eta_plus));
      exp_FW     = arma::diagmat(prod_discounter_right)*Fright_W;
      phi_fac    = arma::pinv(arma::eye<arma::cx_mat>(kk,kk)-
        exp_FW*arma::diagmat(prod_discounter_left)*exp_FW_exp);
      for (int ii=0; ii<=order; ii++) {
        phi.row(ii) = (Wplus.row(ii)-Wminus.row(ii)*exp_FW_exp)*phi_fac;
      }
      // update conditional mean
      proj.row(nn-1) -= arma::reshape(arma::trans(arma::real(phi*(sum1+sum2+exp_FW*(sum3+sum4)))),1,cols);
    }

    // Normalize projection
    proj = proj/alpha_2k;

    // introduce iterator to handle situation with pp=0 and/or qq=0
    arma::mat::col_iterator matCol = Y.begin_col(0);

    // Initialize variables used to estimate fixed, random and serial effects
    arma::vec Yobs(Y.begin(),NN*MM,false);
    arma::vec Yproj(proj.begin(),NN*MM,false);

    if (pp > 0) matCol = Y.begin_col(MM);          else matCol = Y.begin_col(0);
    arma::mat gamma(matCol,NN*MM,pp,false);

    if (pp > 0) matCol = proj.begin_col(MM);       else matCol = proj.begin_col(0);
    arma::mat gammaProj(matCol,NN*MM,pp,false);

    if (qq > 0) matCol = Y.begin_col(MM+MM*pp);    else matCol = Y.begin_col(0);
    arma::mat Z(matCol,NN*MM,qq,false);

    if (qq > 0) matCol = proj.begin_col(MM+MM*pp); else matCol = proj.begin_col(0);
    arma::mat Zproj(matCol,NN*MM,qq,false);

    // Compute variance matrices
    // REMARK: Possible speed optimization by avoiding computation of Cbeta
    // REMARK: Cbeta may also inserted in logLik(1) below
    if (qq > 0) Cu = arma::inv(arma::inv(G)+arma::trans(Z)*(Z-Zproj));
    if (pp > 0) {
      if (qq > 0) {
	Cbeta = arma::inv(arma::trans(gamma)*(gamma-gammaProj)-
 	                  arma::trans(gamma)*(Z-Zproj)*Cu*arma::trans(Z)*(gamma-gammaProj));
      } else {
	Cbeta = arma::inv(arma::trans(gamma)*(gamma-gammaProj));
      }
    }

    // Estimate fixed effect
    if (pp > 0) {
      if (qq > 0) {
	betaHat = Cbeta*(arma::trans(gamma)*(Yobs-Yproj)-
			 arma::trans(gamma)*(Z-Zproj)*Cu*arma::trans(Z)*(Yobs-Yproj));
      } else {
	betaHat = Cbeta*arma::trans(gamma)*(Yobs-Yproj);
      }
    }

    // Predict random effect
    if (qq > 0) {
      if (pp > 0) {
	uBLUP = Cu*arma::trans(Z)*(Yobs-Yproj-gamma*betaHat+gammaProj*betaHat);
      } else {
	uBLUP = Cu*arma::trans(Z)*(Yobs-Yproj);
      }
    }

    // Predict serial correlated effect
    xBLUP.col(0) = Yproj;
    if (pp > 0) xBLUP.col(0) -= gammaProj*betaHat;
    if (qq > 0) xBLUP.col(0) -= Zproj*uBLUP;
    if (order > 0) {
      for (int ii=1; ii<=order; ii++) {
        xBLUP.col(ii) = arma::reshape(proj.cols(ii*MM*(1+pp+qq),ii*MM*(1+pp+qq)+MM-1),NN*MM,1);
        if (pp > 0) xBLUP.col(ii) -= arma::reshape(proj.cols(ii*MM*(1+pp+qq)+MM,ii*MM*(1+pp+qq)+MM*(1+pp)-1),NN*MM,pp)*betaHat;
        if (qq > 0) xBLUP.col(ii) -= arma::reshape(proj.cols(ii*MM*(1+pp+qq)+MM*(1+pp),(ii+1)*MM*(1+pp+qq)-1),NN*MM,qq)*uBLUP;
      }
    }

    // Compute conditional residuals
    condRes = Yobs-xBLUP.col(0);
    if (pp > 0) condRes -= gamma*betaHat;
    if (qq > 0) condRes -= Z*uBLUP;

    // Compute terms appearing in the log likelihood
    double val;
    double sign;
    // Term 1: conditional log determinant of random effect
    if (qq > 0) {
      arma::log_det(val,sign,arma::eye<arma::mat>(qq,qq)-arma::trans(Z)*(Z-Zproj)*G);
      logLik(0)=val;
    } else {logLik(0)=0;}
    // Term 2: REML correction
    if (pp > 0) {
      if (qq > 0) {
	arma::log_det(val,sign,arma::trans(gamma)*(gamma-gammaProj)-
 	              arma::trans(gamma)*(Z-Zproj)*Cu*
		      arma::trans(Z)*(gamma-gammaProj));
	logLik(1)=val;
      } else {
	arma::log_det(val,sign,arma::trans(gamma)*(gamma-gammaProj));
	logLik(1)=val;
      }
    }
    // Term 3: Squared length of conditional residual
    logLik(2)=arma::dot(condRes,condRes);
    // Term 4: Squared length of predicted random effects measured by G
    if (qq > 0) {
      logLik(3)=arma::as_scalar(arma::trans(uBLUP)*solve(G,uBLUP));
    } else {logLik(3)=0;}
    // Term 5 to 5+order: Inner products of predicted serial effect and its derivatives
    for (int ii=0; ii<=order; ii++) {
      logLik(4+ii)=Delta*arma::dot(xBLUP.col(0),xBLUP.col(ii));
    }

    // Result is saved and returned in the external variables:
    //   proj, betaHat, uBLUP, xBLUP, Cbeta, Cu, condRes, logLik

    // Below possible exceptions are taken care off
  } catch( std::exception &ex ) {
      forward_exception_to_r( ex );
  } catch(...) { 
      ::Rf_error( "c++ exception (unknown reason)"); 
  }

  // return to R
  return R_NilValue;
}
