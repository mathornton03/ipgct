#include <stdio.h> 
#include <string.h> 
#include <iostream> 
#include <vector>
#include <fstream>
#include <cassert>
#include <random>
#include <Eigen/Dense>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <cmath>

using namespace std;

vector<float> readTS(string tsfn);
vector<vector<float>> splitTS(vector<float> TS, int sssize);
void displayHelp(int argc, char ** argv);
void genRandTSFile(string ofilename, int seriesize);
vector<vector<float>> getLagsMatrix(vector<float> values, int lags);
struct ols_results OLS(vector<vector<float>> lm, bool estint);
struct irls_results IRLS_logit(vector<vector<float>> lm, bool estint);
void displayHead();
void printTTestResultsWNames(struct ols_results osr, vector<string> effectnames);
float printGrangerCausalConclusion(struct ols_results solo, struct ols_results crossed);
Eigen::VectorXd WLS_iterable(Eigen::MatrixXd X, Eigen::VectorXd Y, Eigen::VectorXd W);
Eigen::VectorXd calculateLogisticLink(Eigen::MatrixXd X, Eigen::VectorXd beta);

struct gct_settings{
    string infilename;
    string outfilename; 
    int subseqsize;
    bool genrand = false;
    bool binTS = false;
    string routfilename; 
    int routseqsize;
    int maxlags;
    float tolerance = 0.0001; 
    int maxiter = 1000;
};

struct ols_results{
    Eigen::VectorXd Y;
    Eigen::MatrixXd X; 
    Eigen::VectorXd beta; 
    Eigen::VectorXd var_beta;
    Eigen::VectorXd t_stat;
    Eigen::VectorXd p_values;

    float mse;
    float msr;
    float F_numdf;
    float F_denomdf;
    float Fstat;
    float p_value;
};

struct wls_results : ols_results {
    Eigen::VectorXd W;
};

struct irls_results : wls_results{
    float numiter;
};

void displayHead(){
    cout << "       Internal Pairwise Granger Causality Testing (IPGCT)   " << endl;
    cout << "               For Time Series of Different Kinds            " << endl;
    cout << "          A Test for Internal Serial Correlation in TS       " << endl;
    cout << "                       ( Thornton 2022 )                     " << endl;
    cout << "                          Version 1.1                        " << endl;
    cout << "                                                             " << endl;
}

struct gct_settings parseargs(int argc, char ** argv){
    /*if (argc != 5 && argc != 8){
        cerr << "Attempted Usage: " << endl; 
        cerr << argv << endl;
        cerr << "Error: incorrect usage." << endl << endl;
        displayHelp(argc, argv);
    }*/
    struct gct_settings gctset;
    for (int i = 1; i < argc; i++){
        if (i == 1){
            gctset.infilename = argv[i];
            continue;
        } else if (i == 2){
            gctset.outfilename = argv[i];
        } else if (i == 3){
            gctset.subseqsize = stoi(argv[i]);
        } else if (i == 4){
            gctset.maxlags = stoi(argv[i]);
        }else if (string(argv[i]) == "-G" | string(argv[i]) == "-g" | string(argv[i]) == "--GEN") {
            gctset.genrand = true;
            gctset.routfilename = argv[i+1];
            gctset.routseqsize = stoi(argv[i+2]);
        } else if (string(argv[i]) == "-B" | string(argv[i]) == "-b" | string(argv[i]) == "--BERN"){
            gctset.binTS = true;
        } else if (string(argv[i]) == "-T" | string(argv[i]) == "-t" | string(argv[i]) == "--TOL"){
            gctset.tolerance = stof(argv[i+1]);
        } else if (string(argv[i]) == "-H" | string(argv[i]) == "-h" | string(argv[i]) == "--HELP"){
            displayHelp(argc, argv);
            exit(1);
        }
    }
    if (gctset.genrand){
        genRandTSFile(gctset.routfilename, gctset.routseqsize);
        cerr << "Generating Random TS of size " << gctset.routseqsize << 
          " on file: " << gctset.routfilename << endl;
        exit(1);
    }
    return(gctset);
}

void displayHelp(int argc, char ** argv){
    cout << "                                                             " << endl;
    cout << " usage: " << argv[0] << " time_series.txt outputfilename.txt " << endl;
    cout << "                          SSSIZE MXLAGS [-b/-B/-BERN]        " << endl; 
    cout << "                          [-g/-G/--GEN] <RFILEN> <RS>        " << endl;
    cout << "                          [-t/-T/--TOL] <RTOLE = 0.0001>     " << endl;
    cout << "                                                             " << endl;
    cout << " time_series.txt:  the location of a time series file, which " << endl;
    cout << "   contains the subset of values that will be used in the    " << endl; 
    cout << "   subsequent Internal Pairwise Granger Causality Testing    " << endl; 
    cout << "   procedure, a series of floating point values separated by " << endl; 
    cout << "   any amount of whitespace desired by the user: ex - 12 34.2" << endl; 
    cout << "                                                             " << endl;
    cout << " outputfilename.txt:  The location to write the results of   " << endl; 
    cout << "   the test to.                                              " << endl;
    cout << "                                                             " << endl; 
    cout << " SSSIZE: The size of the substrings (an integer size)        " << endl;
    cout << "                                                             " << endl; 
    cout << " MXLAGS: The number of AR lags to try in each substring fit  " << endl;
    cout << "                                                             " << endl; 
    cout << " Optional Arguments:                                         " << endl; 
    cout << "                                                             " << endl; 
    cout << "   [-B/-b/--BERN]: Assume the input Time Series to be on a   " << endl;
    cout << "                  Binary support, and use the standard       " << endl;
    cout << "                  Bernoulli Trials form of the General Linear" << endl;
    cout << "                  Model (Logistic Regression) instead of the " << endl;
    cout << "                  Standard Gaussian One.                     " << endl;
    cout << "                                                             " << endl; 
    cout << "   [-T/-t/--TOL] <RTOLE>: relative tolerance, stop iterating " << endl; 
    cout << "                  under during the IRLS procedure of the test" << endl;
    cout << "                  in the case of binary data (logistic       " << endl;
    cout << "                  regression)                                " << endl;
    cout << "                                                             " << endl; 
    cout << "   [-G/-g/--GEN] <RFILEN>: generate a random TS file for test" << endl; 
    cout << "        RFILEN - File name for random output file.           " << endl; 
    cout << "        RS - Size of Random TS to be generated               " << endl;
    cout << "                                                             " << endl;
    cout << "                                                             " << endl;
    exit(1);
}

struct wls_results WLS(vector<vector<float>> lm, vector<float> w, bool estint){
    struct wls_results retres; 
    Eigen::VectorXd Y(lm[0].size());
    for (int i = 0; i < lm[0].size(); i++){
        Y(i) = lm[0][i];
    }
    Eigen::MatrixXd X(lm[0].size(),lm.size()-1);

    if (estint){
        X = Eigen::MatrixXd(lm[0].size(), lm.size());
    }

    for (int i = 1; i <= (lm[i].size()-1); i++ ){
        for (int j = 0; j < lm[i].size(); j++){
            X(j,i) = lm[i][j];
        }
    }
    if (estint){
        for (int j = 0; j < lm[0].size(); j ++){
            X(j,0) = 1;
        }
    }

    Eigen::MatrixXd W(w.size(),w.size());
    for (int i = 0; i < w.size(); i++){
        W(i,i) = w[i];
    }

    Eigen::VectorXd beta = WLS_iterable(X,Y,W);

    retres.W = W;
    retres.beta = beta;
    return(retres);
}

Eigen::VectorXd WLS_iterable(Eigen::MatrixXd X, Eigen::VectorXd Y, Eigen::VectorXd W){
    Eigen::MatrixXd tX = X.transpose();
    Eigen::MatrixXd WX = W * X; 
    Eigen::MatrixXd tXWX =  tX * WX; 
    Eigen::MatrixXd tXWY = tX * W * Y;
    Eigen::MatrixXd tXWXi = tXWX.inverse();
    Eigen::VectorXd betahat = tXWXi * tXWY;
    return(betahat);
}
Eigen::VectorXd calculateLogisticLink(Eigen::MatrixXd X, Eigen::VectorXd beta){
    Eigen::VectorXd yhat(X.rows());
    Eigen::VectorXd xcur(X.cols());
    Eigen::MatrixXd linkcur;
    float link;
    for (int i = 0; i < X.rows(); i++){
        xcur = X(i,Eigen::all);
        linkcur = xcur * beta;
        linkcur *= -1; 
        link = linkcur.value();
        yhat(i) = 1 / (1 + exp(link));
    }
    return(yhat);
}
struct irls_results IRLS_logit(vector<vector<float>> lm, gct_settings gcs, 
                                bool estint){
    struct irls_results retres;
    Eigen::VectorXd Y(lm[0].size());
    for (int i = 0; i < lm[0].size(); i++){
        Y(i) = lm[0][i];
    }
    Eigen::MatrixXd X(lm[0].size(), lm.size()-1);
    if (estint) {
        X = Eigen::MatrixXd(lm[0].size(), lm.size());
    }
    for (int i = 1; i <= (lm.size()-1); i++){
        for (int j = 0; j <= (lm.size()-1); j++){
            X(j,i) = lm[i][j];
        }
    }
    if (estint){
        for(int i = 0; i < lm[0].size(); i++){
            X(i,0) = 1;
        }
    }
    struct ols_results ifit = initfit(lm, estint);
    float relerr = 1;
    int iter = 1;
    Eigen::VectorXd betcur = ifit.beta;
    Eigen::VectorXd betprev = ifit.beta;
    Eigen::VectorXd yhatcur = calculateLogisticLink(X,betcur);
    Eigen::VectorXd curresid = (Y.array() - yhatcur.array());
    Eigen::VectorXd hprieta = yhatcur.array() * (1-yhatcur.array());
    Eigen::VectorXd Zcur = (Y.array()-yhatcur.array())/(hprieta.array());
    curresid = curresid.array().square();
    struct wls_results curwlsres;
    while(relerr >= gcs.tolerance && iter <= gcs.maxiter){
        betprev = betcur;
        betcur = betcur.array()+WLS_iterable(X, Zcur, hprieta).array();
        relerr = ((betcur.array()-betprev.array()).square().sum());
        yhatcur = calculateLogisticLink(X,betcur);
        curresid = (Y.array() - yhatcur.array());
        hprieta = yhatcur.array() * (1-yhatcur.array());
        Zcur = (Y.array()-yhatcur.array())/(hprieta.array());
        iter++;
    }
    retres.beta = betcur;
    return(retres);
}



struct ols_results OLS(vector<vector<float>> lm, bool estint){
    struct ols_results retres;
    Eigen::VectorXd Y(lm[0].size());
    for (int i = 0; i < lm[0].size(); i++){
        Y(i) = lm[0][i];
    }
    // cout << "X of size " << lm1[0].size() << " by " << (lm1.size()-1) << endl;
    Eigen::MatrixXd X(lm[0].size(),lm.size()-1);
    
    if (estint){
        X = Eigen::MatrixXd(lm[0].size(),lm.size());
    }


    for (int i = 1; i <= (lm.size()-1); i++){
    //    cout << "i: " << i << " lm[i].size(): " << lm[i].size() << endl;
        for (int j = 0; j < lm[i].size(); j ++){
            X(j,i) = lm[i][j];
        }
    }
    //    cout << "blah 3" << endl;
    if (estint){
        for (int j = 0; j < lm[0].size(); j ++){
            X(j,0) = 1;
        }
    }

    //cout << "Y:" << endl << Y << endl;
    //cout << "X:" << endl << X << endl;
    Eigen::MatrixXd tX = X.transpose();

    //cout << "t(X): " << endl << tX << endl;

    //cout << "t(X) * X: " << endl << tX * X << endl;

    Eigen::MatrixXd tXX = tX * X;

    Eigen::MatrixXd tXXi = tXX.inverse();

    //cout << "(t(X) * X)^(-1): " << endl << tXXi << endl; 
    
    Eigen::MatrixXd tXY = tX * Y; 

    Eigen::VectorXd beta = tXXi * tXY;

    //cout << "beta: " << endl << beta << endl;

    Eigen::VectorXd yhat = X * beta;

    //cout << "yhat: " << endl << yhat << endl;

    Eigen::VectorXd yres = (Y-yhat);

    float ymean = Y.mean();



    Eigen::VectorXd ytes = (Y.array()-ymean);


    float ytessq = ytes.transpose()*ytes;

    float yressq = yres.transpose()*yres;

    Eigen::VectorXd Xmeans = X.colwise().mean();

    //cout << "Xmeans: " << endl << Xmeans << endl;

    //float xerrsum;
    //vector<float> xerrs;
    //for (int i = 1; i < Xmeans.size(); i++){
    //    xerrsum = 0;
    //    for (int j = 0; j < lm[0].size(); j ++){
    //        xerrsum +=(X(j,i)-Xmeans[i])*(X(j,i)-Xmeans[i]);
    //    }
    //    xerrs.push_back(xerrsum);
    //}

    float s = yressq / (lm[0].size() - lm.size());
    //float sosqn = s/sqrt(lm[0].size());
   // cout << "ssq" <<  s << endl;

    Eigen::MatrixXd sebeta = s*tXXi;
    

   // cout << "sebeta: " << endl << sebeta << endl;

    Eigen::VectorXd tstats = beta.array() / sebeta.diagonal().array().sqrt();

   // cout << "t-stats: " << endl << tstats << endl;

    boost::math::students_t dist(lm[0].size()-lm.size());

    double P_x_gt_t;
    Eigen::VectorXd pvals(tstats.size());
    for (int i = 0; i < tstats.size(); i++){
        P_x_gt_t = 2*(1.0 - boost::math::cdf(dist,abs(tstats(i))));
        pvals(i) = P_x_gt_t;
    }

   // cout << "p-values: " << endl << pvals << endl;
   retres.Y = Y;
   retres.X = X;
   retres.beta = beta; 
   retres.var_beta = sebeta.diagonal().array().sqrt();
   retres.t_stat = tstats;
   retres.p_values = pvals;

   retres.mse = yressq;
   retres.msr = ytessq;
   float denomdf = (lm[0].size()- (lm.size()) );
   retres.F_denomdf = denomdf;

   float numdf = (lm[0].size()) - denomdf;
   retres.F_numdf = numdf;
   retres.Fstat = ((retres.msr - retres.mse)/(numdf)) / (retres.mse/(denomdf));

   boost::math::fisher_f distf(numdf,denomdf);

   retres.p_value = 1-boost::math::cdf(distf,retres.Fstat);
   return(retres);

}

void printFTestResults(struct ols_results osr){
    cout << " Internal F-Test Result:                     " << endl;
    cout << "  Mean Squared Error (Prediction): " << osr.mse << endl; 
    cout << "  Mean Squared Error (Regression): " << osr.msr << endl; 
    cout << "  F-Statistic: " << osr.Fstat << endl;
    cout << "  P-Value: " << osr.p_value << endl;
}

void printTTestResults(struct ols_results osr){
    cout << " Internal T-Test Results:                    " << endl; 
    cout << "  Intercept: " << osr.t_stat[0] << " ("  << osr.p_values[0] << ") " << endl;
    for (int i = 1; i < osr.t_stat.size(); i++){
        cout << "  Lag " << i << ": " << osr.beta[i] << "  -  " << osr.t_stat[i] << " (" << osr.p_values[i] << ") " << endl;
    }
}

void printTTestResultsWNames(struct ols_results osr, vector<string> effectnames){
    cout << "     External T-Test Results:                    " << endl; 
    cout << "      Intercept: " << osr.t_stat[0] << " ("  << osr.p_values[0] << ") " << endl;
    for (int i = 1; i < osr.t_stat.size(); i++){
        cout << "        " <<effectnames[i-1] << ": " << osr.beta[i] << "  -  " << osr.t_stat[i] << " (" << osr.p_values[i] << ") " << endl;
    }
}

float printGrangerCausalConclusion(struct ols_results solo, struct ols_results crossed){
   // cout << "          Granger Causal Comparison:     " << endl; 
  //  cout << "            Self AR MSE: " << solo.mse << endl; 
    int selfdf = solo.X.rows() - solo.beta.size();
   // cout << "            Self AR DF: " << selfdf << endl; 
   // cout << "            Cross AR MSE: " << crossed.mse << endl; 
    int crossdf = crossed.X.rows() - crossed.beta.size();

   // cout << "            Cross AR DF: " << crossdf << endl;
    float fstat =  ((solo.mse-crossed.mse)/(selfdf-crossdf))/(crossed.mse/crossdf);
    boost::math::fisher_f dist(selfdf-crossdf, crossdf);
    //cout << "            F-Stat (Ratio - Cross/Self): " << fstat << endl;
    float p_value = (1 - boost::math::cdf(dist,fstat));
    //cout << "            P-Value (P(F >= f, d1 = " << selfdf-crossdf << ", d2 = " << selfdf << ")): " << p_value;
   // cout << "                                            " << endl; 
    return(p_value);
}

Eigen::MatrixXd ipgct_logistic(vector<vector<float>> spts, struct gct_settings gcs){
    Eigen::MatrixXd granger_F_pvals(spts.size(),spts.size());
    float tol = gcs.tolerance;
    float err = 1;
    int obsperseq = gcs.subseqsize - (gcs.maxlags-1);
    Eigen::VectorXd weightinitial(obsperseq);
    struct ols_results = OLS()
    for (int i = 0; i < obsperseq; i++){
        weightinitial(i) = 1;
    }
    while(err > tol){

    }
    return(granger_F_pvals);
}

Eigen::MatrixXd ipgct(vector<vector<float>> spts, struct gct_settings gcs){
    
    Eigen::MatrixXd granger_F_pvals(spts.size(),spts.size());

    struct ols_results ors; 
    struct ols_results gcsres;
    struct ols_results fcres;
    struct ols_results selfsigres;
    vector<vector<float>> lm;
    vector<vector<float>> om;
    vector<vector<float>> cm;
    vector<vector<float>> fm; // final model
    vector<string> finaleffectnamevec;
    vector<string> effectnamevec; 
    float curpval;

    for (int i = 0; i < spts.size(); i ++){
        lm = getLagsMatrix(spts[i],gcs.maxlags);
        ors = OLS(lm, true);
       // cout << "Subsequence " << i << " Results: " << endl;
      //  cout << "lm size: " << lm.size() << endl;
      //  cout << "lm[0] size: " << lm[0].size() << endl;
       // printFTestResults(ors);
       // printTTestResults(ors);
        for (int j = 0; j < spts.size(); j++){
            if (i == j){
                granger_F_pvals(i,j) = 1;
                continue;
            }
            om = getLagsMatrix(spts[j],gcs.maxlags);
            if (lm[0].size() != om[0].size()){
                continue;
            }
            cm.clear();
            effectnamevec.clear();
            for (int k = 0; k < lm.size(); k++){
                //cout << "k: " << k << endl;
                //cout << "lm[k][0]: " << lm[k][0] << endl;
                //cout << "p_values(k): " << ors.p_values[k] << endl;
                if (k >= 1){
                    if (ors.p_values(k) >= 0.00005){
                        continue;
                    }
                    effectnamevec.push_back("Seq " + to_string(i) + " Lag " + to_string(k));
                }
                cm.push_back(lm[k]);
            }
            selfsigres = OLS(cm,true);
            //printTTestResultsWNames(selfsigres,effectnamevec);
            //cout << "Blah " << endl;
            for (int k = 1; k < om.size(); k++){
                cm.push_back(om[k]);
                effectnamevec.push_back("Seq " + to_string(j) + " Lag " + to_string(k));
            }
            //cout << "Blah 2" << endl;
            gcsres = OLS(cm,true);
            
            //printTTestResultsWNames(gcsres, effectnamevec);
            fm.clear();
            finaleffectnamevec.clear();
            fm.push_back(cm[0]);
            for (int k = 1; k < gcsres.p_values.size(); k++){
                if (gcsres.p_values[k] <= 0.05){
                    fm.push_back(cm[k]);
                    finaleffectnamevec.push_back(effectnamevec[k-1]);
                }
            }
            fcres = OLS(fm,true);
           // printTTestResultsWNames(fcres,finaleffectnamevec);
            //printFTestResults(fcres);
            if (selfsigres.F_numdf < fcres.F_numdf){
              //  cout << "   Subsequence " << to_string(j) << " granger-causes subsequence " << to_string(i)  << ": " << endl; 
                curpval = printGrangerCausalConclusion(selfsigres, fcres);
                granger_F_pvals(i,j) = curpval;
            } else {
                granger_F_pvals(i,j) = 1;
            }
    }
    }
    //cout << "granger_F_pvals: " << endl; 
    //cout << granger_F_pvals << endl;
    return(granger_F_pvals);
}
int main(int argc, char ** argv){
    srand(0XFEEDBEEFDEADBEEFL%0XFEDL);
    displayHead();

    bool estintercept = true;

    struct gct_settings gcs = parseargs(argc,argv);

    vector<float> TS = readTS(gcs.infilename);

    vector<vector<float>> spts = splitTS(TS, stoi(argv[3]));

    Eigen::MatrixXd granger_F_pvals(spts.size(),spts.size());

    if (!gcs.binTS){
        granger_F_pvals = ipgct(spts,gcs);
    } else {
        cout << "Binary input data, using Logistic Regression (IRLS)" << endl;

       // granger_F_pvals = ipgct_logistic(spts,gcs);
       return(0);
    }
    cout << "granger_F_pvals: " << endl; 
    cout << granger_F_pvals << endl;
    float dist_from_no_sig = 1- ((spts.size()*spts.size() - granger_F_pvals.sum())/(spts.size()*spts.size()));
    cout << "Granger Clustering Ratio: (1 indicates complete granger causal-relation) - " << dist_from_no_sig << endl;
}

vector<vector<float>> getLagsMatrix(vector<float> values, int lags){
    vector<vector<float>> outLags;
    vector<float> curlag; 
    if (lags < 0){
        cerr << "Error: Lags value in getLagsMatrix must be positive. " << endl;
        exit(1);
    }
    else if (lags == 0){
        outLags.push_back(values);
        return(outLags);
    }
    else {
        for (int i = 0; i <= lags; i++){
            curlag.clear();
            for (int j = i; j < (i+ values.size() - lags); j++){
                curlag.push_back(values[j]);
            }
            outLags.push_back(curlag);
        }
    }
    return(outLags);
}

vector<float> readTS(string tsfn){
    ifstream tsf;
    tsf.open(tsfn);
    vector<float> TS; 
    string curval; 
    if (tsf.is_open()) {
        while (!tsf.eof()){
            tsf >> curval; 
            if (tsf.eof()){
                break;
            }
            TS.push_back(stof(curval));
        }
    }
    tsf.close();
    return(TS);
}

void genRandTSFile(string ofilename, int seriesize){
    ofstream ofile; 
    ofile.open(ofilename);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(1, 10);
    for (int i = 1; i <= seriesize; i++){
        ofile << dist(gen) << " ";
        if (i%10 == 0){
            ofile << endl;
        }
    }
    ofile.close();
}

vector<vector<float>> splitTS(vector<float> TS, int sssize){
    assert(sssize <= TS.size());
    vector<vector<float>> TS_splits;
    vector<float> TS_cursplit;
    cout << "Splitting String of size " << TS.size() << " into ones of " <<
        sssize << endl;
    int i = 0;
    while (i < TS.size()){
        if (i%sssize == 0 && i !=0){
            TS_splits.push_back(TS_cursplit);
            TS_cursplit.clear();
        }
        TS_cursplit.push_back(TS[i-1]);
        i += 1;
    }
    return(TS_splits);
}