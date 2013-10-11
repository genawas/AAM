#include "AAM_Train2D.h"
#include "AAM_Utilities.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "dirent.h"
#include "delaunay2d.h"
#include <levmar.h>
#include <lmmin.h>

double gaussrand()
{
	static double V1, V2, S;
	static int phase = 0;
	double X;

	if(phase == 0) {
		do {
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;

	return X;
}

double noise (double mean, double var) /* f u n c ti o n to g e n e r a t e n oi s e */
{
    return mean + sqrt(var) * gaussrand();
}

void expfunc(double *p, double *x, int m, int n, void *data)
{
	for (int i=0; i<n; ++i){
		x[i]=p[0]*exp(-p[1]*i)+p[2];
	}
}

void test_levmar()
{
	const int n=100, m=3; // 40 measurements , 3 parameters
	double p[m] ,x[n] ,opts[LM_OPTS_SZ] , info[LM_INFO_SZ] ;
	int  i; int ret;
	for (i=0; i<n ;++i){
		/* generate measurement data */
		x[i]= (5.0*exp(-0.1*i) + 1.0) + noise(0.0, 0.1) ;
	}
	/* init i a l pa rame te r s e s tim a t e : ( 1 . 0 , 0 . 0 , 0 . 0 ) */
	p [0] = 1.0 ; p[1] = 0.0 ; p [2] = 0.0;
	/* o p timi z a ti o n c o n t r o l pa rame te r s */
	opts[0]=LM_INIT_MU; opts [1]=1E-15;
	opts[2]=1E-15; opts [3]=1E-20;
	opts[4]=LM_DIFF_DELTA; // f o r f i n i t e d i f f e r e n c e Jacobian
	/* in vo ke the o p timi z a ti o n f u n c ti o n */
	ret=dlevmar_dif(expfunc, p ,x , m, n ,1000 , NULL, info ,NULL, NULL, NULL) ; // without Jacobian
	/* r e t=dlevmar_der ( expfunc , ja c e x pf u n c , p , x , m, n , 1000 ,
		opts , i nf o , NULL, NULL, NULL) ; with Jacobian */

	printf("Levenberg-Marquardt returned %d in %g iter, reason %g\nSolution: ", ret, info[5], info[6]);
	for(int ii=0; ii<m; ++ii)
		printf("%.7g ", p[ii]);
	printf("\n\nMinimization info:\n");
	for(int ii=0; ii<LM_INFO_SZ; ++ii)
		printf("%g ", info[ii]);
	printf("\n");
}

void evaluate_nonlin1(
    const double *p, int n, const void *data, double *f, int *info )
{
    f[0] = p[0]*p[0] + p[1]*p[1] - 1; /* unit circle    x^2+y^2=1 */
    f[1] = p[1] - p[0]*p[0];          /* standard parabola  y=x^2 */
}

/*
int test_lmfit()
{
    int n = 2;   // dimension of the problem 
    double p[2]; // parameter vector p=(x,y) 

    // auxiliary parameters 
    lm_control_struct control = lm_control_double;
    lm_status_struct  status;
    control.verbosity  = 31;

    // get start values from command line
	p[0] = 1;
    p[1] = 1;
    // the minimization 
    printf( "Minimization:\n" );
    lmmin( n, p, n, NULL, evaluate_nonlin1, &control, &status );

    // print results
    printf( "\n" );
    printf( "lmmin status after %d function evaluations:\n  %s\n",
            status.nfev, lm_infmsg[status.outcome] );

    printf( "\n" );
    printf("Solution:\n");
    printf("  x = %19.11f\n", p[0]);
    printf("  y = %19.11f\n", p[1]);
    printf("  d = %19.11f => ", status.fnorm);

    // convergence of lmfit is not enough to ensure validity of the solution
    if( status.fnorm >= control.ftol )
        printf( "not a valid solution, try other starting values\n" );
    else
        printf( "valid, though not the only solution: "
                "try other starting values\n" );

    return 0;
}
*/

void main(void)
{
	//test_levmar();
	//test_lmfit();

	std::string dir = "C:/Users/genawas/Dropbox/Ims4test";

	AMM_Model2D_Options options;
	options.set_default();
	options.nscales = 4;

	if(false){

		//cv::Mat A = (cv::Mat_ <double> (5, 1) << 1, 2, 3, 4, 5);
		//cv::Mat B = (cv::Mat_ <double> (5, 1) << 5, 4, 3, 2, 1);
		//cv::Mat C;
		//concat_mat(A, B, C, 1);
		//std::cout << C << std::endl;

		//cv::Mat Z = (cv::Mat_ <double> (5, 5) <<  17, 24, 1, 8, 15,
		//	23, 5, 7, 14, 16, 4, 6, 13, 20, 22, 10, 12, 19, 21, 3, 11, 18, 25, 2, 9);
		//AAM_NormalizeAppearance2D(Z);
		//std::cout << Z << std::endl;
		std::string dir_shape = "C:/Users/genawas/Dropbox/Ims4test/Shapesc";
		std::string dir_ims = "C:/Users/genawas/Dropbox/Ims4test/Imsc";
		
		std::vector<AAM_ALL_DATA> Data;
		cv::Mat F;
		load_triangulation(dir + "/faces.yml", F);
		AAMtrainAllScales(dir_shape, dir_ims, Data, F, options);
		AAMsaveAllData(dir, Data);

		/*
		AAMTrainingData TrainingData;
		load_Data(dir_shape, dir_ims, TrainingData);

		//cv::Scalar delaunay_color(255, 0, 0);
		//draw_subdiv(TrainingData.Texture[1], tempC, F, delaunay_color);

		AAMTexture Text;
		AAMShape ShapeData;
		AAM_MakeShapeModel2D_tire(TrainingData, ShapeData, Text, options);
		//load_triangulation(dir + "/faces.yml", ShapeModel.F);

		load_triangulation(dir + "/faces.yml", Text.F);
		AAMAppearance AppearanceData;
		//AppearanceData.ObjectPixels = cv::imread(dir + "/mask.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		cv::FileStorage f;
		f.open(dir + "/mask.yml", cv::FileStorage::READ);
		assert(f.isOpened());
		f["mask"] >> AppearanceData.ObjectPixels;
		AAM_MakeAppearanceModel2D(TrainingData, Text, ShapeData, AppearanceData, options);
		//writeMat(cv::Mat(AppearanceData.Evectors), "s.mat", "s");

		AAMShapeAppearance ShapeAppearanceData;
		AAM_ALL_DATA Data;
		Data.A = AppearanceData;
		Data.S = ShapeData;
		Data.SA = ShapeAppearanceData;
		Data.N = TrainingData.N;
		Data.n = TrainingData.n;

		AAM_CombineShapeAppearance2D_tire(TrainingData, Data, Text, options);
		//writeMat(cv::Mat(ShapeAppearanceData.Evectors), "s.mat", "s");
		cv::Mat R;
		AAM_MakeSearchModel2D_tire(TrainingData, Data, Text, options);
		saveAAMData(Data, dir + "/data.yml");
		saveAAMTextue(Text, dir + "/text.yml");
		*/
	}

	else{
		
		std::vector<AAM_ALL_DATA> Data(options.nscales); 

		cv::Mat F;
		load_triangulation(dir + "/faces.yml", F);
		AAMloadAllData(dir, Data);
		
		cv::Mat im=cv::imread("C:/Users/genawas/Dropbox/ASM_data/tire_test1.tif", CV_LOAD_IMAGE_GRAYSCALE);
		AAMTform tformLarge;
		tformLarge.shift=(cv::Mat_ <double> (1,2) << -439.0046, -476.5241);
		tformLarge.scale=0.7803;
		cv::Mat pos;
		ApplyModel2D(Data, F, im, tformLarge, pos, options);
		writeMat(pos, "p.mat", "p", false);
	}
	//cv::Mat Z = (cv::Mat_ <double> (5, 5) <<  17, 24, 1, 8, 15,
	//	23, 5, 7, 14, 16, 4, 6, 13, 20, 22, 10, 12, 19, 21, 3, 11, 18, 25, 2, 9);
	//cv::Mat eigval, eigvec, psi;
	//pca_eigenvectors(Z, eigval, eigvec, psi);

	//std::cout << eigvec << std::endl;
}