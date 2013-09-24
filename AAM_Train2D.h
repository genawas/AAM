#ifndef AAM_TRAIN_2D_
#define AAM_TRAIN_2D_

#include <opencv2/core/core.hpp>
//#include <opencv2/core/eigen.hpp>

struct AAMTform {
	cv::Point2f shift;
	double scale;
};

struct AAMShape{

	// Eigen Vectors and Eigen Values
cv::Mat Evectors;
cv::Mat Evalues;

cv::Mat data_mean;
cv::Mat data;
cv::Mat varSc;

std::vector<cv::Vec3i> F; //delaunay
//cv::Mat F;

std::vector<cv::Point2f> MeanVertices;
std::vector<cv::Point> Lines;
cv::Size textureSize;

//ShapeData.Tri= delaunay(x_mean(1:end/2),x_mean(end/2+1:end));

};

struct AMM_Model2D_Options{
	// Set options
		// Number of contour points interpolated between the major landmarks.
		int ni;
	// Set normal appearance/contour, limit to +- m*sqrt( eigenvalue )
		int m;
	// Size of appearance texture as amount of orignal image
		int texturesize;
	// Number of image scales
		int nscales;
	// Number of search itterations
		int nsearch;

		void set_default()
		{
			ni=20;
			m=3;
			texturesize=1;
			nscales=4;
			nsearch=15;
		}
};

struct AAMTrainingData{
	int N; //number of shapes
	int n; //number of shape points
	std::vector<cv::Mat> Texture;
	std::vector<std::vector<cv::Point2f> > Shape;
	std::vector<std::vector<cv::Point2f> > ShapeC;
	std::vector<cv::Point2f> Shift;
	std::vector<double> Scale;

	void initialize(int S){
		N=S;
		Texture=std::vector<cv::Mat>(N);
		Shape=std::vector<std::vector<cv::Point2f> > (N);
		ShapeC=std::vector<std::vector<cv::Point2f> > (N);
		Shift=std::vector<cv::Point2f> (N);
		Scale=std::vector<double> (N);
	}
};


struct AAMAppearance{
	int k; //appearance vector size
	cv::Mat Evectors;
	cv::Mat Evalues;
	cv::Mat g_mean;
	cv::Mat  g;
	cv::Mat ObjectPixels;
	std::vector<cv::Point2f> base_points;

};

class AAM_Train2D{

public:

private:
};

void load_triangulation(std::string filename, cv::Mat &F);

void load_triangulation(std::string filename, std::vector<cv::Vec3i> &F);

int pca_eigenvectors(cv::Mat &A, cv::Mat &eigenvalues, cv::Mat &eigenvectors, cv::Mat &psi);

void load_contour_txt(const std::string filename, std::vector<cv::Point2f> &contour);

int load_contour_yml(const std::string filename, std::vector<cv::Point2f> &contour);

void load_Data(std::string dir_shape, std::string dir_ims, AAMTrainingData &TrainingData);

void AAM_MakeShapeModel2D_tire(AAMTrainingData &TrainingData, AAMShape &ShapeModel, AMM_Model2D_Options &options);

void AAM_MakeAppearanceModel2D(AAMTrainingData &TrainingData, AAMShape &ShapeModel, AAMAppearance &AppearanceData, AMM_Model2D_Options &options);

void AAM_Weights2D_tire(AAMTrainingData &TrainingData, AAMShape &ShapeData, AAMAppearance &AppearanceData, AMM_Model2D_Options &options);

#endif
