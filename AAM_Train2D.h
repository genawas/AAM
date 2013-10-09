#ifndef AAM_TRAIN_2D_
#define AAM_TRAIN_2D_

#include <opencv2/core/core.hpp>
//#include <opencv2/core/eigen.hpp>

struct AAMTform {
	//cv::Point2f shift;
	cv::Mat shift;
	double scale;
};

struct AAMShape{

	// Eigen Vectors and Eigen Values
	cv::Mat Evectors;
	cv::Mat Evalues;

	cv::Mat data_mean;
	cv::Mat data;
	cv::Mat varSc;
	//std::vector<cv::Point2f> MeanVertices;
	cv::Mat MeanVertices;

};

struct AAMTexture{
	//std::vector<cv::Vec3i> F; //delaunay
	cv::Mat F;
	//std::vector<cv::Point> Lines;
	cv::Size textureSize;
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
	//std::vector<std::vector<cv::Point2f> > Shape, ShapeC;
	std::vector<cv::Mat> Shape, ShapeC;
	std::vector<cv::Mat> Shift;
	//std::vector<cv::Point2f> Shift;
	std::vector<double> Scale;

	void initialize(int S){
		N=S;
		Texture=std::vector<cv::Mat>(N);
		//Shape=std::vector<std::vector<cv::Point2f> > (N);
		//ShapeC=std::vector<std::vector<cv::Point2f> > (N);
		Shape=std::vector<cv::Mat>(N);
		ShapeC=std::vector<cv::Mat>(N);
		//Shift=std::vector<cv::Point2f> (N);
		Shift=std::vector<cv::Mat>(N);
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
	//std::vector<cv::Point2f> base_points;
	cv::Mat base_points;
};

class AAM_Train2D{

public:

private:
};

struct AAMShapeAppearance{
	cv::Mat Evectors;
	cv::Mat Evalues;
	cv::Mat b_mean;
	cv::Mat b;
	cv::Mat Ws;
};

struct AAM_ALL_DATA{
	AAMAppearance A;
	AAMShapeAppearance SA;
	AAMShape S;
	//AAMTrainingData T;
	cv::Mat R;
	int n;
	int N;
};

void writeMat( cv::Mat const& mat, const char* filename, const char* varName = "A", bool bgr2rgb = true);

void AAM_MakeShapeModel2D_tire(AAMTrainingData &TrainingData, AAMShape &ShapeModel, 
							   AAMTexture &Text, AMM_Model2D_Options &options);

void AAM_MakeAppearanceModel2D(AAMTrainingData &TrainingData, AAMTexture &Text, AAMShape &ShapeModel, 
							   AAMAppearance &AppearanceData, AMM_Model2D_Options &options);

void AAM_Weights2D_tire(AAMTrainingData &TrainingData, AAMShape &ShapeData, AAMAppearance &AppearanceData, 
						AAMTexture &Text, cv::Mat &Ws, AMM_Model2D_Options &options);

void AAM_NormalizeAppearance2D(cv::Mat &gim);

void AAM_CombineShapeAppearance2D_tire(AAMTrainingData &TrainingData, AAM_ALL_DATA &Data, 
									   AAMTexture &Text, AMM_Model2D_Options &options);

void AAM_MakeSearchModel2D_tire(AAMTrainingData &TrainingData, AAM_ALL_DATA &Data, 
								AAMTexture &Text, AMM_Model2D_Options &options);

void ApplyModel2D(std::vector<AAM_ALL_DATA> &Data, AAMTexture &Text, cv::Mat &im, AAMTform &tformLarge, AMM_Model2D_Options &options);

#endif