#include "AAM_Train2D.h"
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

#include "dirent.h"
#include "delaunay2d.h"

void main(void)
{

	std::string dir_shape = "C:/Users/genawas/Dropbox/Ims4test/Shapesc";
	std::string dir_ims = "C:/Users/genawas/Dropbox/Ims4test/Imsc";
	std::string dir = "C:/Users/genawas/Dropbox/Ims4test";

	AAMTrainingData TrainingData;
	load_Data(dir_shape, dir_ims, TrainingData);

	//cv::Scalar delaunay_color(255, 0, 0);
	//draw_subdiv(TrainingData.Texture[1], tempC, F, delaunay_color);

	AMM_Model2D_Options options;
	options.set_default();
	AAMShape ShapeModel;
	AAM_MakeShapeModel2D_tire(TrainingData, ShapeModel, options);
	//load_triangulation(dir + "/faces.yml", ShapeModel.F);
	load_triangulation(dir + "/faces.yml", ShapeModel.F);
	AAMAppearance AppearanceData;
	AAM_MakeAppearanceModel2D(TrainingData, ShapeModel, AppearanceData, options);
	AAM_Weights2D_tire(TrainingData, ShapeModel, AppearanceData, options);

	//cv::Mat img=cv::imread("C:/Users/genawas/Downloads/40406598_fd4e74d51c.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat Z = (cv::Mat_ <double> (5, 5) <<  17, 24, 1, 8, 15,
		23, 5, 7, 14, 16, 4, 6, 13, 20, 22, 10, 12, 19, 21, 3, 11, 18, 25, 2, 9);
	cv::Mat eigval, eigvec, psi;
	pca_eigenvectors(Z, eigval, eigvec, psi);

	std::cout << eigvec << std::endl;
}