#ifndef _AAM_UTILITIES_
#define _AAM_UTILITIES_

#include <opencv2/core/core.hpp>
#include "AAM_Train2D.h"

void load_triangulation(std::string filename, cv::Mat &F);

void vec3i2mat(std::vector<cv::Vec3i> &F, cv::Mat &out);

void load_contour_txt(const std::string filename, std::vector<cv::Point2f> &contour);

int load_contour_yml(const std::string filename, std::vector<cv::Point2f> &contour);

void load_Data(std::string dir_shape, std::string dir_ims, AAMTrainingData &TrainingData);

void vec_average(std::vector<cv::Point2f> &Vertices, cv::Point2f &offestsv);

void shift_scale_vec(std::vector<cv::Point2f> &VerticesIn, std::vector<cv::Point2f> &VerticesOut, 
					 cv::Point2f &shift, float scale);

void scale_shift_vec(std::vector<cv::Point2f> &VerticesIn, 
					 std::vector<cv::Point2f> &VerticesOut, cv::Point2f &shift, float scale);

void sum_vertices(std::vector<cv::Point2f> &VerticesA, std::vector<cv::Point2f> &VerticesB);

int pca_eigenvectors(cv::Mat &A, cv::Mat &eigenvalues, cv::Mat &eigenvectors, cv::Mat &psi);

void get_components(cv::Mat &data, int numcomponents, cv::Mat &mean, cv::Mat &eigenvalues, cv::Mat &eigenvectors);

void loadAAMData(AAM_ALL_DATA &Data, std::string filename);

void saveAAMData(AAM_ALL_DATA &Data, std::string filename);

void saveAAMTextue(AAMTexture &Text, std::string filename);

void loadAAMTexture(AAMTexture &Text, std::string filename);

#endif