#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include "AAM_Train2D.h"
#include "AAM_Utilities.h"

#include "dirent.h"
#include "basicFunctions.h"
#include "imageProcessing.h"
//#include "delaunay2d.h"
#include "levmar.h"
//#include <lmmin.h>

//#include "warp_triangle_double.h"
/*
void AAM_align_data_inverse2D_tire(std::vector<cv::Point2f> &VerticesIn, std::vector<cv::Point2f> &VerticesOut, AAMTform &T)
{
	scale_shift_vec(VerticesIn, VerticesOut, T.shift, (float)T.scale);
}
*/

template <class T>
void my_shift_scale(cv::Mat &in, cv::Mat &out, cv::Mat &shift, double scale= 1.0)
{
	int n = in.rows;

	for(int i=0;i<n;i++){
		out.at<T>(i,0) = (in.at<T>(i,0) + shift.at<T>(0,0))*scale; //x
		out.at<T>(i,1) = (in.at<T>(i,1) + shift.at<T>(0,1))*scale; //y
	}

}

template <class T>
void my_scale_shift(cv::Mat &in, cv::Mat &out, cv::Mat &shift, double scale=1.0)
{
	int n = in.rows;

	for(int i=0;i<n;i++){
		out.at<T>(i,0) = (in.at<T>(i,0)*scale) + shift.at<T>(0,0); //x
		out.at<T>(i,1) = (in.at<T>(i,1)*scale) + shift.at<T>(0,1); //x
	}

}

void AAM_align_data_inverse2D_tire(cv::Mat &VerticesIn, cv::Mat &VerticesOut, AAMTform &T)
{
	cv::Mat shift  = (-1.0)*T.shift;
	my_scale_shift<double>(VerticesIn, VerticesOut, shift, 1.0/T.scale);
	
	//VerticesOut = (VerticesIn/T.scale) - T.shift;
}

void AAM_align_data2D_tire(cv::Mat &Vertices, cv::Mat &VerticesB, cv::Mat &VerticesOut, AAMTform &tform)
{
	// Remove rotation and translation and scale : Procrustes analysis
	int n = Vertices.rows;

	cv::Mat offsetv(1,2,CV_64FC1), offsetvB(1,2,CV_64FC1);
	cv::Mat VerticesS(n, 2, CV_64FC1), VerticesSB(n, 2, CV_64FC1);
	
	cv::Mat tempx = Vertices.col(0);
	cv::Mat tempy = Vertices.col(1);
	cv::Scalar xm = cv::mean(tempx);
	cv::Scalar ym = cv::mean(tempy);
	offsetv.at<double>(0,0) = -xm.val[0];
	offsetv.at<double>(0,1) = -ym.val[0];

	//cv::reduce(Vertices, offsetv, 0, CV_REDUCE_AVG);
	//offsetv = offsetv*(-1.0);
	//offsetv = cv::repeat(offsetv, n, 1);
	//cv::add(Vertices, offsetv, VerticesS);

	my_shift_scale<double>(Vertices, VerticesS, offsetv);

	tempx = VerticesB.col(0);
	tempy = VerticesB.col(1);
	xm = cv::mean(tempx);
	ym = cv::mean(tempy);
	offsetvB.at<double>(0,0) = - xm.val[0];
	offsetvB.at<double>(0,1) = - ym.val[0];

	my_shift_scale<double>(VerticesB, VerticesSB, offsetvB);

	//  Set scaling to base example
	double d = 0;
	double dB = 0;
	for(int i=0;i<n;i++){
		d+=sqrt((pow((double)VerticesS.at<double>(i,0),2.0)+pow((double)VerticesS.at<double>(i,1),2.0)));
		dB+= sqrt((pow((double)VerticesSB.at<double>(i,0),2.0)+pow((double)VerticesSB.at<double>(i,1),2.0)));
	}

	d/=n;
	dB/=n;

	double offsets=dB/d;
	
	cv::multiply(VerticesS, offsets, VerticesOut);
	//VerticesOut = VerticesS*offsets;
	//writeMat(VerticesOut, "cp.mat", "cp");

	tform.shift = offsetv;
	tform.scale = offsets;
}
/*
void AAM_align_data2D_tire(std::vector<cv::Point2f> &Vertices,std::vector<cv::Point2f> &VerticesB, std::vector<cv::Point2f> &VerticesOut, AAMTform &tform)
	// Remove rotation and translation and scale : Procrustes analysis 
{
	cv::Point2f offsetv, offsetvB;
	std::vector<cv::Point2f> VerticesS, VerticesSB;
	vec_average(Vertices, offsetv);
	offsetv = cv::Point2f(-offsetv.x, -offsetv.y);
	shift_scale_vec(Vertices, VerticesS, offsetv, 1.0);

	vec_average(VerticesB, offsetvB);
	offsetvB = cv::Point2f(-offsetvB.x, -offsetvB.y);
	shift_scale_vec(VerticesB, VerticesSB, offsetvB, 1.0);

	//  Set scaling to base example
	int n = Vertices.size();
	double d = 0;
	double dB = 0;
	for(int i=0;i<n;i++){
		d+=sqrt((pow((double)VerticesS[i].x,2.0)+pow((double)VerticesS[i].y,2.0)));
		dB+= sqrt((pow((double)VerticesSB[i].x,2.0)+pow((double)VerticesSB[i].y,2.0)));
	}

	d/=n;
	dB/=n;

	double offsets=dB/d;

	shift_scale_vec(VerticesS, VerticesOut, cv::Point2f(0,0), (float)offsets);

	tform.shift = offsetv;
	tform.scale = offsets;
}
*/

void AAM_MakeShapeModel2D_tire(AAMTrainingData &TrainingData, AAMShape &ShapeModel, 
							   AAMTexture &Text, AMM_Model2D_Options &options)
{
	// Number of datasets
	int s=TrainingData.N;

	// Number of landmarks
	int n = TrainingData.n;

	// Shape model
	// Remove rotation and translation and scale : Procrustes analysis 
	//std::vector<cv::Point2f> MeanVertices=TrainingData.Shape[0];
	cv::Mat MeanVertices=TrainingData.Shape[0].clone();

	//std::vector<cv::Point2f> VerticesC;
	cv::Mat VerticesC(n, 2, CV_64FC1); //[y;x]

	//cv::Point2f Sh;
	cv::Mat Sh;
	double Sc;
	std::vector<AAMTrainingData> TrainingDataS;
	for(int k=0;k<2;k++){
		Sc = 0;
		//Sh = cv::Point2f(0,0);
		Sh = cv::Mat::zeros(1,2,CV_64FC1);
		cv::Mat AllVertices = cv::Mat::zeros(n,2, CV_64FC1);
		//std::vector<cv::Point2f> AllVertices(nl,cv::Point2f(0,0));
		for(int i=0;i<s;i++)
		{
			AAMTform tform;
			AAM_align_data2D_tire(TrainingData.Shape[i], MeanVertices, VerticesC, tform);
			TrainingData.Shift[i]=tform.shift;
			TrainingData.Scale[i]=tform.scale;
			//sum_vertices(VerticesC, AllVertices);
			//writeMat(MeanVertices, "cp.mat", "cp");

			cv::add(AllVertices, VerticesC, AllVertices);
			cv::add(Sh, tform.shift, Sh);
			//Sh.x += tform.shift.x;
			//Sh.y += tform.shift.y;
			Sc += tform.scale;
		}
		//writeMat(AllVertices, "cp.mat", "cp");
		//cv::Point2f temp(0,0);shift_scale_vec(AllVertices, AllVertices, temp, 1.0f/(float)s);
		cv::multiply(AllVertices, 1.0/(double)s, AllVertices); 
		//Sh.x /=s;Sh.y /=s;
		cv::multiply(Sh, 1.0/(double)s, Sh);
		Sc/=s;
		AAMTform T;
		T.scale = Sc;
		T.shift = Sh;
		AAM_align_data_inverse2D_tire(AllVertices, MeanVertices, T);
		//writeMat(MeanVertices, "cp.mat", "cp");
		int er=0;
	}
	//writeMat(MeanVertices, "cp.mat", "cp"); 

	for(int i=0;i<s;i++){
		AAMTform tform;
		AAM_align_data2D_tire(TrainingData.Shape[i], MeanVertices, TrainingData.ShapeC[i], tform);
		TrainingData.Shift[i]=tform.shift;
		TrainingData.Scale[i]=tform.scale;
		//writeMat(TrainingData.ShapeC[i], "cp.mat", "cp"); 
		//int er=0;
	}

	// Construct a matrix with all contour point data of the training data set
	cv::Mat X(n*2, s, CV_64FC1);
	for(int q=0;q<s;q++){
		//for(int i=0;i<nl;i++){X.at<double>(i,q)=TrainingData.ShapeC[q][i].y;X.at<double>(i+nl,q)=TrainingData.ShapeC[q][i].x;}
		cv::Mat xt = X(cv::Range(n,2*n),cv::Range(q,q+1));
		cv::Mat yt = X(cv::Range(0,n),cv::Range(q,q+1));
		TrainingData.ShapeC[q].col(0).copyTo(xt);
		TrainingData.ShapeC[q].col(1).copyTo(yt);
		//TrainingData.ShapeC[q].col(0).copyTo(tempX);//y
		//TrainingData.ShapeC[q].col(1).copyTo(tempY);//x
		//writeMat(TrainingData.ShapeC[0].col(0), "cp.mat", "cp");
		//int er=0;
	}
	//writeMat(X, "X.mat", "X1"); 

	//PCA
	cv::Mat eigenvalues;
	cv::Mat eigenvectors;
	cv::Mat psi;

	get_components(X, s, psi, eigenvalues, eigenvectors);

	//writeMat(psi, "ei.mat", "ei"); 

	ShapeModel.Evalues = eigenvalues;
	ShapeModel.Evectors = eigenvectors;
	ShapeModel.data_mean = psi;
	ShapeModel.data = X;

	double mu, var;

	basicStat(TrainingData.Scale, mu, var, 0, s-1);
	ShapeModel.varSc=var;
	ShapeModel.MeanVertices = MeanVertices;

	// Build Delaunay
	/*
	std::vector<cv::Point2f> tempC(TrainingData.n);
	for(int i=0;i<TrainingData.n;i++){
	tempC[i]=cv::Point2f(TrainingData.Shape[0][i]);
	}
	cv::Subdiv2D subdiv;
	build_delaunay(TrainingData.Texture[0], tempC, subdiv);
	build_tri_face_data(subdiv, tempC, ShapeModel.F, TrainingData.Texture[ind].size());
	*/

	double minVal, maxVal;
	cv::minMaxLoc(ShapeModel.data_mean, &minVal, &maxVal);
	minVal = minVal*-1.0;
#undef max
	int ts = ceil(std::max(maxVal, minVal)*2.0*(double)options.texturesize);
	Text.textureSize = cv::Size(ts,ts);

	return;
	//writeMat(cv::Mat(eig_cumsum), "ei.mat", "ei"); 
	//int yj=0;

	/*
	ShapeData.Lines = TrainingData(1).Lines;
	ts=ceil(max(max(ShapeData.x_mean(:)),-min(ShapeData.x_mean(:)))*2*options.texturesize);
	ShapeData.TextureSize=[ts ts];
	ShapeData.Tri= delaunay(x_mean(1:end/2),x_mean(end/2+1:end));
	*/
}

void drawObject(cv::Size &imsize, cv::Mat &mask, std::vector<cv::Point2f> &base_points)
{
	cv::Mat temp = cv::Mat::zeros(imsize, CV_8UC1);
	//cv::Mat temp1;// = cv::Mat::zeros(imsize, CV_8UC1);
	std::vector<std::vector<cv::Point> > tC(1);
	int n = base_points.size();
	for(int i=0;i<n;i++){
		tC[0].push_back(cv::Point(base_points[i]));
	}
	cv::drawContours(temp, tC, 0, cv::Scalar(255), -1);
	cv::drawContours(temp, tC, 0, cv::Scalar(0), 1);
	//threshold(temp, temp1, 100, 255, cv::THRESH_BINARY_INV);
	//cv::drawContours(temp1, tC, 0, cv::Scalar(1), -1);

	mask = temp;// - temp1;
	//cv::imshow("test", temp);cv::waitKey();
}

template <class T>
void crop_mask(cv::Mat &img, cv::Mat &mask, int k, cv::Mat &out)
{
	int m = img.rows;
	int n = img.cols;

	out = cv::Mat_ <T> (k, 1);
	int s=0;
	for(int j=0;j<n;j++){
		for(int i=0;i<m;i++){
			if(mask.at<uchar>(i,j) > 0){
				out.at<T>(s,0) = img.at<T>(i,j);
				s++;
			}
		}
	}
}

void warp_triangle_double(double *Iin, int* size_Iin, double *Iout, int* size_Iout, 
						  double *XY, int* size_XY, double *UV, int* size_UV, double *TRI, int* size_TRI);

void AAM_Appearance2Vector2D(cv::Mat &in, cv::Size &ts, int k, std::vector<cv::Point2f> &source, std::vector<cv::Point2f> &target, 
							 /*std::vector<cv::Vec3i> &F*/cv::Mat &F, cv::Mat &greyvector, cv::Mat &mask, bool draw_flag=false)
{
	cv::Mat out=cv::Mat::zeros(ts, CV_64FC1);
	int sizeIout[2] = {out.rows, out.cols};
	//out = out.t();
	double *Iout = (double*)out.data;

	cv::Mat tempin;in.convertTo(tempin, CV_64FC1);
	//writeMat(tempin, "in.mat", "in");
	int sizeIin[3] = {tempin.rows, tempin.cols, 1};
	//tempin=tempin.t();
	double *Iin = (double*)tempin.data; 

	cv::Mat xy;
	contour2mat(source, xy);
	xy.convertTo(xy, CV_64FC1);
	//xy=xy.t();
	double *XY = (double*)xy.data; 
	int sizeXY[2] = {source.size(), 2};

	cv::Mat uv;
	contour2mat(target, uv);
	uv.convertTo(uv, CV_64FC1);
	//uv=uv.t();
	double *UV = (double*)uv.data; 
	int sizeUV[2]= {target.size(), 2}; 

	cv::Mat tri=F;
	//vec3i2mat(F, tri);
	int sizeTRI[2] = {tri.rows, 3}; 
	//F.convertTo(tri, CV_64FC1);
	//tri=tri.t();
	double *TRI =(double*)tri.data;
	//writeMat(tri, "gr.mat", "gr");
	//writeMat(uv, "in.mat", "in");
	//writeMat(uv, "a.mat", "a", false);
	warp_triangle_double( Iin, sizeIin, Iout, sizeIout, XY, sizeXY, UV, sizeUV, TRI, sizeTRI);
	uv.release(); tri.release(); xy.release();tempin.release();

	cv::Mat drawing;
	//out=out.t();

	//imageplot(out, drawing);
	//cv::imshow("test", drawing);waitKey();
	//*/
	//cv::Mat out = cv::Mat::zeros(ts, CV_8UC1);
	//wrap_piecewise_nonlin(in, out, F, source, target);
	
	if(draw_flag==true){
		cv::imshow("test", out);waitKey();
	}

	//cout << mask.type() << endl;
	//cout << CV_8UC1 << endl;

	//crop_mask<uchar>(out, mask, k, greyvector);
	crop_mask<double>(out, mask, k, greyvector);
	//writeMat(out, "gr.mat", "gr");
}

void AAM_Appearance2Vector2D(cv::Mat &in, cv::Size &ts, int k, cv::Mat &source, cv::Mat &target, 
							cv::Mat &F, cv::Mat &greyvector, cv::Mat &mask, bool draw_flag=false)
{
	cv::Mat out=cv::Mat::zeros(ts, CV_64FC1);
	int sizeIout[2] = {out.rows, out.cols};
	//out = out.t();
	double *Iout = (double*)out.data;

	cv::Mat tempin;in.convertTo(tempin, CV_64FC1);
	//writeMat(tempin, "in.mat", "in");
	int sizeIin[3] = {tempin.rows, tempin.cols, 1};
	//tempin=tempin.t();
	double *Iin = (double*)tempin.data; 


	double *XY = (double*)source.data; 
	int sizeXY[2] = {source.rows, 2};

	double *UV = (double*)target.data; 
	int sizeUV[2]= {target.rows, 2}; 

	
	int sizeTRI[2] = {F.rows, 3}; 

	double *TRI =(double*)F.data;
	//writeMat(F, "gr.mat", "gr");
	//writeMat(uv, "in.mat", "in");
	//writeMat(uv, "a.mat", "a", false);
	warp_triangle_double( Iin, sizeIin, Iout, sizeIout, XY, sizeXY, UV, sizeUV, TRI, sizeTRI);

	cv::Mat drawing;
	if(draw_flag==true){
		imageplot(out, drawing);
		cv::imshow("test", drawing);waitKey();
	}

	crop_mask<double>(out, mask, k, greyvector);
	//writeMat(out, "gr.mat", "gr");
}

void AAM_NormalizeAppearance2D(cv::Mat &gim)
{
	//Normalize appearance data grey values
	cv::Mat mu, su;
	cv::meanStdDev(gim, mu, su);

	double m = mu.at<double>(0,0);
	double s = su.at<double>(0,0);

	//std::cout << mu << std::endl;
	//std::cout << su << std::endl;

	for(int i = 0;i<gim.rows;i++){
		double *ptr = gim.ptr<double>(i);
		for(int j=0;j<gim.cols;j++){
			ptr[j] = (ptr[j] - m)/s;
		}
	}
}

void AAM_MakeAppearanceModel2D(AAMTrainingData &TrainingData, AAMTexture &Text, AAMShape &ShapeModel, 
							   AAMAppearance &AppearanceData, AMM_Model2D_Options &options)
{
	// Coordinates of mean contour
	/*
	std::vector<cv::Point2f> base_points(TrainingData.n);
	for(int i=0;i<TrainingData.n;i++){
		base_points[i].y = (float)ShapeModel.data_mean.at<double>(i,0);
		base_points[i].x = (float)ShapeModel.data_mean.at<double>(i+TrainingData.n,0);
	} 
	*/

	int n = TrainingData.n;
	cv::Mat base_points(n,2,CV_64FC1);
	ShapeModel.data_mean(cv::Range(0,n), cv::Range::all()).copyTo(base_points.col(1));
	ShapeModel.data_mean(cv::Range(n,2*n), cv::Range::all()).copyTo(base_points.col(0));
	//writeMat(base_points, "cp.mat", "cp");

	// Normalize the base points to range 0..1
	double minX, maxX,  minY, maxY;
	//MinMaxContour(base_points, minX, maxX,  minY, maxY);
	cv::minMaxLoc(base_points.col(0), &minX);
	cv::minMaxLoc(base_points.col(1), &minY);
	
	base_points.col(0) = base_points.col(0) - minX;
	base_points.col(1) = base_points.col(1) - minY;

	/*
	for(int i=0;i<TrainingData.n;i++){
		base_points[i].x = (base_points[i].x-minX);
		base_points[i].y = (base_points[i].y-minY);
	}
	*/
	cv::minMaxLoc(base_points.col(0), &minX, &maxX);
	cv::minMaxLoc(base_points.col(1), &minY, &maxY);
	//MinMaxContour(base_points, minX, maxX,  minY, maxY);
	
	base_points.col(0) = base_points.col(0)*((double)Text.textureSize.width-1)/ maxX;
	base_points.col(1) = base_points.col(1)*((double)Text.textureSize.height-1)/maxY;
	/*
	for(int i=0;i<TrainingData.n;i++){
		base_points[i].x /= maxX;
		base_points[i].y /= maxY;
		//Transform the mean contour points into the coordinates in the texture image.
		base_points[i].x=(Text.textureSize.width-1)*base_points[i].x;
		base_points[i].y=(Text.textureSize.height-1)*base_points[i].y;
	}
	*/

	//writeMat(base_points, "cp.mat", "cp");
	//writeMat(ShapeModel.data_mean, "cp.mat", "cp"); 

	cv::Mat mask = AppearanceData.ObjectPixels;
	mask.convertTo(mask, CV_8UC1);
	//drawObject(ShapeModel.textureSize, mask, base_points);
	//writeMat(Text.F, "m.mat","m");

	int k=cv::countNonZero(mask);

	cv::Mat grey(k, TrainingData.N, CV_64FC1);
	for(int i=0;i<TrainingData.N;i++){
		std::cout << i << std::endl;
		cv::Mat greyvector;

		AAM_Appearance2Vector2D(TrainingData.Texture[i], Text.textureSize, k, TrainingData.Shape[i], base_points, Text.F, greyvector, mask);
		//writeMat(greyvector, "ge.mat","ge");
		cv::Mat temp = grey(cv::Range::all(), cv::Range(i,i+1));
		greyvector.convertTo(temp, CV_64FC1);
		AAM_NormalizeAppearance2D(temp);
		//writeMat(temp, "ei.mat", "ei");
	}

	//PCA
	cv::Mat eigenvalues;
	cv::Mat eigenvectors;
	cv::Mat psi;
	get_components(grey, TrainingData.N, psi, eigenvalues, eigenvectors);
	//writeMat(eigenvectors, "ei.mat", "ei"); 

	// Store the Eigen Vectors and Eigen Values
	AppearanceData.k = k;
	AppearanceData.Evectors=eigenvectors;
	AppearanceData.Evalues=eigenvalues;
	AppearanceData.g_mean=psi;
	AppearanceData.g = grey;
	AppearanceData.ObjectPixels=mask;
	AppearanceData.base_points=base_points;

	//writeMat(AppearanceData.g_mean, "ei.mat", "ei");
}

/*
void transform2pos(cv::Mat &b_mean,  cv::Mat &Evectors, cv::Mat &c, AAMTform &tform, std::vector<cv::Point2f> &pos, int n)
{
	cv::Mat x = b_mean + Evectors*c;

	pos=std::vector<cv::Point2f>(n);
	for(int i1=0;i1<n;i1++){
		pos[i1].y=(float)x.at<double>(i1,0);
		pos[i1].x=(float)x.at<double>(i1+n,0);
	}

	// Transform the Shape back to real image coordinates
	AAM_align_data_inverse2D_tire(pos, pos, tform);
}
*/

void transform2pos(cv::Mat &b_mean,  cv::Mat &Evectors, cv::Mat &c, AAMTform &tform, cv::Mat &pos, int n)
{
	cv::Mat x = b_mean + Evectors*c;

	pos=cv::Mat(n, 2, CV_64FC1);
	x(cv::Range(0,n), cv::Range::all()).copyTo(pos.col(1));
	x(cv::Range(n,2*n), cv::Range::all()).copyTo(pos.col(0));
	
	// Transform the Shape back to real image coordinates
	AAM_align_data_inverse2D_tire(pos, pos, tform);
}

void AAM_Weights2D_tire(AAMTrainingData &TrainingData, AAMShape &ShapeData, AAMAppearance &AppearanceData, 
						AAMTexture &Text, cv::Mat &Ws, AMM_Model2D_Options &options)
{
	int N = TrainingData.N;
	int n = TrainingData.n;

	cv::Mat Change = cv::Mat::zeros(N, ShapeData.Evectors.cols, CV_64FC1);

	for(int i=0;i<N;i++)
	{
		//Remove translation and rotation, as done when training the model.
		//std::vector<cv::Point2f> pos;
		cv::Mat pos;
		AAMTform tform;
		AAM_align_data2D_tire(TrainingData.Shape[i], ShapeData.MeanVertices, pos, tform);

		// Describe the model by a vector b with model parameters
		cv::Mat X(n*2, 1, CV_64FC1);
		/*
		for(int i1=0;i1<n;i1++){
			X.at<double>(i1,0)=(double)pos[i1].y;
			X.at<double>(i1+n,0)=(double)pos[i1].x;
		}
		*/
		pos.col(1).copyTo(X(cv::Range(0,n), cv::Range::all()));
		pos.col(0).copyTo(X(cv::Range(n,2*n), cv::Range::all()));
		
		cv::subtract(X, ShapeData.data_mean, X);
		cv::Mat b = ShapeData.Evectors.t()*X;

		//std::cout << b << std::endl;

		// Get the intensities of the untransformed shape.
		// Because noisy eigenvectors from the shape were removed, the 
		// contour is on a little different position and
		// intensities probabbly differ a little bit from the orignal appearance
		//std::vector<cv::Point2f> pos_normal;
		cv::Mat pos_normal;
		transform2pos(ShapeData.data_mean, ShapeData.Evectors, b, tform, pos_normal, n);

		cv::Mat g_normal;
		AAM_Appearance2Vector2D(TrainingData.Texture[i], Text.textureSize, AppearanceData.k, 
			pos_normal, AppearanceData.base_points, Text.F, g_normal, AppearanceData.ObjectPixels, false);
		g_normal.convertTo(g_normal, CV_64FC1);
		AAM_NormalizeAppearance2D(g_normal);

		double K[2] = {-0.5, 0.5};
		for (int j = 0; j<ShapeData.Evectors.cols; j++){
			for(int k=0; k<2; k++)
			{

				// Change on model parameter a little bit, to see the influence
				// from the shape parameters on appearance parameters
				cv::Mat b_offset=b.clone();  
				b_offset.at<double>(j,0)=b_offset.at<double>(j,0)+K[k];

				// Transform the model parameter vector b , back to contour positions
				//std::vector<cv::Point2f> pos_offset;
				cv::Mat pos_offset;
				transform2pos(ShapeData.data_mean, ShapeData.Evectors, b_offset, tform, pos_offset, n);

				cv::Mat g_offset;
				AAM_Appearance2Vector2D(TrainingData.Texture[i], Text.textureSize, AppearanceData.k, 
					pos_offset, AppearanceData.base_points, Text.F, g_offset, AppearanceData.ObjectPixels, false);
				g_offset.convertTo(g_offset, CV_64FC1);
				AAM_NormalizeAppearance2D(g_offset);

				double s=0;

				//writeMat(g_offset, "g.mat", "g", false);
				//writeMat(g_normal, "g1.mat", "g1", false);
				for(int h=0;h<AppearanceData.k;h++){
					s=s+pow(g_offset.at<double>(h,0)-g_normal.at<double>(h,0),2.0);
				}
				s=s/AppearanceData.k;
				s=sqrt(s);
				Change.at<double>(i,j) = Change.at<double>(i,j)+s;
			}
		}
	}
	//writeMat(Change, "c.mat", "c", false);
	Ws=cv::Mat::zeros(ShapeData.Evectors.cols, ShapeData.Evectors.cols, CV_64FC1);
	for (int j = 0;j<ShapeData.Evectors.cols;j++){
		cv::Scalar mu = mean(Change.col(j));
		Ws.at<double>(j,j) = mu[0];
	}
	//writeMat(Ws, "w.mat", "w", false);
}

void AAM_CombineShapeAppearance2D_tire(AAMTrainingData &TrainingData, AAM_ALL_DATA &Data, 
									   AAMTexture &Text, AMM_Model2D_Options &options)
{
	//This functions combines the shape and appearance of the objects, by
	//adding the weighted vector describing shape and appearance, followed by
	//PCA

	// Get weight matrix. The Weights are a scaling between texture and shape
	//to give a change in shape parameters approximately the same 
	//influences as texture parameters.
	
	AAMShape ShapeData = Data.S;
	AAMAppearance AppearanceData = Data.A; 
	AAMShapeAppearance ShapeAppearanceData = Data.SA;

	cv::Mat Ws;
	AAM_Weights2D_tire(TrainingData, ShapeData, AppearanceData, Text, Ws, options);

	// Combine the Contour and Appearance data
	cv::Mat b=cv::Mat::zeros(ShapeData.Evectors.cols+AppearanceData.Evectors.cols, TrainingData.N, CV_64FC1);

	cv::Mat tempS = ShapeData.Evectors.t();
	cv::Mat tempA = AppearanceData.Evectors.t();

	for (int i=0;i<TrainingData.N;i++){
		cv::Mat b1 = Ws * tempS * (ShapeData.data.col(i)-ShapeData.data_mean);
		cv::Mat b2 = tempA * (AppearanceData.g.col(i)-AppearanceData.g_mean);
		cv::Mat temp = b(cv::Range(0, ShapeData.Evectors.cols), cv::Range(i,i+1));
		b1.copyTo(temp);
		temp = b(cv::Range(ShapeData.Evectors.cols, ShapeData.Evectors.cols+AppearanceData.Evectors.cols), cv::Range(i,i+1));
		b2.copyTo(temp);
		//int y=0;
	}
	//writeMat(b, "bg.mat", "bg", false);

	//PCA
	cv::Mat eigenvalues;
	cv::Mat eigenvectors;
	cv::Mat psi;

	get_components(b, TrainingData.N, psi, eigenvalues, eigenvectors);

	//writeMat(eigenvectors, "ei.mat", "ei");

	Data.SA.Evectors=eigenvectors;
	Data.SA.Evalues=eigenvalues;
	Data.SA.b_mean=psi;
	Data.SA.b=b;
	Data.SA.Ws=Ws;
}

void RealAndModel(AAMTrainingData &TrainingData, AAM_ALL_DATA &Data, AAMTexture &Text, AMM_Model2D_Options &options, 
				  int i, cv::Mat &pos, cv::Mat &g_offset, cv::Mat &g)
{
	int n = TrainingData.n;
	AAMShape ShapeData = Data.S;
	AAMAppearance AppearanceData = Data.A; 
	AAMShapeAppearance ShapeAppearanceData = Data.SA;

	//Sample the image intensities in the training set
	AAM_Appearance2Vector2D(TrainingData.Texture[i], Text.textureSize, AppearanceData.k, 
		pos, AppearanceData.base_points, Text.F, g_offset,  AppearanceData.ObjectPixels);
	g_offset.convertTo(g_offset, CV_64FC1);
	//writeMat(g_offset, "go.mat", "go", false);
	AAM_NormalizeAppearance2D(g_offset);

	//Combine the Shape and Intensity (Appearance) vector
	cv::Mat tempS = ShapeData.Evectors.t();
	cv::Mat temp(n*2, 1, CV_64FC1);
	/*
	for(int i1=0;i1<n;i1++){
		temp.at<double>(i1,0)=TrainingData.ShapeC[i][i1].y;
		temp.at<double>(i1+TrainingData.n,0)=TrainingData.ShapeC[i][i1].x;
	}
	*/
	TrainingData.ShapeC[i].col(1).copyTo(temp(cv::Range(0, n), cv::Range::all()));//y
	TrainingData.ShapeC[i].col(0).copyTo(temp(cv::Range(n, 2*n), cv::Range::all()));//x

	cv::subtract(temp, ShapeData.data_mean, temp);

	cv::Mat b1 = ShapeAppearanceData.Ws * tempS * temp;
	cv::Mat tempA = AppearanceData.Evectors.t();
	cv::Mat gtemp;
	//writeMat(g_offset, "go.mat", "go", false);
	//writeMat(AppearanceData.g_mean, "gm.mat", "gm", false);
	//writeMat(AppearanceData.g_mean, "ei.mat", "ei");
	//cout << AppearanceData.g_mean.type() << endl;
	cv::subtract(g_offset, AppearanceData.g_mean, gtemp);
	cv::Mat b2 = tempA * gtemp;
	cv::Mat b;
	concat_mat(b1, b2, b, 2);

	cv::Mat tempSA = ShapeAppearanceData.Evectors.t();
	//Calculate the ShapeAppearance parameters
	cv::Mat c2 = tempSA*(b -ShapeAppearanceData.b_mean);

	// Go from ShapeAppearance parameters to Appearance parameters
	b = ShapeAppearanceData.b_mean + ShapeAppearanceData.Evectors*c2;
	b2 = b(cv::Range(ShapeAppearanceData.Ws.rows, b.rows), cv::Range::all());

	//From apperance parameters to intensities
	g = AppearanceData.g_mean + AppearanceData.Evectors*b2;
}

void transformShapeAppearance2Shape(AAMTform &tform, cv::Mat &c_offset, cv::Mat &SAb_mean, cv::Mat &SAEvectors, 
									cv::Mat &Ws, ::Mat &Sb_mean, cv::Mat &SEvectors, int n, cv::Mat &pos)
{
	// Transform back from  ShapeAppearance parameters to Shape parameters  
	cv::Mat b_offset = SAb_mean + SAEvectors*c_offset;
	cv::Mat b1_offset = b_offset(cv::Range(0, std::max(Ws.rows, Ws.cols)), cv::Range::all()).clone();
	//writeMat(b1_offset, "x1.mat", "x1", false);
	cv::Mat Ws1=Ws.inv(cv::DECOMP_SVD);
	b1_offset = Ws1*b1_offset;
	//writeMat(b1_offset, "x1.mat", "x1", false);
	//std::cout << b1_offset << std::endl;

	transform2pos(Sb_mean, SEvectors, b1_offset, tform, pos, n);

}

void AAM_MakeSearchModel2D_tire(AAMTrainingData &TrainingData, AAM_ALL_DATA &Data, 
								AAMTexture &Text, AMM_Model2D_Options &options)
{
	AAMShape ShapeData = Data.S;
	AAMAppearance AppearanceData = Data.A; 
	AAMShapeAppearance ShapeAppearanceData = Data.SA;

	//Structure which will contain all weighted errors of model versus real
	//intensities, by several offsets of the parameters
	const int L = 6;
	cv::Mat drdp=cv::Mat::zeros(ShapeAppearanceData.Evectors.cols+4, AppearanceData.k, CV_64FC1);

	// We use the trainingdata images, to train the model. Because we want
	//the background information to be included
	std::vector<double> de(L);
	//Loop through all training images
	for (int i=0;i<TrainingData.N;i++){
		//Loop through all model parameters, bot the PCA parameters as pose
		//parameters
		for(int j = 0;j<ShapeAppearanceData.Evectors.cols+4;j++){
			if(j<ShapeAppearanceData.Evectors.cols){
				// Model parameters, offsets
				de[0]=-0.5;de[1]=-0.3;de[2]=-0.1;de[3]=0.1;de[4]=0.3;de[5]=0.5;
				// First we calculate the real ShapeAppearance parameters of the
				//training data set
				cv::Mat tempSA = ShapeAppearanceData.Evectors.t();
				cv::Mat c = tempSA*(ShapeAppearanceData.b.col(i) - ShapeAppearanceData.b_mean);
				// Standard deviation form the eigenvalue
				double c_std = sqrt(ShapeAppearanceData.Evalues.at<double>(j,0));
				//writeMat(ShapeAppearanceData.Evectors, "ed.mat", "ed");
				for (int k=0;k<L;k++)
				{
					// Offset the ShapeAppearance parameters with a certain
					// value times the std of the eigenvector
					cv::Mat c_offset=c.clone();
					c_offset.at<double>(j,0)=c_offset.at<double>(j,0)+c_std *de[k];

					AAMTform tform;
					tform.scale = TrainingData.Scale[i];
					tform.shift = TrainingData.Shift[i].clone();
					//std::vector<cv::Point2f> pos;
					cv::Mat pos;
					transformShapeAppearance2Shape(tform, c_offset, ShapeAppearanceData.b_mean, ShapeAppearanceData.Evectors, 
						ShapeAppearanceData.Ws, ShapeData.data_mean, ShapeData.Evectors, TrainingData.n, pos);

					//Get the intensities in the real image. Use those
					//intensities to get ShapeAppearance parameters, which
					//are then used to get model intensities
					cv::Mat g_offset, g;
					//writeMat(pos, "p.mat", "p", false);
					RealAndModel(TrainingData, Data, Text, options, i, pos, g_offset, g);
					//writeMat(g_offset, "g1.mat", "g1", false);

					// A weighted sum of difference between model an real
					// intensities gives the "intensity / offset" ratio
					double w = exp ((-pow(de[k],2.0)) / (2.0*pow(c_std,2.0)))/de[k];
					cv::Mat dr = (g-g_offset)*w;
					//cv::Mat tempD = drdp(cv::Range(j, j+1), cv::Range::all());
					//cv::add(tempD, dr.t(), tempD);
					//writeMat(dr, "dr1.mat", "dr1", false);

					dr = dr.t();
					double *ptrD = drdp.ptr<double>(j);
					double *ptrR = dr.ptr<double>(0);
					for(int ii=0;ii<drdp.cols;ii++){
						ptrD[ii] = ptrD[ii]+ptrR[ii];
					}

				}
			}
			else{
				// Pose parameters offsets
				int j2=j-ShapeAppearanceData.Evectors.cols;
				switch(j2){
				case 0: // Translation x
					de[0]=-1;de[1]=-0.6;de[2]=-0.2;de[3]=0.2;de[4]=0.6;de[5]=1;
					break;
				case 1: // Translation y
					de[0]=-1;de[1]=-0.6;de[2]=-0.2;de[3]=0.2;de[4]=0.6;de[5]=1;
					break;
				case 2: // Scaling & Rotation Sx
					de[0]=-0.1;de[1]=-0.06;de[2]=-0.02;de[3]=0.02;de[4]=0.06;de[5]=0.1;
					break;
				case 3: // Scaling & Rotation Sy
					de[0]=-0.1;de[1]=-0.06;de[2]=-0.02;de[3]=0.02;de[4]=0.06;de[5]=0.1;
					break;
				}

				for (int k=0;k<L;k++){
					AAMTform tform;
					tform.scale = TrainingData.Scale[i];
					tform.shift = TrainingData.Shift[i].clone();
					switch(j2){
					case 0: // Translation y
						//tform.shift.y=tform.shift.y+de[k];
						tform.shift.at<double>(0,1) = tform.shift.at<double>(0,1) + de[k];
						break;
					case 1: // Translation x
						//tform.shift.x=tform.shift.x+de[k];
						tform.shift.at<double>(0,0) = tform.shift.at<double>(0,0) + de[k];
						break;
					case 2: // Scaling & Rotation Sx
						//tform.offsetsx=tform.offsetsx+de(k);
						break;
					case 3: // Scaling & Rotation Sy
						//tform.offsetsy=tform.offsetsy+de(k);
						break;
					}

					// From Shape tot real image coordinates, with a certain
					// pose offset
					//std::vector<cv::Point2f> pos(TrainingData.n);
					cv::Mat pos(Data.n, 2, CV_64FC1);

					AAM_align_data_inverse2D_tire(TrainingData.ShapeC[i], pos, tform);

					// Get the intensities in the real image. Use those
					// intensities to get ShapeAppearance parameters, which
					// are then used to get model intensities
					cv::Mat g_offset, g;
					RealAndModel(TrainingData, Data, Text, options, i, pos, g_offset, g);

					//A weighted sum of difference between model an real
					//intensities gives the "intensity / offset" ratio
					double w = exp ((-pow(de[k],2.0)) / (2.0*pow(2.0,2.0)))/de[k];
					cv::Mat dr = (g-g_offset)*w;
					//writeMat(dr, "dr.mat", "dr", false);
					//cv::Mat tempD = drdp(cv::Range(j, j+1), cv::Range::all());
					//cv::add(tempD, dr.t(), tempD);
					dr = dr.t();
					double *ptrD = drdp.ptr<double>(j);
					double *ptrR = dr.ptr<double>(0);
					for(int ii=0;ii<drdp.cols;ii++){
						ptrD[ii] = ptrD[ii]+ptrR[ii];
					}
					if (j==6){
						//writeMat(drdp, "d.mat", "d", false);
						//writeMat(dr, "dr1.mat", "dr1", false);
						int red =0;
					}

					//tempD = tempD + dr.t();
					//writeMat(drdp, "d.mat", "d", false);
					//writeMat(dr, "dr1.mat", "dr1", false);
					//int red =0;
				}
			}

		}
		cout << i << endl;
	}

	//writeMat(drdp, "d.mat", "d", false);

	// Combine the data to the intensity/parameter matrix, 
	//using a pseudo inverse
	// for i=1:length(TrainingData);
	//     drdpt=squeeze(mean(drdp(:,:,i,:),2));
	//     R(:,:,i) = (drdpt * drdpt')\drdpt;
	// end
	// Combine the data intensity/parameter matrix of all training datasets.
	// In case of only a few images, it will be better to use a weighted mean
	// instead of the normal mean, depending on the probability of the trainingset
	// R=mean(R,3);    
	double alpha = double(TrainingData.N)*double(L);

	for(int i=0;i<drdp.rows;i++){
		double *ptr = drdp.ptr<double>(i);
		for(int j=0;j<drdp.cols;j++){
			ptr[j] = ptr[j]/alpha;
		}
	}

	//SVD svd(drdp);
	//Mat R = svd.vt.t()*Mat::diag(1./svd.w)*svd.u.t();
	Data.R = drdp.inv(cv::DECOMP_SVD);
	Data.R=Data.R.t();
	//writeMat(Data.R, "d.mat", "d", false);
}

typedef struct {
	cv::Mat b_mean; 
	cv::Mat	x_mean; 
	cv::Mat	EvectorsSA; 
	cv::Mat	EvectorsS; 
	cv::Mat	Ws; 
	cv::Mat	y;
	cv::Mat x;
	void (*f)(double *c, cv::Mat &bm, cv::Mat &xm, cv::Mat &eSA, cv::Mat &eS, cv::Mat &ws, cv::Mat &x);
} my_data_struct;

void cost_lmmin(double *par, cv::Mat &b_mean, cv::Mat &x_mean, 
				cv::Mat &EvectorsSA, cv::Mat &EvectorsS, cv::Mat &Ws, cv::Mat &X)
{
	cv::Mat c(EvectorsSA.cols, 1, CV_64FC1, par);
	cv::Mat b = b_mean + EvectorsSA*c;
	//writeMat(b, "xx.mat", "xx", false);
	cv::Mat b1 = b(cv::Range(0, Ws.rows), cv::Range::all()).clone();
	cv::Mat b2 = Ws.inv(cv::DECOMP_SVD)*b1;
	//writeMat(Ws_nlsq.inv(cv::DECOMP_SVD), "xx.mat", "xx", false);
	X = x_mean + EvectorsS*b2;

	//writeMat(X, "x1.mat", "x1", false);
	return;
}

 void costc(double *p, double *x, int n_par, int m_dat, void *data)
 {
	my_data_struct *D;
	D = (my_data_struct*)data;
	D->f(p, D->b_mean, D->x_mean, D->EvectorsSA, D->EvectorsS, D->Ws, D->x);
	
	for(int i=0;i<m_dat;i++){
		x[i] = D->x.at<double>(i,0);
	}
	
	return;
 }

void inter_iteration(cv::Mat &Itest, cv::Mat &pos, AAMShape &ShapeData, AAMTexture &Text, AAMAppearance &AppearanceData, 
				   AAMShapeAppearance &ShapeAppearanceData, AAMTform &tform, cv::Mat &R)
{

	double Eold = 10, w;
	cv::Mat c_old, c;
	AAMTform tform_old;

	// Go from ShapeAppearance Parameters to aligned shape coordinates
	cv::Mat b = ShapeAppearanceData.b_mean + ShapeAppearanceData.Evectors*C;
	cv::Mat b1 = b(cv::Range(0,ShapeAppearanceData.Ws.rows), cv::Range::all());
	b1 = ShapeAppearanceData.Ws.inv(cv::DECOMP_SVD)*b1;
	cv::Mat x = ShapeData.data_mean + ShapeData.Evectors*b1;

	AAM_align_data_inverse2D_tire(pos, pos, tform);

	// Sample the intensities
	cv::Mat g;
	AAM_Appearance2Vector2D(Itest, Text.textureSize, AppearanceData.k, 
		pos, AppearanceData.base_points, Text.F, g,  AppearanceData.ObjectPixels);
	g.convertTo(g, CV_64FC1);
	//writeMat(g_offset, "go.mat", "go", false);
	AAM_NormalizeAppearance2D(g);

	cv::Mat se = ShapeData.Evectors.t();
	// Go from intensities and shape back to ShapeAppearance Parameters
	b1 = ShapeAppearanceData.Ws *se * (x-ShapeData.data_mean);
	cv::Mat ae = AppearanceData.Evectors.t();
	cv::Mat b2 = ae * (g-AppearanceData.g_mean);
	concat_mat(b1, b2, b, 2);
	
	cv::Mat sae = ShapeAppearanceData.Evectors.t();
	cv::Mat c2 = sae*(b -ShapeAppearanceData.b_mean);

	// Go from ShapeAppearance Parameters back to model intensities
	b = ShapeAppearanceData.b_mean + ShapeAppearanceData.Evectors*c2;
	b2 = b(cv::Range(ShapeAppearanceData.Ws.rows, b.rows), cv::Range::all());
	cv::Mat g_model = AppearanceData.g_mean + AppearanceData.Evectors*b2;
	
	//Difference between model and real image intensities
	cv::Mat subG(g.rows, g.cols, CV_64FC1);

	for(int i=0;i<g.rows;i++){
		double *pG = g.ptr<double>(i);
		double *pGm = g_model.ptr<double>(i);
		double *psub = subG.ptr<double>(i);
		for(int j=0;j<g.cols;j++){
			psub[j]= pow(pG[j] - pGm[j], 2.0);
		}
	}
	cv::Mat err;
	cv::reduce(subG, err, 0, CV_REDUCE_SUM);

	double E = err.at<double>(0,0);
    
	// Go back to the old location of the previous itteration, if the
    // error was lower.
	if(E>Eold){
		// Not always go back if the error becomes higher, sometimes
		// stay on the higher error (like in simulated annealing)
		// Try a smaller stepsize
		w=w*0.9;
		c=c_old; tform=tform_old;
	}
    else{
		w=w*1.1;
		Eold=E;
	}

	//Store model /pose parameters for next itteration
	c_old=c;
	tform_old=tform;

	// Calculate the needed model parameter update using the 
	// search model matrix
	cv::Mat c_diff=R*(g-g_model);
	
	// Update the ShapeApppearance Parameters
	cv::Mat c_difft = c_diff(cv::Range(0, c_diff.rows-4), cv::Range::all());
	cv::Mat sa;cv::sqrt(ShapeAppearanceData.Evalues, sa);
	cv::multiply(c_difft, sa*w, c_difft);
	c=c+c_difft;
	
	// Update the Pose parameters
	tform.shift.at<double>(0,0) += c_diff.at<double>(c_diff.rows-3,0)*w;
	tform.shift.at<double>(0,1) += c_diff.at<double>(c_diff.rows-4,0)*w; 
}


void one_iteration(cv::Mat &im, cv::Mat &pos, AAMShape &ShapeData, AAMTexture &Text, AAMAppearance &AppearanceData, 
				   AAMShapeAppearance &ShapeAppearanceData, int num, double scale, AAMTform &tform, AMM_Model2D_Options &options)
{

	//AAMTrainingData TrainingData=Data[scale].T;
	// The image scaling of the scale-itteration
	double scaling=pow(2.0,-((double)scale-1.0));

	// Transform the image and coordinates offset, to the cuurent scale
	cv::Mat Itest;
	cv::resize(im, Itest, cv::Size(), scaling, scaling, 1);

	pos = (pos - 0.5)*scaling + 0.5;
	// From real image coordinates to -> algined coordinate
	AAMTform tforms;
	//std::vector<cv::Point2f> pos_align(Data[0].n);
	cv::Mat pos_align;
	AAM_align_data2D_tire(pos, ShapeData.MeanVertices, pos_align, tforms);

	cv::Mat X(num*2, 1, CV_64FC1);
	pos_align.col(0).copyTo(X(cv::Range(num,2*num),cv::Range::all()));
	pos_align.col(1).copyTo(X(cv::Range(0,num),cv::Range::all()));
	//writeMat(X, "xa.mat", "xa", false);
	// Start a new figure
	//show current test image, and initial contour
	//figure, imageplot(Itest); hold on; plot(pos(:,2),pos(:,1),'r.'); h=plot(1,1); 

	// Sample the image intensities
	cv::Mat g;
	AAM_Appearance2Vector2D(Itest, Text.textureSize, AppearanceData.k, 
		pos, AppearanceData.base_points, Text.F, g,  AppearanceData.ObjectPixels);
	g.convertTo(g, CV_64FC1);
	AAM_NormalizeAppearance2D(g);
	//writeMat(g, "go.mat", "go", false);
	// Go from image intesities and contour to ShapeAppearance parameters
	cv::Mat se = ShapeData.Evectors.t();
	cv::Mat b1 = ShapeAppearanceData.Ws * se * (X-ShapeData.data_mean);
	cv::Mat ae = AppearanceData.Evectors.t();
	cv::Mat b2 = ae * (g-AppearanceData.g_mean);
	cv::Mat b;//(b1.rows+b2.rows, b1.cols, CV_64FC1) ;
	concat_mat(b1, b2, b, 2);
	//writeMat(b, "ba.mat", "ba", false);

	cv::Mat sae = ShapeAppearanceData.Evectors.t();
	cv::Mat cin = sae*(b - ShapeAppearanceData.b_mean);
	cv::Mat X2=X.clone();
	//writeMat(cin, "cc.mat", "cc", false);
	
	cv::Mat maxcU(ShapeAppearanceData.Evalues.rows, 1, CV_64FC1);
	cv::Mat maxcL(ShapeAppearanceData.Evalues.rows, 1, CV_64FC1);
	for(int i1=0;i1<maxcU.rows;i1++){
		double *ptrU=maxcU.ptr<double>(i1);
		double *ptrL=maxcL.ptr<double>(i1);
		double *ptrE=ShapeAppearanceData.Evalues.ptr<double>(i1);
		for(int j1=0;j1<maxcU.cols;j1++){
			ptrU[j1] = sqrt(ptrE[j1]) * options.m;
			ptrL[j1] = -sqrt(ptrE[j1]) * options.m;
		}
	}

	//writeMat(X2, "xx.mat", "xx", false);
	//writeMat(maxcU, "mu.mat", "mu", false);
	//writeMat(maxcL, "ml.mat", "ml", false);
	//lsqnonlin optimization
	my_data_struct lm_data;

	lm_data.b_mean = ShapeAppearanceData.b_mean;
	lm_data.x_mean = ShapeData.data_mean;
	lm_data.EvectorsSA = ShapeAppearanceData.Evectors;
	lm_data.EvectorsS = ShapeData.Evectors;
	lm_data.Ws = ShapeAppearanceData.Ws;
	lm_data.f = cost_lmmin;
	lm_data.y = X;
	int n_par = ShapeAppearanceData.Evalues.rows;
	int m_dat = X.rows;
	lm_data.x = cv::Mat::zeros(m_dat, 1, CV_64FC1);
	
	double* par = (double*) cin.data;
	/*
	lm_status_struct status;
    lm_control_struct control = lm_control_double;
    control.verbosity = 3;
	lmmin(n_par, par, m_dat, (const void*) &lm_data, evaluate_dist, &control, &status);
	*/

	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	double* x_nsql = (double*) X2.data;
	//writeMat(X, "xx.mat", "xx", false);
	
	double* p = (double*) cin.data;
	opts[0]=LM_INIT_MU; opts[1]=1E-15; opts[2]=1E-15; opts[3]=1E-20;
	opts[4]=LM_DIFF_DELTA; // for finite difference Jacobian
	double *lb = (double*)maxcL.data;
	double *ub = (double*)maxcU.data;
	//cout << p[3] << endl;

	int ret=dlevmar_bc_dif(costc, p, x_nsql, n_par, m_dat, lb, ub, NULL, 100, opts, info, NULL, NULL, &lm_data); // with analytic Jacobian
	//int ret=dlevmar_dif(costc, p, x_nsql, n_par, m_dat, 100, NULL, info, NULL, NULL, (void*) &lm_data);

	//printf("Levenberg-Marquardt returned %d in %g iter, reason %g\nSolution: ", ret, info[5], info[6]);
	//for(int ii=0; ii<n_par; ++ii)
		//printf("%.7g ", p[ii]);
	//printf("\n\nMinimization info:\n");
	//for(int ii=0; ii<LM_INFO_SZ; ++ii)
	//	printf("%g ", info[ii]);
	//printf("\n");

	cv::Mat C(n_par, 1, CV_64FC1, par); //optimized constants
	cv::Mat c_old = C.clone(); 
	AAMTform tform_old=tform; 
	//Eold=inf; 
	//writeMat(C, "cc.mat", "cc", false);
	//writeMat(lm_data.x, "xx.mat", "xx", false);

	// Starting step size
	int w=1;
	// Search Itterations
	for(int i=0;i<options.nsearch;i++){

		// Stay within 3 (m) standard deviations               
		cv::Mat maxc=options.m*sqrt(ShapeAppearanceData.Evalues);
		c=max(min(c,maxc),- maxc);
	}
}

void ApplyModel2D(std::vector<AAM_ALL_DATA> &Data, AAMTexture &Text, cv::Mat &im, AAMTform &tformLarge, AMM_Model2D_Options &options)
{
	// We start at the coarse scale
	int scale=0; 
	int n = Data[0].n;
	double scaling=pow(2.0,-((double)scale-1.0));

	// Transform the coordinates to match the coarse scale
	AAMTform tform=tformLarge;

	//tform.offsetv=(tform.offsetv-0.5)*scaling+0.5;
	tform.shift=tform.shift*(scaling-1);

	// Get the PCA model for this scale
	AAMShapeAppearance ShapeAppearanceData=Data[scale].SA;
	AAMShape ShapeData=Data[scale].S;
	AAMAppearance AppearanceData=Data[scale].A;

	// Use the mean ShapeAppearance parameters to go get an initial contour
	cv::Mat b = ShapeAppearanceData.b_mean;
	cv::Mat b1 = b(cv::Range(0,ShapeAppearanceData.Ws.rows),cv::Range::all()).clone();
	b1= ShapeAppearanceData.Ws.inv(cv::DECOMP_SVD)*b1;
	// Initial (mean) aligned coordinates
	cv::Mat x = ShapeData.data_mean + ShapeData.Evectors*b1;
	//writeMat(x, "x1.mat", "x1", false);
	// The real image coordinates

	cv::Mat pos(n, 2, CV_64FC1);
	x(cv::Range(n,2*n),cv::Range::all()).copyTo(pos.col(0));
	x(cv::Range(0,n),cv::Range::all()).copyTo(pos.col(1));
	
	AAM_align_data_inverse2D_tire(pos, pos, tform);
	//cout << tform.shift << endl;
	//writeMat(pos, "p.mat", "p", false);

	// Loop through the 4 image size scales
	for (int scale=options.nscales; scale>0;scale--)
	{
		//Get the PCA model for this scale
		cv::Mat R=Data[scale-1].R;
		ShapeAppearanceData=Data[scale-1].SA;
		ShapeData=Data[scale-1].S;
		AppearanceData=Data[scale-1].A;

		one_iteration(im, pos, ShapeData, Text, AppearanceData, ShapeAppearanceData, n, scale, tform, options);

	}
	
}
